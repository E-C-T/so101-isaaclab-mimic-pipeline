#!/usr/bin/env python3
"""
Convert SO101 Isaac Mimic HDF5 demos into a LeRobot-style dataset.

Main design goals:
  1. Use a reference LeRobot meta/info.json as the schema authority.
  2. Preserve camera feature names, FPS, image shapes, state/action names.
  3. Resample Isaac-generated trajectories, e.g. 50 Hz -> 30 Hz.
  4. Export real camera frames when available.
  5. Keep observation.state/action compatible with the original SO101 seed dataset.

Example:

python /home/insol02/IH_ws/so101_IsaacLab/scripts/convert_so101_hdf5_to_lerobot.py \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_with_cameras.hdf5 \
  --out /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_lerobot_30hz \
  --reference-info /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place/meta/info.json \
  --source-fps 50 \
  --task "Pick up the cube and place it in the goal region." \
  --camera-mappings '{"wrist": "observation.images.wrist", "up": "observation.images.up"}'
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd


# -----------------------------
# JSON / JSONL helpers
# -----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# -----------------------------
# Reference schema helpers
# -----------------------------

def get_video_features(reference_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features = reference_info.get("features", {})
    return {
        key: feat
        for key, feat in features.items()
        if isinstance(feat, dict) and feat.get("dtype") == "video"
    }


def get_non_video_features(reference_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features = reference_info.get("features", {})
    return {
        key: feat
        for key, feat in features.items()
        if not (isinstance(feat, dict) and feat.get("dtype") == "video")
    }


def get_feature_shape(reference_info: Dict[str, Any], key: str) -> List[int]:
    features = reference_info.get("features", {})
    if key not in features:
        raise KeyError(f"Reference info.json has no feature key: {key}")
    shape = features[key].get("shape")
    if shape is None:
        raise ValueError(f"Reference feature {key!r} has no shape.")
    return list(shape)


def get_feature_names(reference_info: Dict[str, Any], key: str) -> Optional[List[str]]:
    features = reference_info.get("features", {})
    if key not in features:
        return None
    names = features[key].get("names")
    return names if isinstance(names, list) else None


def infer_default_camera_mappings(video_keys: Iterable[str]) -> Dict[str, str]:
    """
    Example:
      observation.images.wrist -> wrist
      observation.images.up    -> up
    """
    mappings: Dict[str, str] = {}
    for video_key in video_keys:
        short = video_key.split(".")[-1]
        mappings[short] = video_key
    return mappings


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort conversion of common HDF5/JSON scalar values to float."""
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.reshape(-1)[0]
        return float(value)
    except Exception:
        return None


def infer_source_fps_from_hdf5(h5_file: h5py.File) -> Optional[float]:
    """Try to infer source FPS from common HDF5 attrs or timestamp datasets.

    Many Isaac Lab HDF5 datasets do not store FPS explicitly. In that case,
    this returns None and the CLI fallback is used.
    """
    attr_candidates = [
        "fps",
        "source_fps",
        "control_fps",
        "env_fps",
        "hz",
        "frequency",
    ]

    # Check top-level attrs and /data attrs.
    for obj in [h5_file, h5_file.get("data", None)]:
        if obj is None:
            continue
        for key in attr_candidates:
            if key in obj.attrs:
                fps = _coerce_float(obj.attrs[key])
                if fps is not None and fps > 0:
                    return fps

    # Check first demo timestamp-like datasets.
    try:
        demos = sorted_demo_keys(h5_file)
        if not demos:
            return None
        demo = h5_file["data"][demos[0]]
    except Exception:
        return None

    timestamp_candidates = [
        "timestamp",
        "timestamps",
        "obs/timestamp",
        "obs/timestamps",
        "time",
        "obs/time",
    ]

    for path in timestamp_candidates:
        ds = h5_get_if_dataset(demo, path)
        if ds is None or len(ds) < 2:
            continue
        t = np.asarray(ds, dtype=np.float64).reshape(-1)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            continue
        median_dt = float(np.median(dt))
        if median_dt > 0:
            return 1.0 / median_dt

    return None


def choose_fps(
    *,
    reference_info: Dict[str, Any],
    h5_file: h5py.File,
    requested_source_fps: Optional[float],
    fallback_source_fps: Optional[float],
    requested_target_fps: Optional[float],
    allow_upsampling: bool,
) -> Tuple[float, float, float]:
    """Resolve source, reference, and target FPS.

    Policy:
      - Reference FPS comes from meta/info.json.
      - Source FPS comes from CLI, then HDF5 metadata/timestamps, then fallback.
      - Target FPS defaults to min(reference_fps, source_fps), avoiding upsampling.
      - If target_fps > source_fps, error unless --allow-upsampling is set.
    """
    reference_fps = _coerce_float(reference_info.get("fps", None))
    if reference_fps is None or reference_fps <= 0:
        raise ValueError("reference info.json must contain a positive 'fps' value.")

    inferred_source_fps = infer_source_fps_from_hdf5(h5_file)
    if requested_source_fps is not None:
        source_fps = float(requested_source_fps)
        source_origin = "CLI --source-fps"
    elif inferred_source_fps is not None:
        source_fps = float(inferred_source_fps)
        source_origin = "HDF5 metadata/timestamps"
    elif fallback_source_fps is not None:
        source_fps = float(fallback_source_fps)
        source_origin = "CLI --fallback-source-fps"
    else:
        raise ValueError(
            "Could not infer source FPS from input HDF5. Pass --source-fps explicitly "
            "or provide --fallback-source-fps."
        )

    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")

    if requested_target_fps is not None:
        target_fps = float(requested_target_fps)
        target_origin = "CLI --target-fps"
    else:
        target_fps = min(reference_fps, source_fps)
        target_origin = "min(reference fps, source fps)"

    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    if target_fps > source_fps and not allow_upsampling:
        raise ValueError(
            f"target_fps={target_fps} is higher than source_fps={source_fps}. "
            "This would upsample duplicate frames/actions. Lower --target-fps or pass --allow-upsampling."
        )

    print(f"[INFO] Reference FPS: {reference_fps:g}")
    print(f"[INFO] Source FPS: {source_fps:g} ({source_origin})")
    print(f"[INFO] Target FPS: {target_fps:g} ({target_origin})")

    return source_fps, target_fps, reference_fps


# -----------------------------
# HDF5 helpers
# -----------------------------

def h5_has_path(group: h5py.Group, path: str) -> bool:
    try:
        group[path]
        return True
    except KeyError:
        return False


def h5_get_if_dataset(group: h5py.Group, path: str) -> Optional[h5py.Dataset]:
    if not h5_has_path(group, path):
        return None

    node = group[path]
    if isinstance(node, h5py.Dataset):
        return node

    if isinstance(node, h5py.Group):
        # Common nested RGB names.
        for child_name in ("rgb", "rgba", "image", "images", "data"):
            if child_name in node and isinstance(node[child_name], h5py.Dataset):
                return node[child_name]

    return None


def find_dataset_by_candidates(group: h5py.Group, candidates: List[str]) -> Optional[Tuple[str, h5py.Dataset]]:
    for path in candidates:
        ds = h5_get_if_dataset(group, path)
        if ds is not None:
            return path, ds
    return None


def find_camera_dataset(demo_group: h5py.Group, internal_camera_name: str) -> Tuple[str, h5py.Dataset]:
    """
    Search common HDF5 camera layouts.

    Supported examples:
      camera_obs/wrist
      camera_obs/wrist/rgb
      obs/camera_obs/wrist
      obs/camera_obs/wrist/rgb
      obs/images/wrist
      obs/images/wrist/rgb
      observations/images/wrist
      observations/images/wrist/rgb
    """
    cam = internal_camera_name

    candidates = [
        f"camera_obs/{cam}",
        f"camera_obs/{cam}/rgb",
        f"camera_obs/{cam}/rgba",
        f"obs/camera_obs/{cam}",
        f"obs/camera_obs/{cam}/rgb",
        f"obs/camera_obs/{cam}/rgba",
        f"obs/images/{cam}",
        f"obs/images/{cam}/rgb",
        f"obs/images/{cam}/rgba",
        f"observations/images/{cam}",
        f"observations/images/{cam}/rgb",
        f"observations/images/{cam}/rgba",
        f"obs/{cam}",
        f"obs/{cam}/rgb",
        f"{cam}",
        f"{cam}/rgb",
    ]

    found = find_dataset_by_candidates(demo_group, candidates)
    if found is not None:
        return found

    # Fallback recursive search.
    matches: List[Tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        lname = name.lower()
        cam_l = cam.lower()

        # Prefer camera/image-like paths containing the internal camera name.
        if cam_l in lname and any(token in lname for token in ("camera", "image", "rgb", "rgba", "obs")):
            matches.append((name, obj))

    demo_group.visititems(visitor)

    if matches:
        # Prefer rgb datasets if several exist.
        matches = sorted(
            matches,
            key=lambda x: (
                0 if x[0].lower().endswith("/rgb") else 1,
                len(x[0]),
            ),
        )
        return matches[0]

    raise KeyError(
        f"Could not find camera dataset for internal camera {cam!r}. "
        f"Expected something like camera_obs/{cam}, camera_obs/{cam}/rgb, "
        f"obs/camera_obs/{cam}, or obs/camera_obs/{cam}/rgb."
    )


def find_first_existing_dataset(demo_group: h5py.Group, candidates: List[str]) -> Tuple[str, h5py.Dataset]:
    found = find_dataset_by_candidates(demo_group, candidates)
    if found is not None:
        return found

    raise KeyError(
        "Could not find any dataset among candidates:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def sorted_demo_keys(h5_file: h5py.File) -> List[str]:
    if "data" not in h5_file:
        raise KeyError("Expected HDF5 to contain a top-level 'data' group.")

    keys = list(h5_file["data"].keys())

    def sort_key(name: str) -> Tuple[int, str]:
        # Common Isaac Mimic layout: demo_0, demo_1, ...
        try:
            return (int(name.split("_")[-1]), name)
        except Exception:
            return (10**12, name)

    return sorted(keys, key=sort_key)


def parse_episode_indices(raw: Optional[str]) -> Optional[set[int]]:
    """Parse comma-separated episode indices like '0,1,7'."""
    if raw is None or str(raw).strip() == "":
        return None
    return {int(x.strip()) for x in str(raw).split(",") if x.strip()}


def filter_demo_keys(
    demos: List[str],
    *,
    episode_indices: Optional[set[int]] = None,
    only_with_cameras: bool = False,
    h5_file: Optional[h5py.File] = None,
    camera_mappings: Optional[Dict[str, str]] = None,
    max_episodes: Optional[int] = None,
) -> List[str]:
    """Filter sorted demo keys before conversion."""
    selected = demos

    if episode_indices is not None:
        selected = []
        for list_idx, demo_name in enumerate(demos):
            try:
                numeric_idx = int(demo_name.split("_")[-1])
            except Exception:
                numeric_idx = list_idx
            if numeric_idx in episode_indices or list_idx in episode_indices:
                selected.append(demo_name)

    if only_with_cameras:
        if h5_file is None or camera_mappings is None:
            raise ValueError("only_with_cameras requires h5_file and camera_mappings.")

        camera_selected = []
        for demo_name in selected:
            demo = h5_file["data"][demo_name]
            has_all = True
            for internal_cam_name in camera_mappings.keys():
                try:
                    find_camera_dataset(demo, internal_cam_name)
                except KeyError:
                    has_all = False
                    break
            if has_all:
                camera_selected.append(demo_name)
        selected = camera_selected

    if max_episodes is not None:
        selected = selected[:max_episodes]

    return selected


# -----------------------------
# Array helpers
# -----------------------------

def to_numpy(ds: h5py.Dataset, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(ds)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def ensure_2d_feature(arr: np.ndarray, expected_dim: int, key: str, allow_truncate: bool) -> np.ndarray:
    """
    Ensure arr is [T, expected_dim].
    """
    arr = np.asarray(arr)

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2:
        raise ValueError(f"{key} must be 2-D [T, D], got shape {arr.shape}")

    current_dim = arr.shape[1]

    if current_dim == expected_dim:
        return arr.astype(np.float32)

    if current_dim > expected_dim and allow_truncate:
        print(
            f"[WARN] {key} has dim {current_dim}, reference expects {expected_dim}. "
            f"Truncating to first {expected_dim} dims.",
            file=sys.stderr,
        )
        return arr[:, :expected_dim].astype(np.float32)

    raise ValueError(
        f"{key} has dim {current_dim}, but reference expects {expected_dim}. "
        f"Use --allow-truncate if you intentionally want to truncate larger arrays."
    )


def build_resample_indices(source_len: int, source_fps: float, target_fps: float) -> np.ndarray:
    """
    Resample by nearest source index.

    Example:
      source_fps = 50
      target_fps = 30

    idx[t] = round(t * source_fps / target_fps)
    """
    if source_len <= 0:
        return np.zeros((0,), dtype=np.int64)

    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"FPS values must be positive. Got source_fps={source_fps}, target_fps={target_fps}")

    duration_sec = source_len / source_fps
    target_len = int(math.floor(duration_sec * target_fps))

    # Ensure at least one frame if the source has at least one frame.
    target_len = max(target_len, 1)

    idx = np.round(np.arange(target_len, dtype=np.float64) * source_fps / target_fps).astype(np.int64)
    idx = np.clip(idx, 0, source_len - 1)
    return idx


def normalize_video_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Convert camera frames into uint8 [T, H, W, 3].

    Handles:
      [T, H, W, 3]
      [T, H, W, 4]
      [T, 3, H, W]
      [T, 4, H, W]
      [T, 1, H, W, C]
    """
    frames = np.asarray(frames)

    # Squeeze singleton env dimension if present: [T, 1, H, W, C]
    if frames.ndim == 5 and frames.shape[1] == 1:
        frames = frames[:, 0]

    if frames.ndim != 4:
        raise ValueError(f"Expected video frames to be 4-D [T,H,W,C] or [T,C,H,W], got {frames.shape}")

    # Channel-first -> channel-last.
    if frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))

    # Grayscale -> RGB.
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    # RGBA -> RGB.
    if frames.shape[-1] == 4:
        frames = frames[..., :3]

    if frames.shape[-1] != 3:
        raise ValueError(f"Expected 3 RGB channels after normalization, got shape {frames.shape}")

    # Float images.
    if np.issubdtype(frames.dtype, np.floating):
        # Common case: [0, 1]. If already [0, 255], clipping still works.
        if frames.max(initial=0.0) <= 1.5:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    # Integer images.
    elif frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    # Resize if needed.
    current_h, current_w = frames.shape[1], frames.shape[2]
    if (current_h, current_w) != (target_h, target_w):
        frames = resize_video_frames(frames, target_h, target_w)

    return frames


def resize_video_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize frames using PIL if available.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            f"Frames have shape {frames.shape[1:3]}, but reference expects {(target_h, target_w)}. "
            f"Install pillow or configure Isaac cameras to output the reference resolution."
        ) from exc

    resized = np.empty((frames.shape[0], target_h, target_w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img = img.resize((target_w, target_h), resample=Image.BILINEAR)
        resized[i] = np.asarray(img, dtype=np.uint8)
    return resized


# -----------------------------
# Path formatting helpers
# -----------------------------

def format_lerobot_path(
    template: str,
    *,
    chunk_index: int,
    file_index: int,
    episode_index: int,
    video_key: Optional[str] = None,
) -> Path:
    """
    Support both newer and older LeRobot-style templates.

    Newer attached reference:
      data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
      videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4

    Older dummy script:
      data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
      videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
    """
    kwargs = {
        "chunk_index": chunk_index,
        "file_index": file_index,
        "episode_index": episode_index,
        "episode_chunk": chunk_index,
        "video_key": video_key,
    }
    return Path(template.format(**kwargs))


def sanitize_video_key_for_path(video_key: str) -> str:
    # LeRobot templates often intentionally include dots in directory names.
    # Keep the key unchanged.
    return video_key


# -----------------------------
# Video writing
# -----------------------------

def ffmpeg_codec_from_reference(codec: Optional[str]) -> Optional[str]:
    """
    Map metadata codec names to ffmpeg encoder names when useful.
    """
    if codec is None:
        return None

    codec = codec.lower()

    # Reference info may say "av1", but ffmpeg usually wants an encoder name.
    if codec == "av1":
        return "libaom-av1"

    if codec in ("h264", "avc1"):
        return "libx264"

    if codec in ("h265", "hevc"):
        return "libx265"

    return codec


def write_video(
    path: Path,
    frames: np.ndarray,
    fps: float,
    requested_codec: Optional[str],
    pix_fmt: Optional[str],
) -> str:
    """
    Write MP4. Returns the metadata codec string actually intended.

    If the requested codec fails, fallback to libx264 because it is usually
    available and fast enough for debugging.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_codec = ffmpeg_codec_from_reference(requested_codec)
    kwargs: Dict[str, Any] = {"fps": fps}

    if ffmpeg_codec:
        kwargs["codec"] = ffmpeg_codec

    if pix_fmt:
        kwargs["pixelformat"] = pix_fmt

    try:
        iio.imwrite(path, frames, **kwargs)
        return requested_codec or "unknown"
    except Exception as exc:
        fallback_codec = "libx264"
        print(
            f"[WARN] Failed to write {path} with codec={ffmpeg_codec!r}: {exc}\n"
            f"[WARN] Falling back to codec={fallback_codec!r}. "
            f"If strict codec matching is required, install an ffmpeg build with the requested encoder.",
            file=sys.stderr,
        )
        iio.imwrite(path, frames, fps=fps, codec=fallback_codec, pixelformat="yuv420p")
        return "h264"


# -----------------------------
# Metadata generation
# -----------------------------

def build_modality_json_from_reference(reference_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal modality.json compatible with GR00T / VLA Foundry-style consumers.
    """
    features = reference_info.get("features", {})
    modality: Dict[str, Any] = {
        "state": {},
        "action": {},
        "video": {},
    }

    if "observation.state" in features:
        state_shape = features["observation.state"].get("shape", [0])
        state_dim = int(state_shape[0]) if state_shape else 0
        modality["state"]["observation.state"] = {"start": 0, "end": state_dim}

    if "action" in features:
        action_shape = features["action"].get("shape", [0])
        action_dim = int(action_shape[0]) if action_shape else 0
        modality["action"]["action"] = {"start": 0, "end": action_dim}

    for key, feat in features.items():
        if isinstance(feat, dict) and feat.get("dtype") == "video":
            modality["video"][key] = {}

    return modality


def update_info_for_generated_dataset(
    reference_info: Dict[str, Any],
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    target_fps: float,
    actual_video_codecs: Dict[str, str],
) -> Dict[str, Any]:
    """
    Copy the reference info.json but update dataset totals and FPS.
    """
    info = json.loads(json.dumps(reference_info))

    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(total_tasks)
    info["fps"] = float(target_fps)
    info["splits"] = {"train": f"0:{int(total_episodes)}"}

    # These are common LeRobot metadata fields. Preserve if present, update if useful.
    if "total_videos" in info:
        video_feature_count = len(get_video_features(info))
        info["total_videos"] = int(total_episodes * video_feature_count)

    if "total_chunks" in info:
        chunks_size = int(info.get("chunks_size", 1000))
        info["total_chunks"] = int(math.ceil(max(total_episodes, 1) / chunks_size))

    # Update per-video fps and codec metadata.
    for video_key, feat in info.get("features", {}).items():
        if not (isinstance(feat, dict) and feat.get("dtype") == "video"):
            continue

        vinfo = feat.setdefault("info", {})
        vinfo["video.fps"] = float(target_fps)

        if video_key in actual_video_codecs:
            # Preserve reference codec when successful; otherwise record fallback.
            vinfo["video.codec"] = actual_video_codecs[video_key]

    return info


# -----------------------------
# Conversion
# -----------------------------

def convert(args: argparse.Namespace) -> None:
    hdf5_path = Path(args.input_file).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    reference_info_path = Path(args.reference_info).expanduser().resolve()

    reference_info = read_json(reference_info_path)

    video_features = get_video_features(reference_info)
    if not video_features:
        raise ValueError("Reference info.json has no video features.")

    if args.camera_mappings is None:
        camera_mappings = infer_default_camera_mappings(video_features.keys())
    else:
        camera_mappings = json.loads(args.camera_mappings)

    # Validate mappings.
    for internal_name, video_key in camera_mappings.items():
        if video_key not in video_features:
            raise ValueError(
                f"Camera mapping {internal_name!r} -> {video_key!r} is invalid because "
                f"{video_key!r} does not exist in reference video features: {list(video_features.keys())}"
            )

    # Use only mapped video keys.
    mapped_video_keys = list(camera_mappings.values())

    state_dim = int(get_feature_shape(reference_info, "observation.state")[0])
    action_dim = int(get_feature_shape(reference_info, "action")[0])

    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"Output directory already exists: {out_root}. Use --overwrite to replace it.")

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    data_path_template = reference_info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    video_path_template = reference_info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    if args.vla_foundry_compat:
        # VLA Foundry tutorial converter currently discovers episodes from
        # episode_*.parquet names and image/video references from parquet columns.
        data_path_template = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        video_path_template = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    episodes_rows: List[Dict[str, Any]] = []
    tasks_rows: List[Dict[str, Any]] = [{"task_index": 0, "task": args.task}]

    total_frames = 0
    actual_video_codecs: Dict[str, str] = {}

    with h5py.File(hdf5_path, "r") as h5:
        source_fps, target_fps, reference_fps = choose_fps(
            reference_info=reference_info,
            h5_file=h5,
            requested_source_fps=args.source_fps,
            fallback_source_fps=args.fallback_source_fps,
            requested_target_fps=args.target_fps,
            allow_upsampling=args.allow_upsampling,
        )

        demos_all = sorted_demo_keys(h5)
        episode_indices = parse_episode_indices(args.episode_indices)
        demos = filter_demo_keys(
            demos_all,
            episode_indices=episode_indices,
            only_with_cameras=args.only_with_cameras,
            h5_file=h5,
            camera_mappings=camera_mappings,
            max_episodes=args.max_episodes,
        )

        print(f"[INFO] HDF5: {hdf5_path}")
        print(f"[INFO] Reference info.json: {reference_info_path}")
        print(f"[INFO] Output: {out_root}")
        print(f"[INFO] Demos total: {len(demos_all)}")
        print(f"[INFO] Demos selected: {len(demos)}")
        if episode_indices is not None:
            print(f"[INFO] Episode indices filter: {sorted(episode_indices)}")
        print(f"[INFO] Only with cameras: {args.only_with_cameras}")
        print(f"[INFO] Camera mappings: {camera_mappings}")

        if len(demos) == 0:
            raise RuntimeError("No demos selected for conversion. Check --episode-indices, --only-with-cameras, and camera mappings.")

        chunks_size = int(reference_info.get("chunks_size", 1000))

        for ep_idx, demo_name in enumerate(demos):
            demo = h5["data"][demo_name]

            # State: 6-D SO101 absolute measured joint positions.
            #
            # Important:
            #   obs/joint_pos is the Isaac Lab policy observation term and may be
            #   relative joint position because the task config uses mdp.joint_pos_rel.
            #   For LeRobot/VLA training we want observation.state to be in the same
            #   coordinate convention as the absolute joint-position action targets.
            #
            # Prefer the recorder/exported absolute articulation state when present.
            state_path, state_ds = find_first_existing_dataset(
                demo,
                [
                    "states/articulation/robot/joint_position",
                    "obs/robot_state/joint_pos",
                    "observations/joint_pos",
                    "joint_pos",
                    "obs/qpos",
                    "obs/joint_pos",
                ],
            )
            state_raw = to_numpy(state_ds, dtype=np.float32)
            state_raw = ensure_2d_feature(
                state_raw,
                expected_dim=state_dim,
                key=f"{demo_name}/{state_path}",
                allow_truncate=args.allow_truncate,
            )

            # Action: prefer processed_actions if present.
            action_path, action_ds = find_first_existing_dataset(
                demo,
                [
                    "processed_actions",
                    "actions",
                    "action",
                ],
            )
            action_raw = to_numpy(action_ds, dtype=np.float32)
            action_raw = ensure_2d_feature(
                action_raw,
                expected_dim=action_dim,
                key=f"{demo_name}/{action_path}",
                allow_truncate=args.allow_truncate,
            )

            # Camera raw arrays.
            camera_raw: Dict[str, np.ndarray] = {}
            camera_source_paths: Dict[str, str] = {}

            for internal_cam_name, video_key in camera_mappings.items():
                try:
                    cam_path, cam_ds = find_camera_dataset(demo, internal_cam_name)
                    frames_raw = to_numpy(cam_ds)
                    camera_raw[video_key] = frames_raw
                    camera_source_paths[video_key] = cam_path
                except KeyError:
                    if not args.allow_missing_cameras:
                        raise
                    print(
                        f"[WARN] Missing camera {internal_cam_name!r} for {demo_name}. "
                        f"Writing black dummy frames because --allow-missing-cameras was set.",
                        file=sys.stderr,
                    )

            # Align all source streams before resampling.
            source_lengths = [len(state_raw), len(action_raw)]
            source_lengths.extend(len(v) for v in camera_raw.values())
            source_T = int(min(source_lengths))

            if source_T <= 0:
                print(f"[WARN] Skipping empty demo {demo_name}.", file=sys.stderr)
                continue

            idx = build_resample_indices(source_T, source_fps=source_fps, target_fps=target_fps)
            target_T = len(idx)

            state = state_raw[:source_T][idx].astype(np.float32)
            action = action_raw[:source_T][idx].astype(np.float32)

            chunk_index = ep_idx // chunks_size
            file_index = ep_idx

            # Precompute relative video paths. These are also written into
            # parquet video feature columns when --write-image-columns is enabled.
            rel_video_paths_by_key: Dict[str, Path] = {}
            for video_key in mapped_video_keys:
                rel_video_paths_by_key[video_key] = format_lerobot_path(
                    video_path_template,
                    chunk_index=chunk_index,
                    file_index=file_index,
                    episode_index=ep_idx,
                    video_key=sanitize_video_key_for_path(video_key),
                )

            # Parquet path.
            rel_data_path = format_lerobot_path(
                data_path_template,
                chunk_index=chunk_index,
                file_index=file_index,
                episode_index=ep_idx,
            )
            abs_data_path = out_root / rel_data_path
            abs_data_path.parent.mkdir(parents=True, exist_ok=True)

            # Build dataframe according to reference non-video features.
            df_dict: Dict[str, Any] = {}

            non_video_features = get_non_video_features(reference_info)

            # Required standard fields.
            if "observation.state" in non_video_features:
                df_dict["observation.state"] = list(state)

            if "action" in non_video_features:
                df_dict["action"] = list(action)

            if "timestamp" in non_video_features:
                df_dict["timestamp"] = (np.arange(target_T, dtype=np.float64) / target_fps).astype(np.float64)

            if "frame_index" in non_video_features:
                df_dict["frame_index"] = np.arange(target_T, dtype=np.int64)

            if "episode_index" in non_video_features:
                df_dict["episode_index"] = np.full(target_T, ep_idx, dtype=np.int64)

            if "index" in non_video_features:
                df_dict["index"] = np.arange(total_frames, total_frames + target_T, dtype=np.int64)

            if "task_index" in non_video_features:
                df_dict["task_index"] = np.zeros(target_T, dtype=np.int64)

            # Video/image reference columns.
            #
            # Some LeRobot consumers infer cameras from meta/info.json + video_path.
            # VLA Foundry's current LeRobot preprocessor additionally expects image
            # columns to be present in the parquet itself. We therefore write one
            # column per mapped video key. Each row points to the episode-level MP4
            # and the corresponding timestamp inside that MP4.
            if args.write_image_columns:
                timestamps = (np.arange(target_T, dtype=np.float64) / target_fps).astype(np.float64)
                for video_key in mapped_video_keys:
                    rel_video_path = rel_video_paths_by_key[video_key].as_posix()
                    df_dict[video_key] = [
                        {
                            "path": rel_video_path,
                            "timestamp": float(ts),
                        }
                        for ts in timestamps
                    ]

            # Fill any other scalar non-video fields conservatively if reference contains them.
            for key, feat in non_video_features.items():
                if key in df_dict:
                    continue

                dtype = feat.get("dtype")
                shape = feat.get("shape", [1])
                dim = int(shape[0]) if shape else 1

                if dtype in ("int64", "int32"):
                    values = np.zeros((target_T,), dtype=np.int64)
                    df_dict[key] = values
                elif dtype in ("float32", "float64"):
                    if dim == 1:
                        values = np.zeros((target_T,), dtype=np.float32)
                    else:
                        values = [np.zeros((dim,), dtype=np.float32) for _ in range(target_T)]
                    df_dict[key] = values
                elif dtype == "bool":
                    df_dict[key] = np.zeros((target_T,), dtype=bool)
                else:
                    print(f"[WARN] Unknown non-video feature {key!r} with dtype {dtype!r}; filling with None.")
                    df_dict[key] = [None] * target_T

            df = pd.DataFrame(df_dict)
            df.to_parquet(abs_data_path)

            # Videos.
            for internal_cam_name, video_key in camera_mappings.items():
                video_feat = video_features[video_key]
                shape = video_feat.get("shape", [480, 640, 3])
                target_h = int(shape[0])
                target_w = int(shape[1])

                vinfo = video_feat.get("info", {})
                requested_codec = args.video_codec
                if requested_codec == "reference":
                    requested_codec = vinfo.get("video.codec", None)

                pix_fmt = args.video_pix_fmt
                if pix_fmt == "reference":
                    pix_fmt = vinfo.get("video.pix_fmt", "yuv420p")

                if video_key in camera_raw:
                    frames = camera_raw[video_key][:source_T][idx]
                    frames = normalize_video_frames(frames, target_h=target_h, target_w=target_w)
                else:
                    frames = np.zeros((target_T, target_h, target_w, 3), dtype=np.uint8)

                rel_video_path = rel_video_paths_by_key[video_key]
                abs_video_path = out_root / rel_video_path

                actual_codec = write_video(
                    abs_video_path,
                    frames,
                    fps=target_fps,
                    requested_codec=requested_codec,
                    pix_fmt=pix_fmt,
                )
                actual_video_codecs[video_key] = actual_codec

            episodes_rows.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [args.task],
                    "length": int(target_T),
                }
            )

            camera_debug = ", ".join(
                f"{vk}<-{camera_source_paths.get(vk, 'DUMMY')}"
                for vk in mapped_video_keys
            )

            print(
                f"[INFO] {demo_name} -> episode {ep_idx:06d}: "
                f"{source_T} frames @ {source_fps:g} Hz -> {target_T} frames @ {target_fps:g} Hz | "
                f"state={state.shape}, action={action.shape}, cameras=[{camera_debug}]"
            )

            total_frames += target_T

    # Metadata.
    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)

    generated_info = update_info_for_generated_dataset(
        reference_info,
        total_episodes=len(episodes_rows),
        total_frames=total_frames,
        total_tasks=1,
        target_fps=target_fps,
        actual_video_codecs=actual_video_codecs,
    )
    generated_info["data_path"] = data_path_template
    generated_info["video_path"] = video_path_template
    write_json(meta_dir / "info.json", generated_info)

    modality = build_modality_json_from_reference(generated_info)
    write_json(meta_dir / "modality.json", modality)

    print("")
    print(f"[DONE] Wrote LeRobot-style dataset to: {out_root}")
    print(f"[DONE] Episodes: {len(episodes_rows)}")
    print(f"[DONE] Frames: {total_frames}")
    print(f"[DONE] FPS: {target_fps}")
    print(f"[DONE] Video keys: {mapped_video_keys}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input_file",
        "--hdf5",
        dest="input_file",
        required=True,
        help="Input Isaac/Isaac Mimic HDF5 file. --hdf5 is kept as a backwards-compatible alias.",
    )
    ap.add_argument("--out", required=True, help="Output LeRobot dataset directory.")
    ap.add_argument(
        "--reference-info",
        default="/home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place/meta/info.json",
        help=(
            "Path to original seed LeRobot meta/info.json. This controls schema, camera keys, shapes, "
            "and default target FPS."
        ),
    )

    ap.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help=(
            "Source HDF5 sampling rate. If omitted, the script tries to infer from HDF5 metadata/timestamps, "
            "then falls back to --fallback-source-fps."
        ),
    )
    ap.add_argument(
        "--fallback-source-fps",
        type=float,
        default=50.0,
        help=(
            "Fallback source FPS when the input HDF5 does not store FPS. "
            "For your current Isaac setup, generated state datasets are usually 50 Hz."
        ),
    )
    ap.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Target export FPS. If omitted, uses min(reference info.json fps, source fps) to avoid upsampling."
        ),
    )
    ap.add_argument(
        "--allow-upsampling",
        action="store_true",
        help="Allow target_fps > source_fps. By default this is rejected to avoid duplicated frames/actions.",
    )

    ap.add_argument(
        "--task",
        default="Pick up the cube and place it in the goal region.",
        help="Task string to write into tasks.jsonl and episodes.jsonl.",
    )

    ap.add_argument(
        "--camera-mappings",
        default=None,
        help=(
            "JSON dict mapping HDF5 internal camera names to LeRobot video keys. "
            "Example: '{\"wrist\": \"observation.images.wrist\", \"up\": \"observation.images.up\"}'. "
            "If omitted, inferred from reference video keys by suffix."
        ),
    )

    ap.add_argument(
        "--video-codec",
        default="reference",
        help=(
            "Video codec to request. Use 'reference' to use reference info.json video.codec. "
            "For easier debugging, use 'libx264'."
        ),
    )
    ap.add_argument(
        "--video-pix-fmt",
        default="reference",
        help="Video pixel format. Use 'reference' to use reference info.json video.pix_fmt, usually yuv420p.",
    )

    ap.add_argument(
        "--allow-truncate",
        action="store_true",
        default=True,
        help="Allow truncating larger state/action arrays to the reference feature dimension.",
    )
    ap.add_argument(
        "--no-allow-truncate",
        dest="allow_truncate",
        action="store_false",
        help="Error instead of truncating larger state/action arrays.",
    )

    ap.add_argument(
        "--allow-missing-cameras",
        action="store_true",
        help="If a camera is missing, write black dummy frames instead of failing. Use only for smoke tests.",
    )

    ap.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit for debugging after other filters are applied.",
    )

    ap.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Optional comma-separated episode/demo indices to convert, e.g. '0,1,7'.",
    )

    ap.add_argument(
        "--only-with-cameras",
        action="store_true",
        help="Convert only demos that contain all requested cameras. Useful for partial smoke-test HDF5 files.",
    )

    ap.add_argument(
        "--write-image-columns",
        action="store_true",
        default=True,
        help=(
            "Write per-frame video reference columns such as observation.images.wrist into parquet. "
            "This is required by some VLA Foundry/LeRobot preprocessors."
        ),
    )
    ap.add_argument(
        "--no-write-image-columns",
        dest="write_image_columns",
        action="store_false",
        help="Do not write video reference columns into parquet.",
    )
    ap.add_argument(
        "--vla-foundry-compat",
        action="store_true",
        help=(
            "Use legacy/tutorial-compatible filenames: "
            "data/chunk-000/episode_000000.parquet and "
            "videos/chunk-000/<video_key>/episode_000000.mp4."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory if it already exists.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()