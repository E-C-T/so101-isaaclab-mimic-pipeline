#!/usr/bin/env python3
"""
Replay an Isaac HDF5 dataset in a camera-enabled SO101 replay environment and
append camera frames into a copied output HDF5.

Typical use:

/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/data/replay_dataset_with_cameras.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_state_only.hdf5 \
  --output_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_with_cameras.hdf5 \
  --headless --enable_cameras --overwrite

Expected output camera layout:

data/demo_X/camera_obs/wrist  [T, H, W, 3] uint8
data/demo_X/camera_obs/up     [T, H, W, 3] uint8
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Replay an Isaac HDF5 dataset with cameras and append camera_obs to a copied HDF5."
)
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0")
parser.add_argument("--input_file", type=str, required=True, help="Input Isaac HDF5 dataset.")
parser.add_argument("--output_file", type=str, required=True, help="Output HDF5 dataset with camera_obs added.")
parser.add_argument("--camera_names", type=str, default="wrist_camera,up_camera")
parser.add_argument("--camera_keys", type=str, default="wrist,up")
parser.add_argument("--max_episodes", type=int, default=None)
parser.add_argument("--episode_indices", type=str, default=None, help="Comma-separated episode indices, e.g. '0,3,7'.")
parser.add_argument("--reset_state_is_world", action="store_true", default=False)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--render_each_step", action="store_true", default=True)
parser.add_argument("--no_render_each_step", dest="render_each_step", action="store_false")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after AppLauncher
# -----------------------------------------------------------------------------
import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_episode_indices(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    elif x.ndim != 3:
        raise ValueError(f"Expected camera tensor [N,H,W,C] or [H,W,C], got {tuple(x.shape)}")
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA image with 3/4 channels, got {tuple(x.shape)}")
    if x.dtype != torch.uint8:
        if torch.is_floating_point(x) and torch.max(x) <= 1.5:
            x = x * 255.0
        x = torch.clamp(x, 0, 255).to(torch.uint8)
    return x.detach().cpu().numpy()


def _read_camera_rgb(env: ManagerBasedRLEnv, camera_name: str) -> np.ndarray:
    if camera_name not in env.scene.keys():
        raise KeyError(f"Camera {camera_name!r} not found in env.scene. Available keys: {list(env.scene.keys())}")
    cam = env.scene[camera_name]
    if "rgb" not in cam.data.output:
        raise KeyError(f"Camera {camera_name!r} has no 'rgb' output. Available: {list(cam.data.output.keys())}")
    return _to_uint8_rgb(cam.data.output["rgb"])


def _tensorize_action(action: Any, env: ManagerBasedRLEnv) -> torch.Tensor:
    if isinstance(action, torch.Tensor):
        action_tensor = action.to(device=env.device, dtype=torch.float32)
    else:
        action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32)
    action_tensor = action_tensor.flatten()
    batched_actions = torch.zeros((env.num_envs, action_tensor.numel()), device=env.device, dtype=torch.float32)
    batched_actions[0] = action_tensor
    return batched_actions


def _get_episode_initial_state(episode: EpisodeData):
    if hasattr(episode, "get_initial_state"):
        return episode.get_initial_state()
    return episode.data["initial_state"]


def _copy_input_to_output(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_file and output_file must be different.")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite.")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def _write_camera_obs(output_file: Path, episode_name: str, camera_frames: dict[str, list[np.ndarray]]) -> None:
    with h5py.File(output_file, "a") as h5:
        demo_path = f"data/{episode_name}"
        if demo_path not in h5:
            raise KeyError(f"Output HDF5 missing {demo_path}")
        demo = h5[demo_path]
        if "camera_obs" in demo:
            del demo["camera_obs"]
        cam_group = demo.create_group("camera_obs")
        for camera_key, frames_list in camera_frames.items():
            if len(frames_list) == 0:
                raise RuntimeError(f"No frames recorded for camera key {camera_key!r}")
            frames = np.stack(frames_list, axis=0).astype(np.uint8)
            cam_group.create_dataset(
                camera_key,
                data=frames,
                compression="gzip",
                compression_opts=4,
                chunks=(1, frames.shape[1], frames.shape[2], frames.shape[3]),
            )


def _episode_filter_names(all_names: list[str], episode_indices: set[int] | None, max_episodes: int | None) -> list[str]:
    if episode_indices is not None:
        selected = []
        for idx, name in enumerate(all_names):
            try:
                numeric = int(name.split("_")[-1])
            except Exception:
                numeric = idx
            if numeric in episode_indices or idx in episode_indices:
                selected.append(name)
        return selected
    if max_episodes is not None:
        return all_names[:max_episodes]
    return all_names


def main() -> int:
    input_path = Path(args_cli.input_file).expanduser().resolve()
    output_path = Path(args_cli.output_file).expanduser().resolve()
    camera_names = _parse_csv(args_cli.camera_names)
    camera_keys = _parse_csv(args_cli.camera_keys)
    if len(camera_names) != len(camera_keys):
        raise ValueError(f"--camera_names and --camera_keys must have same length. Got {camera_names} and {camera_keys}")

    if not getattr(args_cli, "enable_cameras", False):
        print("[WARN] Recommended to pass AppLauncher --enable_cameras for camera replay.")

    print("=" * 80)
    print("Replay dataset with cameras")
    print("=" * 80)
    print(f"task              : {args_cli.task}")
    print(f"input_file        : {input_path}")
    print(f"output_file       : {output_path}")
    print(f"camera_names      : {camera_names}")
    print(f"camera_keys       : {camera_keys}")
    print(f"headless          : {args_cli.headless}")
    print(f"enable_cameras    : {getattr(args_cli, 'enable_cameras', None)}")
    print(f"reset_to relative : {not args_cli.reset_state_is_world}")
    print("=" * 80)

    _copy_input_to_output(input_path, output_path, overwrite=args_cli.overwrite)

    env_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name
    env_cfg.enable_camera_sensors = True
    env_cfg.terminations = None

    env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(str(input_path))

    all_episode_names = list(dataset_file_handler.get_episode_names())
    episode_indices = _parse_episode_indices(args_cli.episode_indices)
    episode_names = _episode_filter_names(all_episode_names, episode_indices, args_cli.max_episodes)

    print(f"[INFO] Input episodes: {len(all_episode_names)}")
    print(f"[INFO] Episodes to process: {len(episode_names)}")

    processed = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for ep_i, episode_name in enumerate(episode_names):
            print(f"[{ep_i + 1}/{len(episode_names)}] Replaying {episode_name} with cameras...")
            episode = dataset_file_handler.load_episode(episode_name, env.device)
            initial_state = _get_episode_initial_state(episode)

            env.reset()
            env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
            env.reset_to(initial_state, env_ids, is_relative=not args_cli.reset_state_is_world)
            env.sim.render()

            camera_frames: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
            step_idx = 0
            while True:
                action = episode.get_next_action()
                if action is None:
                    break
                env.step(_tensorize_action(action, env))
                if args_cli.render_each_step:
                    env.sim.render()
                for camera_name, camera_key in zip(camera_names, camera_keys):
                    camera_frames[camera_key].append(_read_camera_rgb(env, camera_name))
                step_idx += 1

            _write_camera_obs(output_path, episode_name, camera_frames)
            frame_counts = {k: len(v) for k, v in camera_frames.items()}
            shapes = {k: np.asarray(v[0]).shape if v else None for k, v in camera_frames.items()}
            print(f"  steps={step_idx}, camera_frames={frame_counts}, frame_shapes={shapes}")
            processed += 1

    env.close()
    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"processed episodes : {processed}")
    print(f"output_file        : {output_path}")
    print("=" * 80)
    return processed


if __name__ == "__main__":
    try:
        count = main()
    finally:
        simulation_app.close()
    raise SystemExit(0 if count >= 0 else 1)