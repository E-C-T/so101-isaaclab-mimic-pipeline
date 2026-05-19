from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def to_numpy(x: Any) -> np.ndarray:
    """Convert torch/numpy/scalar-like object to numpy array."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def parse_vec3(raw: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Parse x,y,z string."""
    if raw is None or raw.strip() == "":
        return default

    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated values x,y,z, got: {raw}")

    return vals[0], vals[1], vals[2]


def parse_quat_wxyz(
    raw: str | None,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Parse quaternion string in Isaac/PhysX convention qw,qx,qy,qz."""
    if raw is None or raw.strip() == "":
        return default

    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != 4:
        raise ValueError(f"Expected 4 comma-separated values qw,qx,qy,qz, got: {raw}")

    return vals[0], vals[1], vals[2], vals[3]


def parse_joint_offsets_deg(raw: str | None) -> dict[int, float]:
    """
    Parse comma-separated joint offset mapping.

    Example:
        "4:180,3:-10"

    SO101 joint order:
        0 shoulder_pan
        1 shoulder_lift
        2 elbow_flex
        3 wrist_flex
        4 wrist_roll
        5 gripper
    """
    if raw is None or raw.strip() == "":
        return {}

    result: dict[int, float] = {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    for part in parts:
        if ":" not in part:
            raise ValueError(
                f"Invalid joint offset entry '{part}'. Expected format like '4:180' or '3:-10'."
            )

        idx_str, val_str = part.split(":", maxsplit=1)
        joint_idx = int(idx_str)
        offset_deg = float(val_str)

        if joint_idx < 0 or joint_idx > 5:
            raise ValueError(f"Joint index out of SO101 range [0, 5]: {joint_idx}")

        result[joint_idx] = offset_deg

    return result


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------


def extract_image_dict(step: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Extract LeRobot image observations from one step.

    Input LeRobot keys are expected to look like:
        observation.images.up
        observation.images.wrist
        observation.images.front
        etc.

    Output camera names are the suffixes:
        up
        wrist
        front
    """
    image_dict: dict[str, np.ndarray] = {}

    for key, value in step.items():
        if not key.startswith("observation.images."):
            continue

        cam_name = key.split(".")[-1]
        arr = to_numpy(value)

        # LeRobot image tensors may be CHW. Convert to HWC.
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        # Convert float image to uint8 if needed.
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.5:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Drop alpha if present.
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]

        image_dict[cam_name] = arr

    return image_dict


# -----------------------------------------------------------------------------
# Joint mapping helpers
# -----------------------------------------------------------------------------


def apply_joint_offsets_rad(
    qpos_rad: np.ndarray,
    shoulder_pan_offset_deg: float,
    wrist_roll_offset_deg: float,
    extra_joint_offsets_deg: dict[int, float],
) -> np.ndarray:
    """
    Apply dataset-to-sim convention corrections in radians.

    SO101 ordering:
        0 shoulder_pan
        1 shoulder_lift
        2 elbow_flex
        3 wrist_flex
        4 wrist_roll
        5 gripper
    """
    qpos_rad = qpos_rad.copy()

    if qpos_rad.ndim != 2:
        raise ValueError(f"Expected qpos_rad shape (T, D), got {qpos_rad.shape}")

    if qpos_rad.shape[1] < 6:
        raise ValueError(f"Expected at least 6 SO101 joints, got shape {qpos_rad.shape}")

    if abs(shoulder_pan_offset_deg) > 1e-12:
        qpos_rad[:, 0] += np.deg2rad(shoulder_pan_offset_deg)

    if abs(wrist_roll_offset_deg) > 1e-12:
        qpos_rad[:, 4] += np.deg2rad(wrist_roll_offset_deg)

    for joint_idx, offset_deg in extra_joint_offsets_deg.items():
        qpos_rad[:, joint_idx] += np.deg2rad(offset_deg)

    return qpos_rad


# -----------------------------------------------------------------------------
# HDF5 writing
# -----------------------------------------------------------------------------


def write_episode(
    data_group: h5py.Group,
    new_demo_idx: int,
    source_episode_index: int,
    qpos_list: list[np.ndarray],
    image_buffer: dict[str, list[np.ndarray]],
    env_name: str,
    shoulder_pan_offset_deg: float,
    wrist_roll_offset_deg: float,
    extra_joint_offsets_deg: dict[int, float],
    root_pos: tuple[float, float, float],
    root_rot_wxyz: tuple[float, float, float, float],
) -> None:
    """Write one LeRobot episode into Isaac Lab replay HDF5 format."""
    if len(qpos_list) == 0:
        return

    ep_group = data_group.create_group(f"demo_{new_demo_idx}")
    ep_group.attrs["num_samples"] = len(qpos_list)
    ep_group.attrs["success"] = True
    ep_group.attrs["source_episode_index"] = source_episode_index

    # Traceability attrs.
    ep_group.attrs["env_name"] = env_name
    ep_group.attrs["shoulder_pan_offset_deg"] = shoulder_pan_offset_deg
    ep_group.attrs["wrist_roll_offset_deg"] = wrist_roll_offset_deg
    ep_group.attrs["joint_offsets_deg_json"] = json.dumps(extra_joint_offsets_deg)
    ep_group.attrs["root_pos_xyz"] = json.dumps(list(root_pos))
    ep_group.attrs["root_rot_wxyz"] = json.dumps(list(root_rot_wxyz))

    init_group = ep_group.create_group("initial_state")
    art_group = init_group.create_group("articulation")
    robot_group = art_group.create_group("robot")

    # LeRobot observation.state is assumed to be in degrees.
    qpos_deg = np.stack(qpos_list, axis=0).astype(np.float32)
    qpos = np.deg2rad(qpos_deg).astype(np.float32)

    qpos = apply_joint_offsets_rad(
        qpos_rad=qpos,
        shoulder_pan_offset_deg=shoulder_pan_offset_deg,
        wrist_roll_offset_deg=wrist_roll_offset_deg,
        extra_joint_offsets_deg=extra_joint_offsets_deg,
    ).astype(np.float32)

    qpos0 = qpos[0:1]
    qvel = np.zeros_like(qpos, dtype=np.float32)
    qvel0 = qvel[0:1]

    # Replay/debug uses absolute joint-position-like actions.
    actions = qpos.copy().astype(np.float32)

    root_pose = np.array(
        [
            [
                root_pos[0],
                root_pos[1],
                root_pos[2],
                root_rot_wxyz[0],
                root_rot_wxyz[1],
                root_rot_wxyz[2],
                root_rot_wxyz[3],
            ]
        ],
        dtype=np.float32,
    )

    root_velocity = np.zeros((1, 6), dtype=np.float32)

    # Top-level actions.
    ep_group.create_dataset("actions", data=actions, compression="gzip")

    # Initial robot state.
    robot_group.create_dataset("joint_position", data=qpos0, compression="gzip")
    robot_group.create_dataset("joint_velocity", data=qvel0, compression="gzip")
    robot_group.create_dataset("root_pose", data=root_pose, compression="gzip")
    robot_group.create_dataset("root_velocity", data=root_velocity, compression="gzip")

    # Observations.
    obs_group = ep_group.create_group("obs")
    obs_group.create_dataset("joint_pos", data=qpos, compression="gzip")
    obs_group.create_dataset("joint_vel", data=qvel, compression="gzip")
    obs_group.create_dataset("actions", data=actions, compression="gzip")

    # Optional images.
    #
    # This stores:
    #   obs/images/<camera_name>
    #
    # Example:
    #   obs/images/up
    #   obs/images/wrist
    #
    # Your later Isaac-HDF5 -> LeRobot converter can map:
    #   up    -> observation.images.up
    #   wrist -> observation.images.wrist
    if len(image_buffer) > 0:
        obs_images = obs_group.create_group("images")
        for cam_name, frame_list in image_buffer.items():
            if len(frame_list) == 0:
                continue
            frames = np.stack(frame_list, axis=0).astype(np.uint8)
            obs_images.create_dataset(cam_name, data=frames, compression="gzip")

    data_group.file.flush()

    del qpos_deg, qpos, qpos0, qvel, qvel0, actions, root_pose, root_velocity
    gc.collect()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LeRobot SO101 dataset to Isaac Lab HDF5 replay format. "
            "Supports joint convention offsets and configurable robot root pose "
            "for URDF/I4H/USD scene variants."
        )
    )

    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=None)

    parser.add_argument(
        "--shoulder-pan-offset-deg",
        type=float,
        default=0.0,
        help="Constant offset in degrees applied to joint index 0 (shoulder_pan).",
    )
    parser.add_argument(
        "--wrist-roll-offset-deg",
        type=float,
        default=0.0,
        help="Constant offset in degrees applied to joint index 4 (wrist_roll).",
    )
    parser.add_argument(
        "--joint-offsets-deg",
        type=str,
        default="",
        help=(
            'Optional comma-separated "joint_idx:offset_deg" pairs, e.g. '
            '"4:180,3:-10". Applied after named offsets.'
        ),
    )

    parser.add_argument(
        "--root-pos",
        type=str,
        default="0.0,0.0,0.0",
        help="Robot root position as x,y,z. Default: 0,0,0.",
    )
    parser.add_argument(
        "--root-rot-wxyz",
        type=str,
        default="1.0,0.0,0.0,0.0",
        help="Robot root quaternion as qw,qx,qy,qz. Default: identity.",
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not copy LeRobot image observations into the output HDF5.",
    )

    args = parser.parse_args()

    extra_joint_offsets_deg = parse_joint_offsets_deg(args.joint_offsets_deg)
    root_pos = parse_vec3(args.root_pos, default=(0.0, 0.0, 0.0))
    root_rot_wxyz = parse_quat_wxyz(args.root_rot_wxyz, default=(1.0, 0.0, 0.0, 0.0))

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("LeRobot -> Isaac HDF5 SO101 conversion")
    print("==================================================")
    print(f"repo_id                  : {args.repo_id}")
    print(f"root                     : {args.root}")
    print(f"output_file              : {output_file}")
    print(f"env_name                 : {args.env_name}")
    print(f"episode_index            : {args.episode_index}")
    print(f"shoulder_pan_offset_deg  : {args.shoulder_pan_offset_deg}")
    print(f"wrist_roll_offset_deg    : {args.wrist_roll_offset_deg}")
    print(f"extra_joint_offsets_deg  : {extra_joint_offsets_deg}")
    print(f"root_pos                 : {root_pos}")
    print(f"root_rot_wxyz            : {root_rot_wxyz}")
    print(f"copy_images              : {not args.no_images}")
    print("==================================================")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        video_backend="pyav",
    )

    first_step = dataset[0]
    if "action" not in first_step:
        raise KeyError("Expected key 'action' in LeRobot step.")
    if "observation.state" not in first_step:
        raise KeyError("Expected key 'observation.state' in LeRobot step.")
    if "episode_index" not in first_step:
        raise KeyError("Expected key 'episode_index' in LeRobot step.")

    with h5py.File(output_file, "w") as f:
        data_group = f.create_group("data")

        data_group.attrs["env_args"] = json.dumps({"env_name": args.env_name, "type": 2})
        data_group.attrs["env_name"] = args.env_name
        data_group.attrs["shoulder_pan_offset_deg"] = args.shoulder_pan_offset_deg
        data_group.attrs["wrist_roll_offset_deg"] = args.wrist_roll_offset_deg
        data_group.attrs["joint_offsets_deg_json"] = json.dumps(extra_joint_offsets_deg)
        data_group.attrs["root_pos_xyz"] = json.dumps(list(root_pos))
        data_group.attrs["root_rot_wxyz"] = json.dumps(list(root_rot_wxyz))
        data_group.attrs["copy_images"] = not args.no_images

        current_episode_index: int | None = None
        current_qpos_list: list[np.ndarray] = []
        current_image_buffer: dict[str, list[np.ndarray]] = {}
        written_count = 0

        def should_keep_episode(ep_idx: int) -> bool:
            return args.episode_index is None or ep_idx == args.episode_index

        for i in tqdm(range(len(dataset)), desc="streaming steps"):
            step = dataset[i]
            ep_idx = int(to_numpy(step["episode_index"]).item())

            # Early exit for single episode.
            if args.episode_index is not None and current_episode_index is not None:
                if current_episode_index == args.episode_index and ep_idx > args.episode_index:
                    break

            if current_episode_index is None:
                current_episode_index = ep_idx

            # Episode boundary.
            if ep_idx != current_episode_index:
                if should_keep_episode(current_episode_index):
                    write_episode(
                        data_group=data_group,
                        new_demo_idx=written_count,
                        source_episode_index=current_episode_index,
                        qpos_list=current_qpos_list,
                        image_buffer=current_image_buffer,
                        env_name=args.env_name,
                        shoulder_pan_offset_deg=args.shoulder_pan_offset_deg,
                        wrist_roll_offset_deg=args.wrist_roll_offset_deg,
                        extra_joint_offsets_deg=extra_joint_offsets_deg,
                        root_pos=root_pos,
                        root_rot_wxyz=root_rot_wxyz,
                    )
                    written_count += 1

                current_episode_index = ep_idx
                current_qpos_list = []
                current_image_buffer = {}
                gc.collect()

            if should_keep_episode(ep_idx):
                qpos_deg = to_numpy(step["observation.state"]).reshape(-1).astype(np.float32)
                current_qpos_list.append(qpos_deg)

                if not args.no_images:
                    image_dict = extract_image_dict(step)
                    if len(image_dict) > 0:
                        for cam_name, frame in image_dict.items():
                            current_image_buffer.setdefault(cam_name, []).append(frame)

        # Flush final episode.
        if current_episode_index is not None and should_keep_episode(current_episode_index):
            write_episode(
                data_group=data_group,
                new_demo_idx=written_count,
                source_episode_index=current_episode_index,
                qpos_list=current_qpos_list,
                image_buffer=current_image_buffer,
                env_name=args.env_name,
                shoulder_pan_offset_deg=args.shoulder_pan_offset_deg,
                wrist_roll_offset_deg=args.wrist_roll_offset_deg,
                extra_joint_offsets_deg=extra_joint_offsets_deg,
                root_pos=root_pos,
                root_rot_wxyz=root_rot_wxyz,
            )
            written_count += 1

        data_group.attrs["total"] = written_count

    print("==================================================")
    print(f"[DONE] wrote {output_file}")
    print(f"[DONE] total episodes written: {written_count}")
    print("==================================================")


if __name__ == "__main__":
    main()