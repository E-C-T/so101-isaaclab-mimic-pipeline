from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# ./isaaclab.sh -p /home/insol02/IH_ws/so101_IsaacLab/src/isaac_so_arm101/scripts/calibrate_cube_scene.py \
#   --task Isaac-SO-ARM101-Cube-Replay-v0 \
#   --dataset_file /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place_ep0_final.hdf5 \
#   --episode_index 0 \
#   --sample_index 300 \
#   --mode action_step \
#   --step_size 10
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Interactive SO-ARM101 cube replay calibration tool.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Cube-Replay-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--dataset_file",
    type=str,
    required=True,
    help="Converted Isaac HDF5 replay file.",
)
parser.add_argument(
    "--episode_index",
    type=int,
    default=0,
    help="Episode index inside the converted HDF5 file.",
)
parser.add_argument(
    "--sample_index",
    type=int,
    default=0,
    help="Initial sample index to load.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="action_step",
    choices=["direct_pose", "action_step"],
    help=(
        "direct_pose: directly set robot to obs/joint_pos[sample_index]\n"
        "action_step: reset to initial state, then apply env.step() sequentially up to sample_index"
    ),
)

parser.add_argument(
    "--zero_qpos",
    action="store_true",
    help="Ignore dataset qpos/actions and set all robot joints to zero after env reset.",
)

parser.add_argument(
    "--save_camera_debug",
    action="store_true",
    help="Save wrist/up camera RGB frames when applying a sample.",
)
parser.add_argument(
    "--camera_debug_dir",
    type=str,
    default="/home/insol02/IH_ws/so101_IsaacLab/datasets/debug_camera_views",
    help="Directory where camera debug PNGs are saved.",
)

parser.add_argument(
    "--step_size",
    type=int,
    default=10,
    help="How many samples to move when pressing J/K.",
)


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after AppLauncher
# -----------------------------------------------------------------------------
import h5py
import gymnasium as gym
import numpy as np
import torch

import isaac_so_arm101.tasks  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from pathlib import Path
import imageio.v3 as iio

# -----------------------------------------------------------------------------
# Global interaction state
# -----------------------------------------------------------------------------
STEP_FORWARD = False
STEP_BACKWARD = False
RESET_TO_ZERO = False
PRINT_INFO = False
TOGGLE_MODE = False
STEP_SIZE_UP = False
STEP_SIZE_DOWN = False


def cb_step_forward():
    global STEP_FORWARD
    STEP_FORWARD = True


def cb_step_backward():
    global STEP_BACKWARD
    STEP_BACKWARD = True


def cb_reset_zero():
    global RESET_TO_ZERO
    RESET_TO_ZERO = True


def cb_print_info():
    global PRINT_INFO
    PRINT_INFO = True


def cb_toggle_mode():
    global TOGGLE_MODE
    TOGGLE_MODE = True


def cb_step_size_up():
    global STEP_SIZE_UP
    STEP_SIZE_UP = True


def cb_step_size_down():
    global STEP_SIZE_DOWN
    STEP_SIZE_DOWN = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_episode_arrays(hdf5_path: str, episode_index: int):
    with h5py.File(hdf5_path, "r") as f:
        demo_name = f"demo_{episode_index}"
        if demo_name not in f["data"]:
            raise KeyError(f"{demo_name} not found in dataset.")

        demo = f["data"][demo_name]
        qpos = np.asarray(demo["obs"]["joint_pos"][:], dtype=np.float32)
        actions = np.asarray(demo["actions"][:], dtype=np.float32)

    return qpos, actions


def load_episode_data_object(dataset_file: str, episode_index: int, device):
    handler = HDF5DatasetFileHandler()
    handler.open(dataset_file)
    episode_names = list(handler.get_episode_names())
    episode_data = handler.load_episode(episode_names[episode_index], device)
    handler.close()
    return episode_data


def inject_default_rigid_object_state(initial_state: dict, env, env_id: int) -> dict:
    if "rigid_object" in initial_state:
        return initial_state

    runtime_state = env.scene.get_state(is_relative=True)
    if "rigid_object" not in runtime_state:
        return initial_state

    initial_state["rigid_object"] = {}
    for asset_name, asset_state_dict in runtime_state["rigid_object"].items():
        initial_state["rigid_object"][asset_name] = {}
        for state_name, state_tensor in asset_state_dict.items():
            initial_state["rigid_object"][asset_name][state_name] = (
                state_tensor[env_id : env_id + 1].clone().detach()
            )
    return initial_state

def get_cfg_robot_root_pose_tensor(env) -> torch.Tensor:
    """Return calibrated I4H/URDF root pose from the live env after cfg reset.

    Shape: (1, 7), order: x y z qw qx qy qz
    """
    robot = env.scene["robot"]
    return robot.data.root_state_w[0:1, :7].clone().detach()


def apply_robot_root_pose(env, root_pose_w: torch.Tensor) -> None:
    """Force robot root pose while preserving current joint state."""
    robot = env.scene["robot"]

    root_pose_w = root_pose_w.to(device=env.device, dtype=torch.float32)
    if root_pose_w.ndim == 1:
        root_pose_w = root_pose_w.unsqueeze(0)

    root_velocity_w = torch.zeros((root_pose_w.shape[0], 6), device=env.device, dtype=torch.float32)

    robot.write_root_pose_to_sim(root_pose_w)
    robot.write_root_velocity_to_sim(root_velocity_w)
    robot.reset()
    env.sim.render()
    

def set_robot_to_qpos(env, qpos_vec: np.ndarray, root_pose_w: torch.Tensor | None = None) -> None:
    robot = env.scene["robot"]
    qpos = torch.tensor(qpos_vec, dtype=torch.float32, device=env.device).unsqueeze(0)
    qvel = torch.zeros_like(qpos)

    if root_pose_w is not None:
        apply_robot_root_pose(env, root_pose_w)

    robot.write_joint_state_to_sim(qpos, qvel)

    if root_pose_w is not None:
        apply_robot_root_pose(env, root_pose_w)

    robot.reset()
    env.sim.render()


def print_robot_joint_info(env) -> None:
    robot = env.scene["robot"]
    print("\nRobot joint names:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i}: {name}")

    current_q = robot.data.joint_pos[0].detach().cpu()
    print("\nCurrent robot joint positions [rad]:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {name:20s} {current_q[i].item(): .6f}")


def print_scene_info(env, sample_index: int, mode: str, step_size: int) -> None:
    robot = env.scene["robot"]

    print("\n==================================================")
    print(f"sample_index: {sample_index}")
    print(f"mode: {mode}")
    print(f"step_size: {step_size}")

    print("\nRobot root pose world [x y z qw qx qy qz]:")
    print(robot.data.root_state_w[0, :7].detach().cpu().numpy())

    if "object" in env.scene.keys():
        obj = env.scene["object"]
        print("\nObject root pose world [x y z qw qx qy qz]:")
        print(obj.data.root_state_w[0, :7].detach().cpu().numpy())

    if "ee_frame" in env.scene.keys():
        ee = env.scene["ee_frame"]
        if hasattr(ee.data, "target_pos_w"):
            print("\nEE target world position [x y z]:")
            print(ee.data.target_pos_w[0, 0].detach().cpu().numpy())

    print_robot_joint_info(env)

    print("\nRobot body world poses [x y z qw qx qy qz]:")
    body_state = robot.data.body_state_w[0, :, :7].detach().cpu().numpy()
    for i, name in enumerate(robot.body_names):
        print(f"  {i:02d} {name:24s} {body_state[i]}")

    if "object" in env.scene.keys():
        obj = env.scene["object"]
        print("\nObject CURRENT world pose [x y z qw qx qy qz]:")
        print(obj.data.root_state_w[0, :7].detach().cpu().numpy())
    print("==================================================\n")


def get_camera_rgb_uint8(env, camera_name: str) -> np.ndarray:
    cam = env.scene[camera_name]
    rgb = cam.data.output["rgb"]

    # Expected common shape: [num_envs, H, W, 3] or [H, W, 3]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()

    if rgb.ndim == 4:
        rgb = rgb[0]

    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            if rgb.max() <= 1.5:
                rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


def save_camera_debug_frames(env, sample_index: int, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Force renderer/camera update.
    env.sim.render()

    for scene_cam_name, short_name in [
        ("wrist_camera", "wrist"),
        ("up_camera", "up"),
    ]:
        if scene_cam_name not in env.scene.keys():
            print(f"[CAMERA DEBUG] {scene_cam_name} not found in env.scene")
            continue

        try:
            rgb = get_camera_rgb_uint8(env, scene_cam_name)
            save_path = out_path / f"{short_name}_{sample_index:06d}.png"
            iio.imwrite(save_path, rgb)
            print(f"[CAMERA DEBUG] wrote {save_path} shape={rgb.shape}")
        except Exception as exc:
            print(f"[CAMERA DEBUG] failed for {scene_cam_name}: {exc}")


def reset_env_to_episode_start(env, episode_data, root_pose_w: torch.Tensor | None = None) -> None:
    initial_state = episode_data.get_initial_state()
    initial_state = inject_default_rigid_object_state(initial_state, env, env_id=0)

    # Preserve calibrated cfg robot root pose while still allowing reset_to()
    # to restore joint_position/joint_velocity from the episode.
    if root_pose_w is not None:
        if "articulation" in initial_state and "robot" in initial_state["articulation"]:
            robot_state = initial_state["articulation"]["robot"]

            # Isaac Lab reset_to() requires these keys to exist.
            # Do not pop/delete them.
            robot_state["root_pose"] = root_pose_w.clone().detach()
            robot_state["root_velocity"] = torch.zeros(
                (1, 6),
                device=env.device,
                dtype=torch.float32,
            )

    env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)

    if root_pose_w is not None:
        apply_robot_root_pose(env, root_pose_w)

    env.sim.render()


def apply_sample(
    env,
    episode_data,
    qpos_all: np.ndarray,
    actions_all: np.ndarray,
    sample_index: int,
    mode: str,
    root_pose_w: torch.Tensor | None = None,
) -> None:
    sample_index = max(0, min(sample_index, len(qpos_all) - 1))

    if mode == "direct_pose":
        set_robot_to_qpos(env, qpos_all[sample_index], root_pose_w=root_pose_w)
        return

    if mode == "action_step":
        # Deterministic: always rebuild from episode start.
        reset_env_to_episode_start(env, episode_data, root_pose_w=root_pose_w)

        for t in range(sample_index + 1):
            action = torch.zeros(env.action_space.shape, device=env.device)
            action[0] = torch.tensor(actions_all[t], dtype=torch.float32, device=env.device)

            env.step(action)
            if root_pose_w is not None:
                apply_robot_root_pose(env, root_pose_w)

        env.sim.render()
        return

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env_cfg.seed = 0

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    print("[DEBUG] scene keys:")
    for k in env.scene.keys():
        print(" ", k)

    cfg_robot_root_pose_w = get_cfg_robot_root_pose_tensor(env)
    print("[DEBUG] cfg robot root pose saved:", cfg_robot_root_pose_w[0].detach().cpu().numpy())

    if args_cli.zero_qpos:
        robot = env.scene["robot"]
        qpos = torch.zeros((1, robot.num_joints), device=env.device)
        qvel = torch.zeros_like(qpos)
        robot.write_joint_state_to_sim(qpos, qvel)
        robot.reset()
        env.sim.render()
        print_scene_info(env, sample_index=0, mode="zero_qpos", step_size=1)

        while simulation_app.is_running() and not simulation_app.is_exiting():
            env.sim.render()
        return

    qpos_all, actions_all = load_episode_arrays(args_cli.dataset_file, args_cli.episode_index)
    episode_data = load_episode_data_object(args_cli.dataset_file, args_cli.episode_index, env.device)

    sample_index = max(0, min(args_cli.sample_index, len(qpos_all) - 1))
    mode = args_cli.mode
    step_size = max(1, args_cli.step_size)

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))
    keyboard.add_callback("K", cb_step_forward)
    keyboard.add_callback("J", cb_step_backward)
    keyboard.add_callback("R", cb_reset_zero)
    keyboard.add_callback("P", cb_print_info)
    keyboard.add_callback("M", cb_toggle_mode)
    keyboard.add_callback("U", cb_step_size_down)
    keyboard.add_callback("I", cb_step_size_up)
    keyboard.reset()

    apply_sample(env, episode_data, qpos_all, actions_all, sample_index, mode,root_pose_w=cfg_robot_root_pose_w)
    if args_cli.save_camera_debug:
        save_camera_debug_frames(env, sample_index, args_cli.camera_debug_dir)
    print_scene_info(env, sample_index, mode, step_size)

    print(
        "\nInteractive calibration controls:\n"
        "  K : step forward by step_size samples\n"
        "  J : step backward by step_size samples\n"
        "  R : reset to sample 0\n"
        "  M : toggle mode direct_pose <-> action_step\n"
        "  P : print robot/object/EE info\n"
        "  U : decrease step_size\n"
        "  I : increase step_size\n"
        "\nMode meanings:\n"
        "  direct_pose : directly set robot to obs/joint_pos[sample_index]\n"
        "  action_step : reset to episode start, then apply env.step() up to sample_index\n"
        "\nImportant:\n"
        "  Do NOT use the Isaac Sim GUI pause button with this script.\n"
        "  This script already acts like a manual pause/step controller.\n"
    )

    global STEP_FORWARD, STEP_BACKWARD, RESET_TO_ZERO, PRINT_INFO, TOGGLE_MODE, STEP_SIZE_UP, STEP_SIZE_DOWN

    try:
        while simulation_app.is_running() and not simulation_app.is_exiting():
            if STEP_SIZE_UP:
                step_size = min(step_size * 2, max(1, len(qpos_all) - 1))
                print(f"[STEP_SIZE] -> {step_size}")
                STEP_SIZE_UP = False

            if STEP_SIZE_DOWN:
                step_size = max(1, step_size // 2)
                print(f"[STEP_SIZE] -> {step_size}")
                STEP_SIZE_DOWN = False

            if TOGGLE_MODE:
                mode = "action_step" if mode == "direct_pose" else "direct_pose"
                print(f"[MODE] switched to: {mode}")
                apply_sample(env, episode_data, qpos_all, actions_all, sample_index, mode, root_pose_w=cfg_robot_root_pose_w)
                if args_cli.save_camera_debug:
                    save_camera_debug_frames(env, sample_index, args_cli.camera_debug_dir)
                print_scene_info(env, sample_index, mode, step_size)
                TOGGLE_MODE = False

            if RESET_TO_ZERO:
                sample_index = 0
                print("[RESET] sample_index -> 0")
                apply_sample(env, episode_data, qpos_all, actions_all, sample_index, mode, root_pose_w=cfg_robot_root_pose_w)
                if args_cli.save_camera_debug:
                    save_camera_debug_frames(env, sample_index, args_cli.camera_debug_dir)
                print_scene_info(env, sample_index, mode, step_size)
                RESET_TO_ZERO = False

            if STEP_FORWARD:
                old_index = sample_index
                sample_index = min(sample_index + step_size, len(qpos_all) - 1)
                print(f"[STEP] {old_index} -> {sample_index}")
                apply_sample(env, episode_data, qpos_all, actions_all, sample_index, mode, root_pose_w=cfg_robot_root_pose_w)
                if args_cli.save_camera_debug:
                    save_camera_debug_frames(env, sample_index, args_cli.camera_debug_dir)
                print_scene_info(env, sample_index, mode, step_size)
                STEP_FORWARD = False

            if STEP_BACKWARD:
                old_index = sample_index
                sample_index = max(sample_index - step_size, 0)
                print(f"[STEP] {old_index} -> {sample_index}")
                apply_sample(env, episode_data, qpos_all, actions_all, sample_index, mode, root_pose_w=cfg_robot_root_pose_w)
                if args_cli.save_camera_debug:
                    save_camera_debug_frames(env, sample_index, args_cli.camera_debug_dir)
                print_scene_info(env, sample_index, mode, step_size)
                STEP_BACKWARD = False

            if PRINT_INFO:
                print_scene_info(env, sample_index, mode, step_size)
                PRINT_INFO = False

            env.sim.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()