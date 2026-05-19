from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Replay HDF5 demos headless and save only successful episodes.")
parser.add_argument("--task", type=str, required=True, help="Gym task name, e.g. Isaac-SO-ARM101-Cube-Replay-v0")
parser.add_argument("--input-hdf5", type=str, required=True, help="Input replay HDF5 file")
parser.add_argument("--output-hdf5", type=str, required=True, help="Output HDF5 file with successful episodes only")
parser.add_argument("--num-envs", type=int, default=1, help="Use 1 for deterministic filtering")
parser.add_argument(
    "--check-mode",
    type=str,
    default="ever_success",
    choices=["ever_success", "final_success"],
    help=(
        "ever_success: keep episode if cube ever enters goal region during replay\n"
        "final_success: keep episode only if cube is in goal region at final state"
    ),
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Optional list of episode indices to replay/filter. "
        "Example: --select_episodes 0 3 7"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after AppLauncher
# -----------------------------------------------------------------------------
import isaac_so_arm101.tasks  # noqa: F401
import isaac_so_arm101.tasks.cube_replay_i4h.success as replay_success
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clone_default_rigid_object_state(default_rigid_object_state: dict, env_id: int) -> dict:
    rigid_object = {}
    for asset_name, asset_state_dict in default_rigid_object_state.items():
        rigid_object[asset_name] = {}
        for state_name, state_tensor in asset_state_dict.items():
            rigid_object[asset_name][state_name] = state_tensor[env_id : env_id + 1].clone().detach()
    return rigid_object


def inject_default_rigid_object_state(initial_state: dict, default_rigid_object_state: dict | None, env_id: int) -> dict:
    if "rigid_object" in initial_state:
        return initial_state
    if default_rigid_object_state is None:
        return initial_state

    initial_state["rigid_object"] = clone_default_rigid_object_state(default_rigid_object_state, env_id)
    return initial_state


def copy_demo_group(src_demo_group: h5py.Group, dst_data_group: h5py.Group, new_demo_name: str) -> None:
    src_demo_group.file.copy(src_demo_group, dst_data_group, name=new_demo_name)


def get_object_debug_state(env, asset_name: str = "object") -> dict:
    obj = env.scene[asset_name]
    root_state = obj.data.root_state_w[0].detach().cpu()

    pos = root_state[0:3].numpy().tolist()
    quat = root_state[3:7].numpy().tolist()
    lin_vel = root_state[7:10].numpy().tolist()

    return {
        "pos": pos,
        "quat": quat,
        "lin_vel": lin_vel,
    }

def demo_has_rigid_object_initial_state(demo_group: h5py.Group) -> bool:
    return (
        "initial_state" in demo_group
        and "rigid_object" in demo_group["initial_state"]
    )

def write_rigid_object_initial_state_to_demo(
    demo_group: h5py.Group,
    default_rigid_object_state: dict,
    env_id: int = 0,
) -> None:
    """
    Ensure demo_group contains initial_state/rigid_object populated from the env default state.
    """
    if "initial_state" not in demo_group:
        init_group = demo_group.create_group("initial_state")
    else:
        init_group = demo_group["initial_state"]

    if "rigid_object" in init_group:
        return

    rigid_group = init_group.create_group("rigid_object")

    for asset_name, asset_state_dict in default_rigid_object_state.items():
        asset_group = rigid_group.create_group(asset_name)
        for state_name, state_tensor in asset_state_dict.items():
            value = state_tensor[env_id : env_id + 1].detach().cpu().numpy()
            asset_group.create_dataset(state_name, data=value, compression="gzip")

def get_cfg_robot_root_pose_tensor(env) -> torch.Tensor:
    """Save calibrated task-cfg robot root pose after env.reset()."""
    robot = env.scene["robot"]
    return robot.data.root_state_w[0:1, :7].clone().detach()


def apply_robot_root_pose(env, root_pose_w: torch.Tensor) -> None:
    """Force robot root pose while preserving current joint state."""
    robot = env.scene["robot"]

    root_pose_w = root_pose_w.to(device=env.device, dtype=torch.float32)
    if root_pose_w.ndim == 1:
        root_pose_w = root_pose_w.unsqueeze(0)

    root_velocity_w = torch.zeros(
        (root_pose_w.shape[0], 6),
        device=env.device,
        dtype=torch.float32,
    )

    robot.write_root_pose_to_sim(root_pose_w)
    robot.write_root_velocity_to_sim(root_velocity_w)
    robot.reset()
    env.sim.render()


def overwrite_robot_root_pose_in_initial_state(
    initial_state: dict,
    root_pose_w: torch.Tensor,
    env,
) -> dict:
    """Keep reset_to() valid while preventing HDF5 root_pose from overriding calibrated cfg pose."""
    if "articulation" not in initial_state:
        return initial_state
    if "robot" not in initial_state["articulation"]:
        return initial_state

    robot_state = initial_state["articulation"]["robot"]
    robot_state["root_pose"] = root_pose_w.clone().detach().to(device=env.device, dtype=torch.float32)
    robot_state["root_velocity"] = torch.zeros((1, 6), device=env.device, dtype=torch.float32)
    return initial_state


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    input_path = Path(args_cli.input_hdf5)
    output_path = Path(args_cli.output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env_cfg.seed = 0

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    cfg_robot_root_pose_w = get_cfg_robot_root_pose_tensor(env)
    print("[DEBUG] cfg robot root pose saved:", cfg_robot_root_pose_w[0].detach().cpu().numpy())

    # Cache default rigid object state once after reset
    runtime_state_after_reset = env.scene.get_state(is_relative=True)
    default_rigid_object_state = runtime_state_after_reset.get("rigid_object", None)

    # Open source dataset
    dataset_handler = HDF5DatasetFileHandler()
    dataset_handler.open(str(input_path))
    episode_names = sorted(
        list(dataset_handler.get_episode_names()),
        key=lambda name: int(name.split("_")[-1]),
    )

    if args_cli.select_episodes is not None:
        selected_set = set(args_cli.select_episodes)
        episode_names = [
            name for name in episode_names
            if int(name.split("_")[-1]) in selected_set
        ]

    num_episodes = len(episode_names)

    # Read logical goal bounds from env cfg if present; otherwise hard-code fallback
    if hasattr(env_cfg, "goal_region"):
        success_params = dict(env_cfg.goal_region)
    else:
        success_params = {
            "asset_name": "object",
            "x_min": 0.11,
            "x_max": 0.19,
            "y_min": 0.11,
            "y_max": 0.19,
            "z_min": 0.0,
            "z_max": 0.08,
            "max_lin_vel": 0.15,
        }

    success_params.setdefault("asset_name", "object")

    print("\nFiltering configuration:")
    print(json.dumps(success_params, indent=2))
    print(f"check_mode = {args_cli.check_mode}")
    print(f"num_episodes = {num_episodes}")
    print(f"task = {args_cli.task}\n")

    successful_episode_indices: list[int] = []

    # Replay and evaluate
    with torch.inference_mode():
        for local_idx, ep_name in enumerate(episode_names):
            ep_idx = int(ep_name.split("_")[-1])
            print(f"[Episode {ep_idx}] Loading {ep_name}")

            episode_data = dataset_handler.load_episode(ep_name, env.device)
            initial_state = episode_data.get_initial_state()
            initial_state = inject_default_rigid_object_state(initial_state, default_rigid_object_state, env_id=0)

            # Preserve calibrated robot root pose from task cfg.
            # This prevents old/root000 HDF5 files from making the I4H robot float again.
            initial_state = overwrite_robot_root_pose_in_initial_state(
                initial_state=initial_state,
                root_pose_w=cfg_robot_root_pose_w,
                env=env,
            )

            env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
            apply_robot_root_pose(env, cfg_robot_root_pose_w)
            env.sim.render()

            ever_success = False
            step_idx = 0

            while True:
                action = episode_data.get_next_action()
                if action is None:
                    break

                actions = torch.zeros(env.action_space.shape, device=env.device)
                actions[0] = action
                env.step(actions)

                success_now = replay_success.object_in_aabb_success(
                    env,
                    **success_params,
                    debug_print=False,
                )[0].item()

                if success_now and not ever_success:
                    obj_debug = get_object_debug_state(env, asset_name=success_params.get("asset_name", "object"))
                    print(
                        f"[Episode {ep_idx}] Success region entered at step {step_idx}. "
                        f"object_pos={obj_debug['pos']} "
                        f"object_lin_vel={obj_debug['lin_vel']}"
                    )
                    ever_success = True

                step_idx += 1

            # Final detailed debug print
            final_success = replay_success.object_in_aabb_success(
                env,
                **success_params,
                debug_print=True,
            )[0].item()

            obj_debug = get_object_debug_state(env, asset_name=success_params.get("asset_name", "object"))

            if args_cli.check_mode == "ever_success":
                is_success = ever_success
            else:
                is_success = final_success

            if is_success:
                print(f"[Episode {ep_idx}] Success!")
                successful_episode_indices.append(ep_idx)
            else:
                print(f"[Episode {ep_idx}] Failure")

            print(
                f"[Episode {ep_idx}] "
                f"ever_success={ever_success} "
                f"final_success={final_success} "
                f"selected={is_success}"
            )
            print(
                f"[Episode {ep_idx}] "
                f"object_pos={obj_debug['pos']} "
                f"object_quat={obj_debug['quat']} "
                f"object_lin_vel={obj_debug['lin_vel']}\n"
            )

    dataset_handler.close()
    env.close()

    # Copy only successful demos into a new HDF5
    with h5py.File(input_path, "r") as src_f, h5py.File(output_path, "w") as dst_f:
        dst_data = dst_f.create_group("data")
        src_data = src_f["data"]

        for k, v in src_data.attrs.items():
            dst_data.attrs[k] = v
        dst_data.attrs["total"] = len(successful_episode_indices)
        dst_data.attrs["filter_seed"] = 0
        dst_data.attrs["filter_check_mode"] = args_cli.check_mode
        dst_data.attrs["goal_region_json"] = json.dumps(success_params)
        dst_data.attrs["rigid_object_injected_if_missing"] = True

        new_demo_idx = 0
        injected_count = 0

        for old_ep_idx in successful_episode_indices:
            old_demo_name = f"demo_{old_ep_idx}"
            new_demo_name = f"demo_{new_demo_idx}"

            # Copy the demo first
            copy_demo_group(src_data[old_demo_name], dst_data, new_demo_name)

            # Then patch in rigid_object initial state if missing
            new_demo_group = dst_data[new_demo_name]
            if not demo_has_rigid_object_initial_state(new_demo_group):
                if default_rigid_object_state is None:
                    raise RuntimeError(
                        f"Demo {new_demo_name} is missing initial_state/rigid_object, "
                        "and no default rigid object state was available from the env."
                    )
                write_rigid_object_initial_state_to_demo(
                    new_demo_group,
                    default_rigid_object_state=default_rigid_object_state,
                    env_id=0,
                )
                injected_count += 1
                print(f"[Patch] Injected initial_state/rigid_object into {new_demo_name}")

            new_demo_idx += 1
    print(f"Injected rigid_object initial state into {injected_count} demos")

    print("==================================================")
    print(f"Successful episodes: {successful_episode_indices}")
    print(f"Kept {len(successful_episode_indices)} / {num_episodes}")
    print(f"Wrote filtered dataset to: {output_path}")
    print("==================================================")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()