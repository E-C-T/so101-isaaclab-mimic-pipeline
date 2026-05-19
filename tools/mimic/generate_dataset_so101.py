# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SO-101 / I4H-safe Isaac Lab Mimic dataset generation script.

Main changes from the stock/custom generate script:

1. Quiet by default. Use --debug for detailed reset/scene prints.
2. Camera recording is opt-in with --record_cameras and expects a camera-enabled task.
3. For I4H SO-ARM USD tasks, the calibrated robot root pose is forced into:
   - env_cfg.scene.robot.init_state before env construction
   - every env.reset()
   - every env.reset_to(...)

This prevents Isaac Mimic generation resets from falling back to the wrong
identity robot root pose.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab Mimic environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, default=None, help="Number of demos to be generated.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to instantiate.")
parser.add_argument("--input_file", type=str, required=True, help="Annotated source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="Output HDF5 file for generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="Pause after every subtask during generation for debugging. Useful with GUI/rendering.",
)
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio before app launch.")
parser.add_argument("--use_skillgen", action="store_true", default=False, help="Use SkillGen / motion planners.")
parser.add_argument("--debug", action="store_true", default=False, help="Print detailed generation/reset diagnostics.")

# Camera recording is intentionally separate from AppLauncher --enable_cameras.
parser.add_argument(
    "--record_cameras",
    action="store_true",
    default=False,
    help=(
        "Record camera_obs/wrist and camera_obs/up during Mimic generation. "
        "Requires using a camera-enabled task and passing AppLauncher --enable_cameras. "
        "Recommended workflow is usually to leave this off, then replay successful generated demos with cameras."
    ),
)
parser.add_argument("--wrist_camera_name", type=str, default="wrist_camera", help="Scene key for wrist camera sensor.")
parser.add_argument("--up_camera_name", type=str, default="up_camera", help="Scene key for up/room camera sensor.")

# Root-pose safety for I4H USD asset.
parser.add_argument(
    "--no_force_cfg_robot_root",
    action="store_true",
    default=False,
    help="Disable forcing the calibrated cfg robot root pose during env reset/reset_to.",
)
parser.add_argument(
    "--robot-root-pos",
    type=str,
    default=None,
    help=(
        "Optional robot root position x,y,z to force into env cfg and resets. "
        "Use --robot-root-pos='-0.02079,-0.01576,-0.03248' for the calibrated I4H SO101 root."
    ),
)
parser.add_argument(
    "--robot-root-rot-wxyz",
    type=str,
    default=None,
    help=(
        "Optional robot root quaternion qw,qx,qy,qz to force into env cfg and resets. "
        "Use --robot-root-rot-wxyz='0.707,0.0,0.0,0.707' for the calibrated I4H SO101 root."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the version installed by IsaacLab.
    import pinocchio  # noqa: F401

# Launch Isaac Sim first.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after AppLauncher
# -----------------------------------------------------------------------------
import asyncio
import inspect
import logging
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Parsing / root-pose helpers
# -----------------------------------------------------------------------------
def _parse_vec(raw: str | None, expected_len: int, name: str) -> tuple[float, ...] | None:
    if raw is None or str(raw).strip() == "":
        return None
    vals = tuple(float(x.strip()) for x in str(raw).split(","))
    if len(vals) != expected_len:
        raise ValueError(f"{name} expected {expected_len} comma-separated values, got: {raw}")
    return vals


def _task_looks_like_i4h(task_name: str | None) -> bool:
    return task_name is not None and "I4H" in task_name


def _resolve_forced_root_pose(env_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """Return local robot root pose to force, or None if disabled."""
    if args_cli.no_force_cfg_robot_root:
        return None

    user_pos = _parse_vec(args_cli.robot_root_pos, 3, "--robot-root-pos")
    user_rot = _parse_vec(args_cli.robot_root_rot_wxyz, 4, "--robot-root-rot-wxyz")

    if user_pos is not None or user_rot is not None:
        if user_pos is None or user_rot is None:
            raise ValueError("Provide both --robot-root-pos and --robot-root-rot-wxyz, or neither.")
        return user_pos, user_rot  # type: ignore[return-value]

    # Default protection for the I4H SO-ARM USD task family.
    if _task_looks_like_i4h(env_name):
        return (-0.02079, -0.01576, -0.03248), (0.707, 0.0, 0.0, 0.707)

    return None


def _force_env_cfg_robot_root(env_cfg: Any, root_pos: tuple[float, float, float], root_rot: tuple[float, float, float, float]) -> None:
    """Force root pose in env cfg before env construction."""
    if not hasattr(env_cfg, "scene") or not hasattr(env_cfg.scene, "robot"):
        raise AttributeError("env_cfg.scene.robot does not exist; cannot force robot root pose.")
    if not hasattr(env_cfg.scene.robot, "init_state"):
        raise AttributeError("env_cfg.scene.robot.init_state does not exist; cannot force robot root pose.")

    env_cfg.scene.robot.init_state.pos = tuple(root_pos)
    env_cfg.scene.robot.init_state.rot = tuple(root_rot)


def _root_pose_local_tensor(
    env: ManagerBasedRLMimicEnv,
    root_pos: tuple[float, float, float],
    root_rot: tuple[float, float, float, float],
    rows: int,
) -> torch.Tensor:
    pose = torch.tensor([*root_pos, *root_rot], device=env.device, dtype=torch.float32).view(1, 7)
    return pose.repeat(rows, 1)


def _root_pose_world_tensor(
    env: ManagerBasedRLMimicEnv,
    root_pos: tuple[float, float, float],
    root_rot: tuple[float, float, float, float],
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build world-frame root pose for write_root_pose_to_sim.

    Isaac Lab reset_to(..., is_relative=True) uses env-local scene states. Direct
    write_root_pose_to_sim expects world-frame pose, so add env origins.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    local = _root_pose_local_tensor(env, root_pos, root_rot, rows=len(env_ids))

    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        local[:, 0:3] += env_origins[env_ids]

    return local


def _apply_robot_root_pose_world(
    env: ManagerBasedRLMimicEnv,
    root_pos: tuple[float, float, float],
    root_rot: tuple[float, float, float, float],
    env_ids: torch.Tensor | None = None,
    render: bool = False,
) -> None:
    robot = env.scene["robot"]
    root_pose_w = _root_pose_world_tensor(env, root_pos, root_rot, env_ids=env_ids)
    root_velocity_w = torch.zeros((root_pose_w.shape[0], 6), device=env.device, dtype=torch.float32)

    # Most Isaac Lab versions accept env_ids for partial writes. Fall back to full write if needed.
    try:
        robot.write_root_pose_to_sim(root_pose_w, env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_velocity_w, env_ids=env_ids)
    except TypeError:
        if env_ids is not None and len(env_ids) != env.num_envs:
            # Conservative fallback: write all envs.
            root_pose_w = _root_pose_world_tensor(env, root_pos, root_rot, env_ids=None)
            root_velocity_w = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
        robot.write_root_pose_to_sim(root_pose_w)
        robot.write_root_velocity_to_sim(root_velocity_w)

    robot.reset()
    if render:
        env.sim.render()


def _patch_initial_state_robot_root(
    env: ManagerBasedRLMimicEnv,
    initial_state: dict,
    root_pos: tuple[float, float, float],
    root_rot: tuple[float, float, float, float],
    env_ids: torch.Tensor | None,
) -> dict:
    """Overwrite HDF5/datagen reset state root_pose so reset_to never uses identity root."""
    if not isinstance(initial_state, dict):
        return initial_state
    if "articulation" not in initial_state:
        return initial_state
    if "robot" not in initial_state["articulation"]:
        return initial_state

    robot_state = initial_state["articulation"]["robot"]

    # reset_to state is local when is_relative=True. Use repeated local calibrated root.
    if env_ids is None:
        rows = env.num_envs
    else:
        rows = int(len(env_ids))

    if "root_pose" in robot_state and isinstance(robot_state["root_pose"], torch.Tensor):
        rows = int(robot_state["root_pose"].shape[0])

    robot_state["root_pose"] = _root_pose_local_tensor(env, root_pos, root_rot, rows=rows)
    robot_state["root_velocity"] = torch.zeros((rows, 6), device=env.device, dtype=torch.float32)
    return initial_state


def _install_root_pose_guards(
    env: ManagerBasedRLMimicEnv,
    root_pos: tuple[float, float, float],
    root_rot: tuple[float, float, float, float],
) -> None:
    """Monkey-patch env.reset and env.reset_to to preserve calibrated robot root pose.

    This is intentionally aggressive for I4H Mimic generation because Isaac Mimic
    may reset envs repeatedly. Without this guard, some reset paths can restore
    the robot to [0,0,0,1,0,0,0].
    """
    original_reset = env.reset
    original_reset_to = env.reset_to

    def guarded_reset(*args, **kwargs):
        out = original_reset(*args, **kwargs)
        _apply_robot_root_pose_world(env, root_pos, root_rot, env_ids=None, render=False)
        return out

    def guarded_reset_to(initial_state, env_ids, *args, **kwargs):
        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=env.device, dtype=torch.long)

        initial_state = _patch_initial_state_robot_root(
            env=env,
            initial_state=initial_state,
            root_pos=root_pos,
            root_rot=root_rot,
            env_ids=env_ids_tensor,
        )
        out = original_reset_to(initial_state, env_ids, *args, **kwargs)
        _apply_robot_root_pose_world(env, root_pos, root_rot, env_ids=env_ids_tensor, render=False)
        return out

    env.reset = guarded_reset  # type: ignore[method-assign]
    env.reset_to = guarded_reset_to  # type: ignore[method-assign]


def _print_reset_check(env: ManagerBasedRLMimicEnv, prefix: str = "[GEN RESET CHECK]") -> None:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    print(prefix, "robot_root_pos =", robot.data.root_state_w[0, 0:3].detach().cpu().numpy())
    print(prefix, "robot_root_quat=", robot.data.root_state_w[0, 3:7].detach().cpu().numpy())
    print(prefix, "robot_joint_pos=", robot.data.joint_pos[0].detach().cpu().numpy())
    print(prefix, "object_pos     =", obj.data.root_state_w[0, 0:3].detach().cpu().numpy())


# -----------------------------------------------------------------------------
# Optional camera recorder
# -----------------------------------------------------------------------------
def _camera_rgb_uint8(env: ManagerBasedRLMimicEnv, camera_name: str) -> torch.Tensor:
    rgb = env.scene[camera_name].data.output["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != torch.uint8:
        if torch.is_floating_point(rgb) and torch.max(rgb) <= 1.5:
            rgb = rgb * 255.0
        rgb = torch.clamp(rgb, 0, 255).to(torch.uint8)
    return rgb


class PostStepCameraObsRecorder(RecorderTerm):
    """Record SO101 camera RGB observations after each env step."""

    def record_post_step(self):
        env = self._env
        env.sim.render()
        wrist = _camera_rgb_uint8(env, args_cli.wrist_camera_name)
        up = _camera_rgb_uint8(env, args_cli.up_camera_name)
        return "camera_obs", {"wrist": wrist, "up": up}


@configclass
class PostStepCameraObsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepCameraObsRecorder


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)

    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    print("==================================================")
    print("SO101 Mimic generation")
    print("==================================================")
    print(f"task/env_name          : {env_name}")
    print(f"input_file             : {args_cli.input_file}")
    print(f"output_file            : {args_cli.output_file}")
    print(f"num_envs               : {args_cli.num_envs}")
    print(f"generation_num_trials  : {args_cli.generation_num_trials}")
    print(f"record_cameras         : {args_cli.record_cameras}")
    print(f"debug                  : {args_cli.debug}")

    forced_root = _resolve_forced_root_pose(env_name)
    if forced_root is not None:
        root_pos, root_rot = forced_root
        print(f"force_cfg_robot_root   : True")
        print(f"forced root pos        : {root_pos}")
        print(f"forced root rot wxyz   : {root_rot}")
    else:
        root_pos = root_rot = None  # type: ignore[assignment]
        print("force_cfg_robot_root   : False")
    print("==================================================")

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )

    # Force calibrated root before the environment is constructed.
    if forced_root is not None:
        _force_env_cfg_robot_root(env_cfg, root_pos, root_rot)  # type: ignore[arg-type]

    # Camera recording is intentionally opt-in. Prefer replaying successes with cameras later.
    if args_cli.record_cameras:
        if not args_cli.enable_cameras:
            raise RuntimeError("--record_cameras requires AppLauncher --enable_cameras.")
        env_cfg.enable_camera_sensors = True
        env_cfg.recorders.record_post_step_camera_obs = PostStepCameraObsRecorderCfg()
    else:
        env_cfg.enable_camera_sensors = False
        if hasattr(env_cfg.recorders, "record_post_step_camera_obs"):
            env_cfg.recorders.record_post_step_camera_obs = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    env: ManagerBasedRLMimicEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Install root guards immediately after env construction, before any generation reset path.
    if forced_root is not None:
        _install_root_pose_guards(env, root_pos, root_rot)  # type: ignore[arg-type]

    # Initial reset sanity check.
    env.reset()
    _print_reset_check(env)

    # Check if the mimic API from this environment contains deprecated signatures.
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API '
            '"target_eef_pose_to_action" is deprecated. Please update the API to take action_noise_dict instead.'
        )

    # Set seed for generation.
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting generation. Guard should preserve root.
    env.reset()
    if args_cli.debug:
        _print_reset_check(env, prefix="[GEN RESET CHECK 2]")

    motion_planners = None
    if args_cli.use_skillgen:
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        motion_planners = {}
        for env_id in range(args_cli.num_envs):
            print(f"Initializing motion planner for environment {env_id}")
            planner_config = CuroboPlannerCfg.from_task_name(env_name)
            if env_id != 0:
                planner_config.visualize_spheres = False
                planner_config.visualize_plan = False
            motion_planners[env_id] = CuroboPlanner(
                env=env,
                robot=env.scene["robot"],
                config=planner_config,
                env_id=env_id,
            )
        env.cfg.datagen_config.use_skillgen = True

    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as exc:
            print(f"Error cancelling remaining tasks: {exc}")

        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            motion_planners.clear()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    simulation_app.close()