# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
SO-101 wrapper for Isaac Lab Mimic demo annotation.

Key SO-101-specific changes from the stock annotate_demos.py:

1. Imports isaac_so_arm101.tasks after AppLauncher starts Isaac Sim.
2. Uses the same replay/reset/action stepping pattern as the known-good SO-101
   replay/filter scripts:
      - episode.get_initial_state()
      - env.reset_to(initial_state, env_ids, is_relative=True)
      - batched action tensor [num_envs, action_dim] on env.device
3. Adds debug prints for reset state, first actions, object pose, robot root pose,
   end-effector pose, and success/subtask signals.
4. Adds --reset_state_is_world to test is_relative=False without editing code.

Important:
    reset_to(..., is_relative=True) refers to whether the saved *scene state* is
    relative to the env origin, not whether the action commands are absolute or
    relative joint commands. Your SO-101 action semantics are controlled by the
    JointPositionActionCfg(use_default_offset=False) in the task cfg.
"""

import argparse
import math

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Annotate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument(
    "--auto",
    action="store_true",
    default=True,
    help="Automatically annotate subtasks. Default: enabled.",
)
parser.add_argument(
    "--manual",
    action="store_true",
    default=False,
    help="Use interactive/manual annotation instead of automatic annotation.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable verbose reset/action/scene debug prints. Default: quiet.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio before Isaac Sim app launch.",
)
parser.add_argument(
    "--annotate_subtask_start_signals",
    action="store_true",
    default=False,
    help="Enable annotating start points of subtasks.",
)
parser.add_argument(
    "--debug_every",
    type=int,
    default=0,
    help="Print replay debug info every N action steps when --debug is set. Set <=0 to disable periodic debug prints.",
)
parser.add_argument(
    "--debug_first_n_actions",
    type=int,
    default=0,
    help="Print the first N action vectors for each episode when --debug is set.",
)
parser.add_argument(
    "--reset_state_is_world",
    action="store_true",
    default=False,
    help=(
        "Use env.reset_to(..., is_relative=False). Default is False, meaning the script uses "
        "is_relative=True, which matches Isaac Lab replay_demos.py and your working replay/filter scripts."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Default behavior should be non-interactive automatic annotation.
# Use --manual for keyboard marking, and --debug for verbose prints.
if args_cli.manual:
    args_cli.auto = False
if not args_cli.debug:
    args_cli.debug_every = 0
    args_cli.debug_first_n_actions = 0

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed
    # by IsaacLab and not the one installed by Isaac Sim.
    import pinocchio  # noqa: F401

# Launch Isaac Sim first. Do not import SO-101/IsaacLab task modules before this.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after AppLauncher
# -----------------------------------------------------------------------------
import contextlib
import os
from typing import Any

import gymnasium as gym
import torch

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False
current_action_index = 0
marked_subtask_action_indices: list[int] = []
skip_episode = False


def log_info(*args, **kwargs):
    print(*args, **kwargs)


def log_debug(*args, **kwargs):
    if args_cli.debug:
        print(*args, **kwargs)


# -----------------------------------------------------------------------------
# Keyboard callbacks
# -----------------------------------------------------------------------------
def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def skip_episode_cb():
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked a subtask signal at action index: {current_action_index}")


# -----------------------------------------------------------------------------
# Recorder terms
# -----------------------------------------------------------------------------
class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records Mimic datagen info before each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskStartsObservationsRecorder(RecorderTerm):
    """Recorder term that records subtask start observations before each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask start observations recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskStartsObservationsRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records subtask completion observations before each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic-specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsObservationsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


# -----------------------------------------------------------------------------
# Debug helpers
# -----------------------------------------------------------------------------
def _to_cpu_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def _tensorize_action(action: Any, env: ManagerBasedRLMimicEnv) -> torch.Tensor:
    """Convert one episode action into shape [num_envs, action_dim] on env.device."""
    if isinstance(action, torch.Tensor):
        action_tensor = action.to(device=env.device, dtype=torch.float32)
    else:
        action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32)

    action_tensor = action_tensor.flatten()
    batched_actions = torch.zeros((env.num_envs, action_tensor.numel()), device=env.device, dtype=torch.float32)
    batched_actions[0] = action_tensor
    return batched_actions


def _get_episode_initial_state(episode: EpisodeData):
    """Use EpisodeData API when available, with fallback for older/custom data."""
    if hasattr(episode, "get_initial_state"):
        return episode.get_initial_state()
    return episode.data["initial_state"]


def _get_object_root(env: ManagerBasedRLMimicEnv) -> torch.Tensor | None:
    if "object" not in env.scene.keys():
        return None
    return env.scene["object"].data.root_state_w[0].detach().cpu()


def _get_robot_root(env: ManagerBasedRLMimicEnv) -> torch.Tensor | None:
    if "robot" not in env.scene.keys():
        return None
    robot = env.scene["robot"]
    if not hasattr(robot.data, "root_state_w"):
        return None
    return robot.data.root_state_w[0].detach().cpu()


def _get_robot_joint_debug(env: ManagerBasedRLMimicEnv, max_items: int = 8) -> torch.Tensor | None:
    if "robot" not in env.scene.keys():
        return None
    robot = env.scene["robot"]
    if not hasattr(robot.data, "joint_pos"):
        return None
    return robot.data.joint_pos[0, :max_items].detach().cpu()


def _print_scene_debug(env: ManagerBasedRLMimicEnv, prefix: str, success_term: TerminationTermCfg | None = None):
    if not args_cli.debug:
        return
    obj_root = _get_object_root(env)
    if obj_root is not None:
        log_debug(f"{prefix} object_pos={_to_cpu_numpy(obj_root[0:3])}")
        log_debug(f"{prefix} object_quat={_to_cpu_numpy(obj_root[3:7])}")
        log_debug(f"{prefix} object_lin_vel={_to_cpu_numpy(obj_root[7:10])}")

    robot_root = _get_robot_root(env)
    if robot_root is not None:
        log_debug(f"{prefix} robot_root_pos={_to_cpu_numpy(robot_root[0:3])}")
        log_debug(f"{prefix} robot_root_quat={_to_cpu_numpy(robot_root[3:7])}")

    joint_debug = _get_robot_joint_debug(env)
    if joint_debug is not None:
        log_debug(f"{prefix} robot_joint_pos_first={_to_cpu_numpy(joint_debug)}")

    if hasattr(env, "get_robot_eef_pose"):
        try:
            eef_pose = env.get_robot_eef_pose("end_effector")[0].detach().cpu()
            log_debug(f"{prefix} eef_pose={_to_cpu_numpy(eef_pose)}")
        except Exception as exc:
            log_debug(f"{prefix} eef_pose_debug_failed={type(exc).__name__}: {exc}")

    if hasattr(env.cfg, "goal_region"):
        log_debug(f"{prefix} goal_region={env.cfg.goal_region}")

    if hasattr(env, "get_subtask_term_signals"):
        try:
            signals = env.get_subtask_term_signals()
            signal_summary = {k: bool(v[0].detach().cpu()) for k, v in signals.items()}
            log_debug(f"{prefix} subtask_signals={signal_summary}")
        except Exception as exc:
            log_debug(f"{prefix} subtask_signal_debug_failed={type(exc).__name__}: {exc}")

    if success_term is not None:
        try:
            success_tensor = success_term.func(env, **success_term.params)
            log_debug(f"{prefix} success_term={bool(success_tensor[0].detach().cpu())}")
        except Exception as exc:
            log_debug(f"{prefix} success_debug_failed={type(exc).__name__}: {exc}")


def _print_initial_state_summary(initial_state: dict):
    log_debug("[RESET DEBUG] initial_state top-level keys:", list(initial_state.keys()))
    for asset_type in ["articulation", "rigid_object"]:
        if asset_type not in initial_state:
            log_debug(f"[RESET DEBUG] initial_state missing {asset_type}")
            continue
        log_debug(f"[RESET DEBUG] initial_state[{asset_type}] assets:", list(initial_state[asset_type].keys()))
        for asset_name, state_dict in initial_state[asset_type].items():
            log_debug(f"[RESET DEBUG] initial_state[{asset_type}][{asset_name}] fields:", list(state_dict.keys()))
            for state_name, state_value in state_dict.items():
                if isinstance(state_value, torch.Tensor):
                    flat = state_value.detach().cpu().flatten()
                    log_debug(
                        f"[RESET DEBUG] {asset_type}/{asset_name}/{state_name}: "
                        f"shape={tuple(state_value.shape)} first={flat[:min(10, flat.numel())].numpy()}"
                    )
                else:
                    log_debug(f"[RESET DEBUG] {asset_type}/{asset_name}/{state_name}: type={type(state_value)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    global is_paused, current_action_index, marked_subtask_action_indices

    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        log_info("No episodes found in the dataset.")
        return 0

    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name

    log_info("==================================================")
    log_info("SO101 Mimic annotation")
    log_info("==================================================")
    log_info(f"task             : {env_name}")
    log_info(f"input_file       : {args_cli.input_file}")
    log_info(f"output_file      : {args_cli.output_file}")
    log_info(f"episodes         : {episode_count}")
    log_info(f"mode             : {'auto' if args_cli.auto else 'manual'}")
    log_info(f"debug            : {args_cli.debug}")
    log_info("==================================================")

    log_debug("[CONFIG DEBUG] env_name=", env_name)
    log_debug("[CONFIG DEBUG] env_cfg class=", type(env_cfg))
    log_debug("[CONFIG DEBUG] device=", args_cli.device)
    log_debug("[CONFIG DEBUG] reset_state_is_world=", args_cli.reset_state_is_world)
    log_debug("[CONFIG DEBUG] reset_to is_relative will be=", not args_cli.reset_state_is_world)
    if hasattr(env_cfg, "goal_region"):
        log_debug("[CONFIG DEBUG] goal_region=", env_cfg.goal_region)
    if hasattr(env_cfg, "subtask_configs"):
        log_debug("[CONFIG DEBUG] subtask_configs keys=", list(env_cfg.subtask_configs.keys()))
        for eef_name, cfgs in env_cfg.subtask_configs.items():
            log_debug(f"[CONFIG DEBUG] subtask_configs[{eef_name}] signals=", [c.subtask_term_signal for c in cfgs])

    # Extract success checking function to invoke manually. Then disable runtime terminations.
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        log_debug("[CONFIG DEBUG] extracted success_term=", success_term)
        log_debug("[CONFIG DEBUG] success_term params=", getattr(success_term, "params", None))
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations.
    env_cfg.recorders = MimicRecorderManagerCfg()
    if not args_cli.auto:
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    if not args_cli.auto or (args_cli.auto and not args_cli.annotate_subtask_start_signals):
        env_cfg.recorders.record_pre_step_subtask_start_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # Create environment.
    env: ManagerBasedRLMimicEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    log_debug("[ENV DEBUG] env type=", type(env))
    log_debug("[ENV DEBUG] env.device=", env.device)
    log_debug("[ENV DEBUG] env.num_envs=", env.num_envs)
    log_debug("[ENV DEBUG] env.action_space=", env.action_space)
    if hasattr(env, "action_manager"):
        log_debug("[ENV DEBUG] action_manager action shape after construction=", getattr(env.action_manager, "action", None))

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )
        if (
            args_cli.annotate_subtask_start_signals
            and env.get_subtask_start_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_start_signals
        ):
            raise NotImplementedError(
                "The environment does not implement the get_subtask_start_signals method required "
                "to run automatic annotations."
            )
    else:
        subtask_term_signal_names = {}
        subtask_start_signal_names = {}
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_start_signal_names[eef_name] = (
                [subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs]
                if args_cli.annotate_subtask_start_signals
                else []
            )
            subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            if args_cli.annotate_subtask_start_signals:
                if any(name in (None, "") for name in subtask_start_signal_names[eef_name]):
                    raise ValueError(
                        f"Missing 'subtask_term_signal' for one or more subtasks in eef '{eef_name}'. "
                        "When '--annotate_subtask_start_signals' is enabled, each subtask must specify "
                        "'subtask_term_signal'."
                    )
            subtask_term_signal_names[eef_name].pop()

    # Reset once to initialize managers/sensors.
    env.reset()
    _print_scene_debug(env, "[ENV RESET DEBUG]", success_term=success_term)

    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)
        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)
        keyboard_interface.reset()

    exported_episode_count = 0
    processed_episode_count = 0
    successful_task_count = 0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                log_info(f"[Episode {episode_index + 1}/{episode_count}] Annotating {episode_name}...")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                if args_cli.auto:
                    is_episode_annotated_successfully = annotate_episode_in_auto_mode(env, episode, success_term)
                else:
                    is_episode_annotated_successfully = annotate_episode_in_manual_mode(
                        env, episode, success_term, subtask_term_signal_names, subtask_start_signal_names
                    )

                _print_scene_debug(env, "[ANNOTATE FINAL DEBUG]", success_term=success_term)

                if is_episode_annotated_successfully and not skip_episode:
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    successful_task_count += 1
                    log_info(f"[Episode {episode_index + 1}/{episode_count}] success=True exported=True")
                else:
                    log_info(f"[Episode {episode_index + 1}/{episode_count}] success=False exported=False")
            break

    log_info("==================================================")
    log_info("Annotation summary")
    log_info("==================================================")
    log_info(f"processed_episodes : {processed_episode_count}")
    log_info(f"exported_episodes  : {exported_episode_count}")
    log_info(f"successful_tasks   : {successful_task_count}")
    log_info(f"output_file        : {args_cli.output_file}")
    log_info("==================================================")
    log_info("Exiting the app.")

    env.close()
    return successful_task_count


# -----------------------------------------------------------------------------
# Replay / annotation helpers
# -----------------------------------------------------------------------------
def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Replay one episode using the same reset/action stepping semantics as SO-101 replay scripts."""
    global current_action_index, skip_episode, is_paused

    initial_state = _get_episode_initial_state(episode)

    env.recorder_manager.reset()

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    is_relative_state = not args_cli.reset_state_is_world

    log_debug("[REPLAY DEBUG] applying episode initial_state")
    log_debug("[REPLAY DEBUG] reset_to env_ids=", env_ids)
    log_debug("[REPLAY DEBUG] reset_to is_relative=", is_relative_state)
    _print_initial_state_summary(initial_state)

    env.reset_to(initial_state, env_ids, is_relative=is_relative_state)
    env.sim.render()

    _print_scene_debug(env, "[REPLAY RESET DEBUG]", success_term=success_term)

    first_action = True
    step_idx = 0
    last_action = None

    # Iterate using EpisodeData.get_next_action() to match replay_demos.py/filter_successful_replays.py behavior.
    while True:
        action = episode.get_next_action()
        if action is None:
            break

        current_action_index = step_idx

        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
                continue

        actions = _tensorize_action(action, env)
        last_action = actions[0].detach().cpu()

        if step_idx < args_cli.debug_first_n_actions:
            log_debug(f"[ACTION DEBUG] step={step_idx} action_shape={tuple(actions.shape)} action={_to_cpu_numpy(actions[0])}")

        env.step(actions)

        if args_cli.debug_every > 0 and step_idx % args_cli.debug_every == 0:
            _print_scene_debug(env, f"[REPLAY STEP DEBUG step={step_idx}]", success_term=success_term)

        step_idx += 1

    log_debug("[REPLAY DEBUG] completed action loop")
    log_debug("[REPLAY DEBUG] num_steps=", step_idx)
    if last_action is not None:
        log_debug("[REPLAY DEBUG] last_action=", _to_cpu_numpy(last_action))

    _print_scene_debug(env, "[REPLAY FINAL DEBUG]", success_term=success_term)

    if success_term is not None:
        success_tensor = success_term.func(env, **success_term.params)
        success_bool = bool(success_tensor[0].detach().cpu())
        log_debug("[REPLAY FINAL DEBUG] success_bool=", success_bool)
        if not success_bool:
            return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Annotate an episode in automatic mode."""
    global skip_episode
    skip_episode = False

    is_episode_annotated_successfully = replay_episode(env, episode, success_term)
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
        return False

    annotated_episode = env.recorder_manager.get_episode(0)
    subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
    for signal_name, signal_flags in subtask_term_signal_dict.items():
        signal_flags = torch.tensor(signal_flags, device=env.device)
        signal_any = bool(torch.any(signal_flags).detach().cpu())
        log_info(
            f"[SUBTASK DEBUG] {signal_name}: any={signal_any} "
            f"sum={int(torch.sum(signal_flags).detach().cpu())} len={len(signal_flags)}"
        )
        if not signal_any:
            is_episode_annotated_successfully = False
            print(f'\tDid not detect completion for the subtask "{signal_name}".')

    if args_cli.annotate_subtask_start_signals:
        subtask_start_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_start_signals"]
        for signal_name, signal_flags in subtask_start_signal_dict.items():
            signal_flags = torch.tensor(signal_flags, device=env.device)
            signal_any = bool(torch.any(signal_flags).detach().cpu())
            log_info(
                f"[SUBTASK START DEBUG] {signal_name}: any={signal_any} "
                f"sum={int(torch.sum(signal_flags).detach().cpu())} len={len(signal_flags)}"
            )
            if not signal_any:
                is_episode_annotated_successfully = False
                print(f'\tDid not detect start for the subtask "{signal_name}".')

    return is_episode_annotated_successfully


def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] = {},
    subtask_start_signal_names: dict[str, list[str]] = {},
) -> bool:
    """Annotate an episode in manual mode."""
    global is_paused, marked_subtask_action_indices, skip_episode

    subtask_term_signal_action_indices = {}
    subtask_start_signal_action_indices = {}
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        eef_subtask_start_signal_names = subtask_start_signal_names[eef_name]
        if len(eef_subtask_term_signal_names) == 0 and len(eef_subtask_start_signal_names) == 0:
            continue

        while True:
            is_paused = True
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            if len(eef_subtask_start_signal_names) > 0:
                print(f"\t\t- Start:\t{eef_subtask_start_signal_names}")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")

            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            marked_subtask_action_indices = []
            task_success_result = replay_episode(env, episode, success_term)
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            expected_subtask_signal_count = len(eef_subtask_term_signal_names) + len(eef_subtask_start_signal_names)
            if task_success_result and expected_subtask_signal_count == len(marked_subtask_action_indices):
                print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                for marked_signal_index in range(expected_subtask_signal_count):
                    if args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 0:
                        subtask_start_signal_action_indices[
                            eef_subtask_start_signal_names[int(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                    if not args_cli.annotate_subtask_start_signals:
                        subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                            marked_subtask_action_indices[marked_signal_index]
                        )
                    elif args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 1:
                        subtask_term_signal_action_indices[
                            eef_subtask_term_signal_names[math.floor(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                break

            if not task_success_result:
                print("\tThe final task was not completed.")
                return False

            if expected_subtask_signal_count != len(marked_subtask_action_indices):
                print(
                    f"\tOnly {len(marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were'
                    " annotated."
                )

            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    annotated_episode = env.recorder_manager.get_episode(0)
    for subtask_term_signal_name, subtask_term_signal_action_index in subtask_term_signal_action_indices.items():
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)

    if args_cli.annotate_subtask_start_signals:
        for subtask_start_signal_name, subtask_start_signal_action_index in subtask_start_signal_action_indices.items():
            subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
            subtask_signals[:subtask_start_signal_action_index] = False
            annotated_episode.add(
                f"obs/datagen_info/subtask_start_signals/{subtask_start_signal_name}", subtask_signals
            )

    return True


if __name__ == "__main__":
    successful_task_count = main()
    simulation_app.close()
    exit(successful_task_count)