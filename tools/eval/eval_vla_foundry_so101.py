"""
Evaluate a VLA Foundry diffusion-policy checkpoint directly inside Isaac Lab.

This script is intentionally conservative/debuggable:
  - single-env first
  - camera capture from Isaac Lab sensors
  - two-frame image history: wrist_t-1, up_t-1, wrist_t0, up_t0
  - lowdim action history for VLA Foundry diffusion policy
  - receding-horizon execution with replanning

Run with the cloned hybrid env:
  conda activate env_so101_vla_isaaclab

  export VLA_FOUNDRY_ROOT=/home/insol02/IH_ws/vla_foundry
  export PYTHONPATH="$VLA_FOUNDRY_ROOT:$PYTHONPATH"
  export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

  /home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
    /home/insol02/IH_ws/so101_IsaacLab/tools/eval/eval_vla_foundry_so101.py \
    --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
    --checkpoint_dir /home/insol02/IH_ws/vla_foundry/tutorials/checkpoints/2026_05_20-12_28_03-model_diffusion_policy-lr_0.0005-bsz_4 \
    --max_steps 500 \
    --reset_steps 40 \
    --replan_steps 3 \
    --num_inference_steps 10
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys
import builtins
from functools import partial
print = partial(builtins.print, flush=True)
from dataclasses import dataclass
from typing import Any
from pathlib import Path


def checkpoint_sort_key(path):
    """Sort checkpoint_*.pt files by numeric checkpoint index."""
    import re

    name = Path(path).name
    m = re.search(r"checkpoint_(\d+)\.pt$", name)
    return int(m.group(1)) if m else -1

import numpy as np
import torch
from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI / Isaac Sim launch
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluate SO101 VLA Foundry policy in Isaac Lab.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0",
    help="Isaac Lab task name. Use the camera-enabled replay task.",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="VLA Foundry diffusion-policy checkpoint directory containing config_model.yaml, stats.json, checkpoints/.",
)
parser.add_argument(
    "--vla_foundry_root",
    type=str,
    default=os.environ.get("VLA_FOUNDRY_ROOT", "/home/insol02/IH_ws/vla_foundry"),
    help="Path to VLA Foundry repo root. Added to sys.path.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab envs. Start with 1.")
parser.add_argument("--max_steps", type=int, default=500, help="Max env steps for one rollout.")
parser.add_argument("--reset_steps", type=int, default=40, help="Settling steps after env reset.")
parser.add_argument("--replan_steps", type=int, default=3, help="Number of predicted actions to execute before replanning.")
parser.add_argument(
    "--execute_start_offset",
    type=int,
    default=1,
    help="Offset after lowdim_past_timesteps to start executing predicted actions. Use 1 to skip the anchor/present slot.",
)
parser.add_argument("--num_inference_steps", type=int, default=10, help="Diffusion denoising steps.")
parser.add_argument(
    "--task_description",
    type=str,
    default="Pick up the cube and place it in the goal region.",
    help="Language instruction passed to the VLA processor.",
)
parser.add_argument("--wrist_camera_key", type=str, default="wrist_camera")
parser.add_argument("--up_camera_key", type=str, default="up_camera")
parser.add_argument("--robot_key", type=str, default="robot")
parser.add_argument("--object_key", type=str, default="object")
parser.add_argument("--debug_every", type=int, default=10)
parser.add_argument(
    "--action_smoothing",
    type=float,
    default=0.0,
    help="Low-pass smoothing for executed actions. 0.0 = no smoothing, 0.2 = mild smoothing, 0.5 = strong smoothing.",
)
parser.add_argument(
    "--temporal_ensemble",
    action="store_true",
    help="Use ACT-style temporal ensembling over overlapping predicted action chunks.",
)
parser.add_argument(
    "--temporal_ensemble_window",
    type=int,
    default=8,
    help="Maximum number of candidate actions to average for each env step when temporal_ensemble is enabled.",
)
parser.add_argument(
    "--temporal_ensemble_decay",
    type=float,
    default=0.01,
    help="Exponential weighting decay for older predictions. Smaller means nearly uniform averaging.",
)
parser.add_argument(
    "--policy_fps",
    type=float,
    default=30.0,
    help="Action frequency used during VLA/LeRobot training.",
)
parser.add_argument(
    "--env_fps",
    type=float,
    default=50.0,
    help="Isaac Lab control frequency. For dt=0.005 and decimation=4, this is 50 Hz.",
)
parser.add_argument(
    "--respect_policy_fps",
    action="store_true",
    help="Repeat policy actions over multiple env steps to approximately preserve policy_fps timing.",
)
parser.add_argument(
    "--warm_start_action",
    type=str,
    default=None,
    help="Optional comma-separated 6D joint-position action to step before rollout, e.g. '-0.135,-1.62,1.69,1.29,-1.759,0.06'.",
)
parser.add_argument(
    "--warm_start_steps",
    type=int,
    default=80,
    help="Number of env steps to hold/interpolate toward warm_start_action before rollout.",
)
parser.add_argument(
    "--clip_to_action_stats",
    action="store_true",
    default=True,
    help="Clip denormalized predicted actions to training action min/max from stats.json.",
)
parser.add_argument(
    "--no_clip_to_action_stats",
    dest="clip_to_action_stats",
    action="store_false",
    help="Disable clipping predicted actions to training action min/max.",
)
parser.add_argument(
    "--dry_run_inference",
    action="store_true",
    help="Only reset, capture cameras, run one inference call, print predicted actions, and exit.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Make VLA Foundry importable before importing its modules.
if args_cli.vla_foundry_root and args_cli.vla_foundry_root not in sys.path:
    sys.path.insert(0, args_cli.vla_foundry_root)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports after Isaac Sim launch
# -----------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# Register SO101 tasks.
import isaac_so_arm101.tasks.cube_replay_i4h  # noqa: F401,E402
import isaac_so_arm101.tasks.cube_mimic_i4h  # noqa: F401,E402

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor  # noqa: E402
from vla_foundry.file_utils import load_model_checkpoint  # noqa: E402
from vla_foundry.models import create_model  # noqa: E402
from vla_foundry.params.model_params import ModelParams  # noqa: E402
from vla_foundry.params.train_experiment_params import load_params_from_yaml  # noqa: E402


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_uint8_rgb(img: Any) -> np.ndarray:
    """Convert Isaac Lab camera output to uint8 HWC RGB."""
    arr = _to_numpy(img)

    # Drop batch/env dim if present.
    if arr.ndim == 4:
        arr = arr[0]

    # Convert CHW -> HWC if needed.
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims after squeezing, got shape={arr.shape}")

    # RGBA -> RGB
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with last dim 3, got shape={arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    # Handle [0, 1] floats.
    if arr.max() <= 1.5:
        arr = arr * 255.0

    return np.clip(arr, 0, 255).astype(np.uint8)


def capture_camera_rgb(env, camera_key: str) -> np.ndarray:
    """Read an RGB frame from an Isaac Lab camera sensor."""
    scene = env.unwrapped.scene
    if camera_key not in scene.keys():
        raise KeyError(
            f"Camera key '{camera_key}' not found in scene. Available scene keys: {list(scene.keys())}"
        )

    cam = scene[camera_key]
    data = cam.data

    # Common Isaac Lab Camera/TiledCamera style: data.output["rgb"] or ["rgba"].
    if hasattr(data, "output"):
        output = data.output
        if "rgb" in output:
            return _as_uint8_rgb(output["rgb"])
        if "rgba" in output:
            return _as_uint8_rgb(output["rgba"])

        # Fallback: print available keys.
        raise KeyError(f"Camera '{camera_key}' data.output has no rgb/rgba. Keys: {list(output.keys())}")

    # Fallback for possible direct data.rgb style.
    if hasattr(data, "rgb"):
        return _as_uint8_rgb(data.rgb)

    raise AttributeError(f"Could not find RGB output for camera '{camera_key}'. data attrs={dir(data)}")


def get_joint_state_action(env, robot_key: str = "robot") -> np.ndarray:
    """Return current 6D SO101 joint-position state/action vector for env 0."""
    robot = env.unwrapped.scene[robot_key]
    q = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)

    # Your action space and training data are 6D.
    if q.shape[0] < 6:
        raise ValueError(f"Expected at least 6 robot joints, got shape={q.shape}")

    return q[:6].copy()


def get_object_pos_local(env, object_key: str = "object") -> np.ndarray | None:
    """Return object position in env-local coordinates for env 0 if available."""
    scene = env.unwrapped.scene
    if object_key not in scene.keys():
        return None

    obj = scene[object_key]
    if hasattr(obj.data, "root_state_w"):
        pos_w = obj.data.root_state_w[0, :3]
    elif hasattr(obj.data, "root_pos_w"):
        pos_w = obj.data.root_pos_w[0]
    else:
        return None

    pos_w = pos_w.detach()
    if hasattr(scene, "env_origins"):
        pos_local = pos_w - scene.env_origins[0]
    else:
        pos_local = pos_w
    return pos_local.cpu().numpy().astype(np.float32)


def get_goal_region(env) -> dict[str, float]:
    """Best-effort goal region lookup; fallback matches the shifted I4H goal you tuned."""
    cfg = getattr(env.unwrapped, "cfg", None)
    region = getattr(cfg, "goal_region", None)
    if isinstance(region, dict):
        return region

    # Fallback from the tuned I4H Mimic/replay config discussion.
    return {
        "x_min": 0.025,
        "x_max": 0.175,
        "y_min": 0.125,
        "y_max": 0.275,
        "z_min": 0.0,
        "z_max": 0.10,
    }


def is_object_in_goal(env) -> bool:
    pos = get_object_pos_local(env)
    if pos is None:
        return False
    g = get_goal_region(env)
    return (
        g["x_min"] <= float(pos[0]) <= g["x_max"]
        and g["y_min"] <= float(pos[1]) <= g["y_max"]
        and g["z_min"] <= float(pos[2]) <= g["z_max"]
    )


def step_env(env, action_np: np.ndarray):
    """Step env with one 6D action repeated over num_envs."""
    action_np = np.asarray(action_np, dtype=np.float32).reshape(1, -1)
    if action_np.shape[1] != env.unwrapped.action_manager.total_action_dim:
        raise ValueError(
            f"Action dim mismatch: action_np={action_np.shape}, "
            f"env action dim={env.unwrapped.action_manager.total_action_dim}"
        )

    action_t = torch.as_tensor(action_np, device=env.unwrapped.device, dtype=torch.float32)
    action_t = action_t.repeat(env.unwrapped.num_envs, 1)
    return env.step(action_t)


def parse_action_csv(s: str) -> np.ndarray:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 6:
        raise ValueError(f"Expected 6 comma-separated values, got {len(vals)}: {s}")
    return np.asarray(vals, dtype=np.float32)


def move_to_action_smooth(env, target_action: np.ndarray, num_steps: int = 80):
    """Smoothly move from current joint position to target joint-position action."""
    q0 = get_joint_state_action(env)
    target_action = np.asarray(target_action, dtype=np.float32)

    print(f"[WARM START] q0={q0}")
    print(f"[WARM START] target={target_action}")
    print(f"[WARM START] steps={num_steps}")

    for i in range(max(1, num_steps)):
        alpha = float(i + 1) / float(max(1, num_steps))
        # Smoothstep interpolation to avoid abrupt velocity at beginning/end.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        a = (1.0 - alpha) * q0 + alpha * target_action
        step_env(env, a)

    q1 = get_joint_state_action(env)
    print(f"[WARM START] q1={q1}")



def policy_action_repeat_count(policy_fps: float, env_fps: float, policy_action_index: int) -> int:
    """Return how many Isaac env steps to hold one policy action.

    Example:
        policy_fps=30, env_fps=50 gives an approximate repeat pattern [1, 2, 2, 1, 2, 2, ...],
        so the 30 Hz policy is not played too quickly in a 50 Hz sim.
    """
    if policy_fps <= 0.0 or env_fps <= 0.0:
        return 1

    t0 = float(policy_action_index) / float(policy_fps)
    t1 = float(policy_action_index + 1) / float(policy_fps)

    env_i0 = int(np.floor(t0 * env_fps + 1e-9))
    env_i1 = int(np.floor(t1 * env_fps + 1e-9))

    return max(1, env_i1 - env_i0)



class TemporalActionEnsembler:
    """ACT-style temporal ensemble for absolute joint-position action chunks.

    At each policy call, add the predicted future actions with their intended
    absolute env-step indices. At execution time, average all predictions that
    target the current env-step index.
    """

    def __init__(self, max_window: int = 8, decay: float = 0.01):
        self.max_window = int(max_window)
        self.decay = float(decay)
        self._buffer = {}

    def add_chunk(self, current_step: int, actions: np.ndarray, repeat_counts: list[int] | None = None):
        actions = np.asarray(actions, dtype=np.float32)

        step_idx = int(current_step)
        for i, a in enumerate(actions):
            if repeat_counts is None:
                repeats = 1
            else:
                repeats = int(repeat_counts[i])

            for _ in range(max(1, repeats)):
                self._buffer.setdefault(step_idx, []).append(a.copy())
                # Keep only recent predictions for this target step.
                if len(self._buffer[step_idx]) > self.max_window:
                    self._buffer[step_idx] = self._buffer[step_idx][-self.max_window:]
                step_idx += 1

    def get(self, step_idx: int) -> np.ndarray | None:
        candidates = self._buffer.pop(int(step_idx), None)
        if not candidates:
            return None

        arr = np.stack(candidates, axis=0).astype(np.float32)
        n = arr.shape[0]

        # Newer candidates are at the end. Weight newer predictions slightly more.
        ages = np.arange(n - 1, -1, -1, dtype=np.float32)
        weights = np.exp(-self.decay * ages)
        weights = weights / np.sum(weights)

        return np.sum(arr * weights[:, None], axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# VLA Foundry runner
# -----------------------------------------------------------------------------

@dataclass
class SO101VLAState:
    prev_wrist: np.ndarray
    prev_up: np.ndarray
    curr_wrist: np.ndarray
    curr_up: np.ndarray
    action_history: collections.deque


class VLAFoundrySO101Runner:
    def __init__(
        self,
        checkpoint_dir: str,
        task_description: str,
        num_inference_steps: int = 10,
        clip_to_action_stats: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.task_description = task_description
        self.num_inference_steps = num_inference_steps
        self.clip_to_action_stats = clip_to_action_stats

        ckpts = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoints", "checkpoint_*.pt")),
            key=checkpoint_sort_key,
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint_*.pt found in {checkpoint_dir}/checkpoints")
        self.ckpt_path = ckpts[-1]

        print(f"[VLA] checkpoint_dir: {self.checkpoint_dir}")
        print(f"[VLA] checkpoint:     {self.ckpt_path}")

        # VLA Foundry configs often contain relative paths such as
        # tutorials/checkpoints/... for the Stage-2 VLM checkpoint.
        # Resolve those relative to the VLA Foundry repo root.
        vla_root = args_cli.vla_foundry_root
        print(f"[VLA] changing cwd to VLA Foundry root: {vla_root}")
        os.chdir(vla_root)

        model_params = load_params_from_yaml(ModelParams, os.path.join(checkpoint_dir, "config_model.yaml"))
        self.model = create_model(model_params)
        load_model_checkpoint(self.model, self.ckpt_path)
        self.model.eval().cuda()

        self.processor = RoboticsProcessor.from_pretrained(checkpoint_dir)
        self.data_params = self.processor.data_params

        self.image_names = list(self.data_params.image_names)
        self.action_fields = list(self.data_params.action_fields)
        self.proprioception_fields = list(self.data_params.proprioception_fields)
        self.lowdim_past_timesteps = int(self.data_params.lowdim_past_timesteps)
        self.lowdim_future_timesteps = int(self.data_params.lowdim_future_timesteps)
        self.total_action_timesteps = self.lowdim_past_timesteps + 1 + self.lowdim_future_timesteps

        if self.action_fields != ["action"]:
            raise ValueError(f"Expected action_fields ['action'], got {self.action_fields}")
        if self.proprioception_fields != ["observation.state"]:
            raise ValueError(f"Expected proprioception_fields ['observation.state'], got {self.proprioception_fields}")

        self.action_field = self.action_fields[0]

        stats = self.processor.normalizer.stats.get(self.action_field, {})
        self.action_min = np.asarray(stats.get("min", [-np.inf] * 6), dtype=np.float32)
        self.action_max = np.asarray(stats.get("max", [np.inf] * 6), dtype=np.float32)

        print("[VLA] image_names:", self.image_names)
        print("[VLA] action_fields:", self.action_fields)
        print("[VLA] proprioception_fields:", self.proprioception_fields)
        print("[VLA] total_action_timesteps:", self.total_action_timesteps)
        print("[VLA] action_min:", self.action_min)
        print("[VLA] action_max:", self.action_max)

    def initialize_state(self, wrist_rgb: np.ndarray, up_rgb: np.ndarray, current_action: np.ndarray) -> SO101VLAState:
        action_history = collections.deque(maxlen=self.lowdim_past_timesteps)

        # Seed past action history with current joint-position action.
        for _ in range(self.lowdim_past_timesteps):
            action_history.append(np.asarray(current_action, dtype=np.float32).copy())

        return SO101VLAState(
            prev_wrist=wrist_rgb.copy(),
            prev_up=up_rgb.copy(),
            curr_wrist=wrist_rgb.copy(),
            curr_up=up_rgb.copy(),
            action_history=action_history,
        )

    def update_images(self, state: SO101VLAState, wrist_rgb: np.ndarray, up_rgb: np.ndarray) -> None:
        state.prev_wrist = state.curr_wrist
        state.prev_up = state.curr_up
        state.curr_wrist = wrist_rgb.copy()
        state.curr_up = up_rgb.copy()

    def append_executed_action(self, state: SO101VLAState, action: np.ndarray) -> None:
        state.action_history.append(np.asarray(action, dtype=np.float32).copy())

    def _build_action_sequence(self, state: SO101VLAState, current_action: np.ndarray) -> torch.Tensor:
        """Build [17,6] action tensor: 2 past + 1 present/dummy + 14 future/dummy."""
        current_action = np.asarray(current_action, dtype=np.float32)
        if current_action.shape != (6,):
            raise ValueError(f"Expected current_action shape (6,), got {current_action.shape}")

        seq = []

        # Past action slots.
        hist = list(state.action_history)
        if len(hist) < self.lowdim_past_timesteps:
            hist = [current_action.copy()] * (self.lowdim_past_timesteps - len(hist)) + hist
        seq.extend(hist[-self.lowdim_past_timesteps :])

        # Present + future dummy slots.
        while len(seq) < self.total_action_timesteps:
            seq.append(current_action.copy())

        arr = np.stack(seq, axis=0).astype(np.float32)
        return torch.as_tensor(arr, dtype=torch.float32)

    def _build_proprioception_sequence(self, current_state: np.ndarray) -> torch.Tensor:
        """Build [past+present,6] proprioception tensor."""
        current_state = np.asarray(current_state, dtype=np.float32)
        arr = np.stack([current_state.copy()] * (self.lowdim_past_timesteps + 1), axis=0)
        return torch.as_tensor(arr, dtype=torch.float32)

    def infer_action_chunk(self, state: SO101VLAState, current_state: np.ndarray, current_action: np.ndarray) -> np.ndarray:
        """Return denormalized action chunk [17,6]. Execute from index lowdim_past_timesteps onward."""
        images = {
            "wrist_t-1": state.prev_wrist,
            "up_t-1": state.prev_up,
            "wrist_t0": state.curr_wrist,
            "up_t0": state.curr_up,
        }

        lowdim = {
            "action": self._build_action_sequence(state, current_action),
            "observation.state": self._build_proprioception_sequence(current_state),
        }

        processor_input = {
            "images": [images],
            "lowdim": [lowdim],
            "metadata": [
                {
                    "anchor_relative_idx": self.lowdim_past_timesteps,
                    "original_anchor_relative_idx": self.lowdim_past_timesteps,
                }
            ],
            "language_instruction": [self.task_description],
        }

        batch = self.processor.process_inputs(processor_input, image_names=self.image_names)
        batch = self.processor.add_action_and_proprioception_fields(
            batch,
            action_fields=self.action_fields,
            proprioception_fields=self.proprioception_fields,
        )

        actions = batch["actions"].cuda()

        past_mask_np = np.zeros((1, self.total_action_timesteps), dtype=bool)
        past_mask_np[:, : self.lowdim_past_timesteps] = True
        past_mask = torch.as_tensor(past_mask_np, dtype=torch.bool).cuda()

        with torch.no_grad():
            predicted = self.model.generate_actions(
                input_ids=batch["input_ids"].cuda(),
                pixel_values=batch["pixel_values"].cuda(),
                actions=actions,
                attention_mask=batch["attention_mask"].cuda().bool(),
                attention_mask_images=batch["attention_mask_images"].cuda().bool(),
                past_mask=past_mask,
                proprioception=batch["proprioception"].cuda() if "proprioception" in batch else None,
                num_inference_steps=self.num_inference_steps,
            )

        pred_denorm = self.processor.normalizer.denormalize_tensor(
            predicted.detach().cpu(),
            self.action_field,
            anchor_timestep=self.lowdim_past_timesteps,
        )

        pred_np = pred_denorm.numpy()[0].astype(np.float32)

        if not np.isfinite(pred_np).all():
            raise RuntimeError("VLA predicted NaN/Inf actions after denormalization.")

        if self.clip_to_action_stats:
            pred_np = np.clip(pred_np, self.action_min, self.action_max)

        return pred_np


# -----------------------------------------------------------------------------
# Main eval
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Creating Isaac Lab env...")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Avoid time-limit auto-reset if present.
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    print("[INFO] Env type:", type(unwrapped))
    print("[INFO] Device:", unwrapped.device)
    print("[INFO] Num envs:", unwrapped.num_envs)
    print("[INFO] Action dim:", unwrapped.action_manager.total_action_dim)
    print("[INFO] Scene keys:", list(unwrapped.scene.keys()))
    print("[INFO] Goal region:", get_goal_region(env))

    if args_cli.num_envs != 1:
        raise NotImplementedError("This first evaluator intentionally supports num_envs=1 only.")

    if unwrapped.action_manager.total_action_dim != 6:
        raise ValueError(f"Expected 6D joint-position action env, got action dim {unwrapped.action_manager.total_action_dim}")

    print("[INFO] Resetting env...")
    obs = env.reset()

    # Let env/cameras settle using current joint position as a hold action.
    for i in range(args_cli.reset_steps):
        hold = get_joint_state_action(env, args_cli.robot_key)
        step_env(env, hold)

    # Optional: move robot to the same nominal joint-position manifold as the training data.
    if args_cli.warm_start_action is not None:
        target = parse_action_csv(args_cli.warm_start_action)
        move_to_action_smooth(env, target, num_steps=args_cli.warm_start_steps)

    wrist = capture_camera_rgb(env, args_cli.wrist_camera_key)
    up = capture_camera_rgb(env, args_cli.up_camera_key)
    q = get_joint_state_action(env, args_cli.robot_key)

    print("[INFO] Initial wrist image:", wrist.shape, wrist.dtype)
    print("[INFO] Initial up image:", up.shape, up.dtype)
    print("[INFO] Initial joint/action:", q)
    print("[INFO] Initial object pos local:", get_object_pos_local(env, args_cli.object_key))

    runner = VLAFoundrySO101Runner(
        checkpoint_dir=args_cli.checkpoint_dir,
        task_description=args_cli.task_description,
        num_inference_steps=args_cli.num_inference_steps,
        clip_to_action_stats=args_cli.clip_to_action_stats,
    )

    state = runner.initialize_state(wrist, up, q)

    if args_cli.dry_run_inference:
        action_chunk = runner.infer_action_chunk(state, current_state=q, current_action=q)
        start = runner.lowdim_past_timesteps + args_cli.execute_start_offset
        print("[DRY RUN] action_chunk shape:", action_chunk.shape)
        print("[DRY RUN] first executable actions:")
        for i in range(start, min(start + args_cli.replan_steps, action_chunk.shape[0])):
            print(f"  idx={i}: {action_chunk[i]}")
        env.close()
        return

    action_plan: collections.deque[np.ndarray] = collections.deque()
    ever_success = False
    prev_executed_action = get_joint_state_action(env, args_cli.robot_key)
    policy_action_counter = 0
    env_action_counter = 0
    temporal_ensembler = TemporalActionEnsembler(
        max_window=args_cli.temporal_ensemble_window,
        decay=args_cli.temporal_ensemble_decay,
    ) if args_cli.temporal_ensemble else None

    print("[INFO] Starting closed-loop rollout...")
    with torch.inference_mode():
        for t in range(args_cli.max_steps):
            wrist = capture_camera_rgb(env, args_cli.wrist_camera_key)
            up = capture_camera_rgb(env, args_cli.up_camera_key)
            runner.update_images(state, wrist, up)

            q = get_joint_state_action(env, args_cli.robot_key)

            if not action_plan:
                action_chunk = runner.infer_action_chunk(state, current_state=q, current_action=q)

                # Execute predictions after the past slots.
                # Offset=1 skips the anchor/present slot and starts at the first true future action.
                start = runner.lowdim_past_timesteps + args_cli.execute_start_offset
                end = min(start + args_cli.replan_steps, action_chunk.shape[0])
                chunk_to_execute = [a.copy() for a in action_chunk[start:end]]

                if temporal_ensembler is not None:
                    repeat_counts = []
                    local_policy_counter = policy_action_counter
                    for _ in chunk_to_execute:
                        if args_cli.respect_policy_fps:
                            rc = policy_action_repeat_count(
                                policy_fps=args_cli.policy_fps,
                                env_fps=args_cli.env_fps,
                                policy_action_index=local_policy_counter,
                            )
                        else:
                            rc = 1
                        repeat_counts.append(rc)
                        local_policy_counter += 1

                    temporal_ensembler.add_chunk(
                        current_step=env_action_counter,
                        actions=np.stack(chunk_to_execute, axis=0),
                        repeat_counts=repeat_counts,
                    )
                else:
                    for a in chunk_to_execute:
                        action_plan.append(a.copy())

                if args_cli.debug_every > 0:
                    print(f"[PLAN t={t}] action_chunk[{start}:{end}]")
                    for i in range(start, end):
                        print(f"  idx={i}: {action_chunk[i]}")

            if temporal_ensembler is not None:
                raw_action = temporal_ensembler.get(env_action_counter)
                if raw_action is None:
                    # If no ensembled action is available for this step, force a replan next loop.
                    action_plan.clear()
                    continue
            else:
                raw_action = action_plan.popleft()

            if args_cli.action_smoothing > 0.0:
                beta = float(args_cli.action_smoothing)
                beta = max(0.0, min(beta, 0.95))
                action = beta * prev_executed_action + (1.0 - beta) * raw_action
            else:
                action = raw_action

            runner.append_executed_action(state, action)
            prev_executed_action = action.copy()

            repeat_count = 1
            if args_cli.respect_policy_fps:
                repeat_count = policy_action_repeat_count(
                    policy_fps=args_cli.policy_fps,
                    env_fps=args_cli.env_fps,
                    policy_action_index=policy_action_counter,
                )
            policy_action_counter += 1

            success = False
            for _repeat_i in range(repeat_count):
                step_env(env, action)
                env_action_counter += 1
                success = is_object_in_goal(env)
                ever_success = ever_success or success
                if success:
                    break

            if args_cli.debug_every > 0 and (t % args_cli.debug_every == 0 or success):
                obj_pos = get_object_pos_local(env, args_cli.object_key)
                print(
                    f"[STEP {t:04d}] repeat={repeat_count} success={success} ever_success={ever_success} "
                    f"obj_local={obj_pos} action={action}"
                )

            if success:
                print(f"[DONE] Success reached at step {t}.")
                break

    print("[RESULT] ever_success:", ever_success)
    print("[RESULT] final_object_pos_local:", get_object_pos_local(env, args_cli.object_key))

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("[FATAL] eval_vla_foundry_so101.py crashed:")
        traceback.print_exc()
        raise
    finally:
        print("[INFO] Closing simulation app.")
        simulation_app.close()