#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export VLA_FOUNDRY_ROOT="${VLA_FOUNDRY_ROOT:-/home/insol02/IH_ws/vla_foundry}"
export PYTHONPATH="$VLA_FOUNDRY_ROOT:${PYTHONPATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1"
ISAACLAB_SH="${ISAACLAB_SH:-/home/insol02/IH_ws/IsaacLab/isaaclab.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/home/insol02/IH_ws/so101_IsaacLab/tools/eval/eval_vla_foundry_so101.py}"
CKPT_DIR="${1:?Usage: $0 /absolute/path/to/diffusion_checkpoint_dir}"
"$ISAACLAB_SH" -p "$EVAL_SCRIPT" \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --checkpoint_dir "$CKPT_DIR" \
  --max_steps 800 \
  --reset_steps 12 \
  --warm_start_action="-0.135,-1.62,1.69,1.29,-1.759,0.36" \
  --warm_start_steps 200 \
  --replan_steps 10 \
  --execute_start_offset 1 \
  --num_inference_steps 10 \
  --respect_policy_fps \
  --policy_fps 30 \
  --env_fps 50 \
  --action_smoothing 0.1 \
  --debug_every 10 \
  --enable_cameras
