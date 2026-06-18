#!/usr/bin/env bash
set -euo pipefail

# Live SO101 VLA inference sweep.
# Edit paths as needed, then run:
#   bash run_live_vla_tests.sh
#
# This writes logs under outputs/vla_live_tests/<timestamp>/.

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/insol02/IH_ws/IsaacLab}"
SO101_REPO="${SO101_REPO:-/home/insol02/IH_ws/so101_IsaacLab}"
VLA_FOUNDRY_ROOT="${VLA_FOUNDRY_ROOT:-/home/insol02/IH_ws/vla_foundry}"
CKPT_DIR="${CKPT_DIR:-/home/insol02/IH_ws/vla_foundry/tutorials/checkpoints/2026_05_21-13_14_36-model_diffusion_policy-lr_0.0005-bsz_16}"

TASK="${TASK:-Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0}"
OUT_ROOT="${OUT_ROOT:-$SO101_REPO/outputs/vla_live_tests/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

export PYTHONUNBUFFERED=1
export VLA_FOUNDRY_ROOT="$VLA_FOUNDRY_ROOT"
export PYTHONPATH="$VLA_FOUNDRY_ROOT:${PYTHONPATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-}:/lib/aarch64-linux-gnu/libgomp.so.1"

BASE_CMD=(
  "$ISAACLAB_ROOT/isaaclab.sh" -p
  "$SO101_REPO/tools/eval/eval_vla_foundry_so101.py"
  --task "$TASK"
  --checkpoint_dir "$CKPT_DIR"
  --max_steps 800
  --reset_steps 12
  --warm_start_action="-0.13564736,-1.62364730,1.68500000,1.31176210,-1.75949300,0.03890861"
  --warm_start_steps 200
  --num_inference_steps 10
  --respect_policy_fps
  --policy_fps 30
  --env_fps 50
  --action_smoothing 0.0
  --debug_every 5
  --enable_cameras
)

run_case () {
  local name="$1"; shift
  local log="$OUT_ROOT/${name}.log"
  echo
  echo "================================================================================"
  echo "[RUN] $name"
  echo "[LOG] $log"
  echo "================================================================================"
  "${BASE_CMD[@]}" "$@" 2>&1 | tee "$log"
}

# Core tests:
# 1. Full-ish chunk, skip anchor/current slot.
run_case "warm_mimic_replan60_offset1" \
  --replan_steps 60 \
  --execute_start_offset 1

# 2. Full-ish chunk, execute from anchor/current slot.
run_case "warm_mimic_replan60_offset0" \
  --replan_steps 60 \
  --execute_start_offset 0

# 3. Medium chunk.
run_case "warm_mimic_replan30_offset1" \
  --replan_steps 30 \
  --execute_start_offset 1

echo
echo "[DONE] Logs saved under: $OUT_ROOT"
echo "Tip:"
echo "  grep -R \"RESULT\\|ever_success\\|obj_local\\|action=\" \"$OUT_ROOT\" | tail -100"
