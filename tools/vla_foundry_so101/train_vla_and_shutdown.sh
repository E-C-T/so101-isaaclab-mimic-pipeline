#!/usr/bin/env bash
set -euo pipefail

# Long VLA training job with optional shutdown.
#
# Single-dataset example:
#   PREPROC_ROOT=/path/to/preprocessed \
#   SHUTDOWN_ON_FINISH=1 \
#   bash train_vla_and_shutdown.sh
#
# Mixed-dataset example:
#   MIMIC_PREPROC=/path/to/mimic/preprocessed \
#   BASE_PREPROC=/path/to/base/preprocessed \
#   SHUTDOWN_ON_FINISH=1 \
#   bash train_vla_and_shutdown.sh
#
# Notes:
# - Uses VLA Foundry's uv environment.
# - Shutdown may require sudo privileges.
# - By default, shutdown is OFF for safety. Set SHUTDOWN_ON_FINISH=1 to shut down
#   after training finishes, regardless of success or failure.
# - If you want Ctrl-C / SIGTERM to also trigger shutdown, set SHUTDOWN_ON_INTERRUPT=1.

SO101_REPO="${SO101_REPO:-/home/insol02/IH_ws/so101_IsaacLab}"
VLA_FOUNDRY_ROOT="${VLA_FOUNDRY_ROOT:-/home/insol02/IH_ws/vla_foundry}"

# Single-dataset mode.
PREPROC_ROOT="${PREPROC_ROOT:-}"

# Mixed-dataset mode. If both are set, this script uses direct torchrun with two manifests.
MIMIC_PREPROC="${MIMIC_PREPROC:-}"
BASE_PREPROC="${BASE_PREPROC:-}"
MIMIC_WEIGHT="${MIMIC_WEIGHT:-0.75}"
BASE_WEIGHT="${BASE_WEIGHT:-0.25}"

PAST_TIMESTEPS="${PAST_TIMESTEPS:-2}"
FUTURE_TIMESTEPS="${FUTURE_TIMESTEPS:-60}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
TOTAL_TRAIN_SAMPLES="${TOTAL_TRAIN_SAMPLES:-1500000}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
MAX_CHECKPOINT_LIMIT="${MAX_CHECKPOINT_LIMIT:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROPRIOCEPTION_FIELDS="${PROPRIOCEPTION_FIELDS:-[\"observation.state\"]}"

# New shutdown behavior:
#   SHUTDOWN_ON_FINISH=1 schedules shutdown after success OR failure.
# Backward-compatible aliases are still honored.
SHUTDOWN_ON_FINISH="${SHUTDOWN_ON_FINISH:-0}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-0}"
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-0}"
SHUTDOWN_ON_INTERRUPT="${SHUTDOWN_ON_INTERRUPT:-0}"
SHUTDOWN_DELAY_MINUTES="${SHUTDOWN_DELAY_MINUTES:-1}"

LOG_DIR="${LOG_DIR:-$VLA_FOUNDRY_ROOT/tutorials/logs/long_train_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train.log"

FINALIZED=0

optimizer_steps=$(( TOTAL_TRAIN_SAMPLES / GLOBAL_BATCH_SIZE ))
optimizer_remainder=$(( TOTAL_TRAIN_SAMPLES % GLOBAL_BATCH_SIZE ))

schedule_shutdown() {
  local status="$1"

  # Backward compatibility: old flags still work.
  local should_shutdown="0"
  if [[ "$SHUTDOWN_ON_FINISH" == "1" ]]; then
    should_shutdown="1"
  elif [[ "$status" -eq 0 && "$SHUTDOWN_ON_SUCCESS" == "1" ]]; then
    should_shutdown="1"
  elif [[ "$status" -ne 0 && "$SHUTDOWN_ON_FAILURE" == "1" ]]; then
    should_shutdown="1"
  fi

  # Do not shut down on manual interrupt unless explicitly requested.
  if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
    if [[ "$SHUTDOWN_ON_INTERRUPT" != "1" ]]; then
      echo "[INFO] Interrupted/terminated with status $status. Shutdown skipped."
      echo "[INFO] Set SHUTDOWN_ON_INTERRUPT=1 if you want interrupts to also shut down."
      return 0
    fi
  fi

  if [[ "$should_shutdown" != "1" ]]; then
    echo "[INFO] Shutdown disabled. Set SHUTDOWN_ON_FINISH=1 to shut down after success or failure."
    return 0
  fi

  echo "[INFO] Scheduling shutdown in ${SHUTDOWN_DELAY_MINUTES} minute(s)."
  echo "[INFO] Cancel with: sudo shutdown -c"
  sudo shutdown -h +"$SHUTDOWN_DELAY_MINUTES" "VLA training finished with exit status $status"
}

finish() {
  local status="$?"
  if [[ "$FINALIZED" == "1" ]]; then
    return 0
  fi
  FINALIZED=1

  echo "[INFO] Final exit status: $status"
  echo "[INFO] Log file: $LOG_FILE"

  schedule_shutdown "$status"
  return "$status"
}

trap finish EXIT
trap 'echo "[WARN] Received SIGINT."; exit 130' INT
trap 'echo "[WARN] Received SIGTERM."; exit 143' TERM

echo "[INFO] VLA_FOUNDRY_ROOT=$VLA_FOUNDRY_ROOT"
echo "[INFO] SO101_REPO=$SO101_REPO"
echo "[INFO] PREPROC_ROOT=$PREPROC_ROOT"
echo "[INFO] MIMIC_PREPROC=$MIMIC_PREPROC"
echo "[INFO] BASE_PREPROC=$BASE_PREPROC"
echo "[INFO] MIMIC_WEIGHT=$MIMIC_WEIGHT"
echo "[INFO] BASE_WEIGHT=$BASE_WEIGHT"
echo "[INFO] PAST_TIMESTEPS=$PAST_TIMESTEPS"
echo "[INFO] FUTURE_TIMESTEPS=$FUTURE_TIMESTEPS"
echo "[INFO] PER_GPU_BATCH_SIZE=$PER_GPU_BATCH_SIZE"
echo "[INFO] GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "[INFO] TOTAL_TRAIN_SAMPLES=$TOTAL_TRAIN_SAMPLES"
echo "[INFO] Approx optimizer steps: $optimizer_steps"
if [[ "$optimizer_remainder" -ne 0 ]]; then
  echo "[INFO] Optimizer-step estimate has remainder samples: $optimizer_remainder"
fi
echo "[INFO] NUM_CHECKPOINTS=$NUM_CHECKPOINTS"
echo "[INFO] MAX_CHECKPOINT_LIMIT=$MAX_CHECKPOINT_LIMIT"
echo "[INFO] NUM_WORKERS=$NUM_WORKERS"
echo "[INFO] PROPRIOCEPTION_FIELDS=$PROPRIOCEPTION_FIELDS"
echo "[INFO] SHUTDOWN_ON_FINISH=$SHUTDOWN_ON_FINISH"
echo "[INFO] SHUTDOWN_DELAY_MINUTES=$SHUTDOWN_DELAY_MINUTES"
echo "[INFO] LOG_FILE=$LOG_FILE"

cd "$VLA_FOUNDRY_ROOT"

set +e

if [[ -n "$MIMIC_PREPROC" && -n "$BASE_PREPROC" ]]; then
  echo "[INFO] Running mixed-dataset training."

  VLM_CKPT=$(find tutorials/checkpoints -path "*model_vlm*/checkpoints/checkpoint_*.pt" | sort -V | tail -1)
  if [[ -z "${VLM_CKPT}" ]]; then
    echo "[ERROR] No VLM checkpoint found under tutorials/checkpoints."
    exit 1
  fi

  MIMIC_MANIFEST="$MIMIC_PREPROC/shards/manifest.jsonl"
  MIMIC_STATS="$MIMIC_PREPROC/shards/stats.json"
  BASE_MANIFEST="$BASE_PREPROC/shards/manifest.jsonl"
  BASE_STATS="$BASE_PREPROC/shards/stats.json"

  for f in "$MIMIC_MANIFEST" "$MIMIC_STATS" "$BASE_MANIFEST" "$BASE_STATS"; do
    if [[ ! -f "$f" ]]; then
      echo "[ERROR] Missing required file: $f"
      exit 1
    fi
  done

  echo "[INFO] VLM_CKPT=$VLM_CKPT"
  echo "[INFO] MIMIC_MANIFEST=$MIMIC_MANIFEST"
  echo "[INFO] BASE_MANIFEST=$BASE_MANIFEST"

  uv run torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --master_port=0 \
    vla_foundry/main.py \
    --model "include vla_foundry/config_presets/models/vla_diffusion_100m.yaml" \
    --model.vision_language_backbone.resume_from_checkpoint "$VLM_CKPT" \
    --distributed.fsdp False \
    --data.type robotics \
    --data.processor simple_vlm \
    --data.image_size 224 \
    --data.img_num_tokens 256 \
    --data.seq_len 2048 \
    --data.dataset_manifest "[\"$MIMIC_MANIFEST\",\"$BASE_MANIFEST\"]" \
    --data.dataset_statistics "[\"$MIMIC_STATS\",\"$BASE_STATS\"]" \
    --data.dataset_modality "[\"robotics\",\"robotics\"]" \
    --data.dataset_weighting "[$MIMIC_WEIGHT,$BASE_WEIGHT]" \
    --data.camera_names "[\"wrist\",\"up\"]" \
    --data.action_fields "[\"action\"]" \
    --data.proprioception_fields "$PROPRIOCEPTION_FIELDS" \
    --data.language_instruction_types "[\"original\"]" \
    --data.pose_groups "[]" \
    --data.intrinsics_fields "[]" \
    --data.extrinsics_fields "[]" \
    --data.lowdim_past_timesteps "$PAST_TIMESTEPS" \
    --data.lowdim_future_timesteps "$FUTURE_TIMESTEPS" \
    --data.allow_multiple_epochs True \
    --data.num_workers "$NUM_WORKERS" \
    --hparams "include vla_foundry/config_presets/hparams/diffusion_policy.yaml" \
    --hparams.per_gpu_batch_size "$PER_GPU_BATCH_SIZE" \
    --hparams.global_batch_size "$GLOBAL_BATCH_SIZE" \
    --hparams.warmup 200 \
    --total_train_samples "$TOTAL_TRAIN_SAMPLES" \
    --num_checkpoints "$NUM_CHECKPOINTS" \
    --max_checkpoint_limit "$MAX_CHECKPOINT_LIMIT" \
    --save_path ./tutorials/checkpoints \
    --wandb False \
    --db_logging False \
    2>&1 | tee "$LOG_FILE"

  STATUS=${PIPESTATUS[0]}
else
  echo "[INFO] Running single-dataset training."

  if [[ -z "$PREPROC_ROOT" ]]; then
    PREPROC_ROOT="$VLA_FOUNDRY_ROOT/tutorials/data/so101_mimic96_abs_jointstate/preprocessed"
    echo "[INFO] PREPROC_ROOT was unset; using default: $PREPROC_ROOT"
  fi

  if [[ ! -d "$PREPROC_ROOT" ]]; then
    echo "[ERROR] PREPROC_ROOT does not exist: $PREPROC_ROOT"
    exit 1
  fi

  uv run python "$SO101_REPO/tools/vla_foundry_so101/train_vla.py" \
    --preproc-root "$PREPROC_ROOT" \
    --past-lowdim-timesteps "$PAST_TIMESTEPS" \
    --future-lowdim-timesteps "$FUTURE_TIMESTEPS" \
    --per-gpu-batch-size "$PER_GPU_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --total-train-samples "$TOTAL_TRAIN_SAMPLES" \
    --num-checkpoints "$NUM_CHECKPOINTS" \
    --max-checkpoint-limit "$MAX_CHECKPOINT_LIMIT" \
    2>&1 | tee "$LOG_FILE"

  STATUS=${PIPESTATUS[0]}
fi

set -e

echo "[INFO] training exit status: $STATUS"

if [[ "$STATUS" -eq 0 ]]; then
  echo "[INFO] Training completed successfully."
else
  echo "[ERROR] Training failed. See $LOG_FILE"
fi

exit "$STATUS"