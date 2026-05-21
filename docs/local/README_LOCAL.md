# Local SO101 VLA Pipeline Notes

This file is intended for local convenience only. It contains machine-specific paths and commands.

---

## Local paths

```bash
export ISAACLAB_ROOT=/home/insol02/IH_ws/IsaacLab
export SO101_REPO=/home/insol02/IH_ws/so101_IsaacLab
export VLA_FOUNDRY_ROOT=/home/insol02/IH_ws/vla_foundry

export DATASET_DIR=$SO101_REPO/datasets
export SOURCE_LEROBOT=$DATASET_DIR/so101_pickplace_cube_1020_same_place
export REFERENCE_INFO=$SOURCE_LEROBOT/meta/info.json
```

---

## Environment use

### Isaac Lab / live simulation

```bash
conda activate env_so101_vla_isaaclab
```

### VLA Foundry preprocessing and training

```bash
cd /home/insol02/IH_ws/vla_foundry
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## VLA Foundry inference inside Isaac Lab

Before running the live evaluator:

```bash
conda activate env_so101_vla_isaaclab

export PYTHONUNBUFFERED=1
export VLA_FOUNDRY_ROOT=/home/insol02/IH_ws/vla_foundry
export PYTHONPATH="$VLA_FOUNDRY_ROOT:$PYTHONPATH"
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
```

---

## Check dataset replay at a specific episode/frame

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/debug/calibrate_cube_scene.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --dataset_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_debug_32env_96trial_20260519_184403.hdf5 \
  --episode_index 27 \
  --sample_index 70 \
  --mode action_step \
  --step_size 100 \
  --save_camera_debug \
  --enable_cameras
```

---

## Ground-truth replay diagnostic

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/replay_gt_episode_isaaclab.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --dataset-file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_debug_32env_96trial_20260519_184403.hdf5 \
  --episode-index 27 \
  --start-index 0 \
  --end-index 403 \
  --debug-every 20 \
  --save-camera-debug \
  --enable_cameras
```

---

## Build a compact LeRobot subset

Use compact episode indices for VLA Foundry preprocessing. The subset should remap selected source episodes to `0, 1, ...`.

```bash
cd /home/insol02/IH_ws/vla_foundry

uv run python /tmp/make_lerobot_compact_subset.py \
  --src /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_i4h_lerobot_mimic_vla_compat \
  --dst /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_i4h_lerobot_mimic_vla_compat_subset \
  --episodes 27 65 \
  --overwrite
```

Verify:

```bash
cat /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_i4h_lerobot_mimic_vla_compat_subset/meta/episodes.jsonl

find /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_i4h_lerobot_mimic_vla_compat_subset/videos -type f | sort
```

---

## Preprocess compact subset for VLA Foundry

```bash
cd /home/insol02/IH_ws/vla_foundry

uv run python /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/preprocess_so101.py \
  --compat-root /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_i4h_lerobot_mimic_vla_compat_subset \
  --output-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_subset_preprocessed \
  --past-lowdim-steps 2 \
  --future-lowdim-steps 60 \
  --resize 224 224 \
  --num-workers 1
```

---

## Train VLA policy

```bash
cd /home/insol02/IH_ws/vla_foundry

uv run python /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/train_vla.py \
  --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_subset_preprocessed \
  --past-lowdim-timesteps 2 \
  --future-lowdim-timesteps 60 \
  --per-gpu-batch-size 16 \
  --global-batch-size 16 \
  --total-train-samples 300000 \
  --num-checkpoints 10 \
  --max-checkpoint-limit 10
```

---

## Offline VLA single-sample diagnostic

```bash
cd /home/insol02/IH_ws/vla_foundry

uv run python /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/vla_vibe_check.py \
  --checkpoint-dir /home/insol02/IH_ws/vla_foundry/tutorials/checkpoints/<checkpoint_dir> \
  --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_subset_preprocessed \
  --sample-index 0 \
  --num-inference-steps 10 \
  --out-dir /home/insol02/IH_ws/vla_foundry/tutorials/diagnostics/so101_subset_vibe_check
```

---

## Offline VLA anchor sweep

```bash
cd /home/insol02/IH_ws/vla_foundry

uv run python /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/vla_anchor_sweep.py \
  --checkpoint-dir /home/insol02/IH_ws/vla_foundry/tutorials/checkpoints/<checkpoint_dir> \
  --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_subset_preprocessed \
  --max-samples 20 \
  --num-inference-steps 10 \
  --out-dir /home/insol02/IH_ws/vla_foundry/tutorials/diagnostics/so101_subset_anchor_sweep
```

---

## Live Isaac Lab VLA evaluation

```bash
conda activate env_so101_vla_isaaclab

export PYTHONUNBUFFERED=1
export VLA_FOUNDRY_ROOT=/home/insol02/IH_ws/vla_foundry
export PYTHONPATH="$VLA_FOUNDRY_ROOT:$PYTHONPATH"
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/eval/eval_vla_foundry_so101.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --checkpoint_dir /home/insol02/IH_ws/vla_foundry/tutorials/checkpoints/<checkpoint_dir> \
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
```

---

## Local cleanup reminders

Do not commit:

```text
datasets/
logs/
outputs/
tutorials/checkpoints/
*.hdf5
*.mp4
*.pt
*.tar
*.parquet
.local_env_snapshots/
```

Keep curated environment notes under:

```text
docs/env/
```
