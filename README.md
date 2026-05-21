# SO101 Isaac Lab → Isaac Mimic → LeRobot / VLA Foundry Pipeline

This repository contains an Isaac Lab workflow for an SO-ARM101 / SO101 cube pick-and-place task. The pipeline supports dataset conversion, Isaac Lab replay, Isaac Mimic generation, camera rendering, LeRobot export, VLA Foundry preprocessing/training, offline diagnostics, and live Isaac Lab policy evaluation.

The repository should track **code, configuration, and documentation only**. Large datasets, generated videos, model checkpoints, logs, and rendered artifacts should stay outside Git.

<img width="1776" height="1121" alt="Annotation_Visual_Debug" src="https://github.com/user-attachments/assets/7fe691a1-3350-46d9-b368-adaa69a81724" />
(Subtask Annotation Visual Debugger, tools/mimic/annotate_demos_so101.py --debug) 

---

## Contents

- [Repository layout](#repository-layout)
- [Environment strategy](#environment-strategy)
- [Important conventions](#important-conventions)
- [End-to-end workflow](#end-to-end-workflow)
- [Step 0: configure paths](#step-0-configure-paths)
- [Step 1: convert source LeRobot data to Isaac HDF5](#step-1-convert-source-lerobot-data-to-isaac-hdf5)
- [Step 2: calibrate and replay the scene](#step-2-calibrate-and-replay-the-scene)
- [Step 3: filter successful replays](#step-3-filter-successful-replays)
- [Step 4: annotate demonstrations for Isaac Mimic](#step-4-annotate-demonstrations-for-isaac-mimic)
- [Step 5: generate synthetic Mimic trajectories](#step-5-generate-synthetic-mimic-trajectories)
- [Step 6: render camera observations for generated trajectories](#step-6-render-camera-observations-for-generated-trajectories)
- [Step 7: export camera-augmented Isaac HDF5 to LeRobot](#step-7-export-camera-augmented-isaac-hdf5-to-lerobot)
- [Step 8: preprocess LeRobot data for VLA Foundry](#step-8-preprocess-lerobot-data-for-vla-foundry)
- [Step 9: train a VLA Foundry diffusion policy](#step-9-train-a-vla-foundry-diffusion-policy)
- [Step 10: run offline VLA diagnostics](#step-10-run-offline-vla-diagnostics)
- [Step 11: run live Isaac Lab VLA evaluation](#step-11-run-live-isaac-lab-vla-evaluation)
- [Common pitfalls](#common-pitfalls)
- [Git and data hygiene](#git-and-data-hygiene)

---

## Repository layout

```text
src/isaac_so_arm101/
  robots/
    i4h_so101/                     # Isaac 4 Healthcare SO101 USD robot configuration
    trs_so101/                     # Original / URDF-based SO101 support
  tasks/
    cube_replay_i4h/               # I4H replay environments and success checks
    cube_mimic_i4h/                # I4H Isaac Mimic environments
    camera_config/                 # camera sensor configuration

tools/
  data/
    convert_lerobot_to_isaac_hdf5_so101.py
    convert_isaac_hdf5_to_lerobot_so101.py
    inspect_hdf5.py
    replay_dataset_with_cameras.py

  mimic/
    annotate_demos_so101.py
    generate_dataset_so101.py

  debug/
    calibrate_cube_scene.py
    filter_successful_replays.py
    debug_so101_frame_audit.py
    debug_replay_annotated_eef_targets.py

  eval/
    eval_vla_foundry_so101.py

  vla_foundry_so101/
    preprocess_so101.py
    train_vla.py
    vla_vibe_check.py
    replay_gt_episode_isaaclab.py
    vla_anchor_sweep.py
    eval_vla_isaaclab_example.sh
    README_DIAGNOSTICS.md
```

---

## Environment strategy

This project currently uses two environments:

```text
Isaac Lab / Isaac Sim environment
  Use for Isaac Lab replay, Isaac Mimic, camera rendering, and live simulation evaluation.

VLA Foundry environment
  Use for VLA Foundry preprocessing, training, and offline diagnostics.
```

A single merged environment is convenient, but it is not required. Keeping training/preprocessing in the VLA Foundry environment reduces the risk of breaking Isaac Lab dependencies, especially packages such as NumPy, PyTorch, Transformers, Ray, and WebDataset.

---

## Important conventions

### Calibrated robot root pose

For the I4H SO101 USD asset, the calibrated root pose used in this pipeline is:

```text
root position:   -0.02079, -0.01576, -0.03248
root quaternion:  0.707,    0.0,      0.0,      0.707   # wxyz
```

### Wrist-roll offset

When converting original LeRobot demonstrations into Isaac HDF5, use:

```text
--wrist-roll-offset-deg -155
```

### Physics and control rate

The replay and camera task commonly use:

```text
sim.dt      = 0.005
decimation  = 4
env step dt = 0.02
control FPS = 50
```

If VLA training data is exported at another FPS, the live policy evaluator may need to respect the policy FPS during simulation.

### Camera task versus no-camera task

Use separate task variants for speed and reproducibility:

```text
Camera-disabled replay task
  Use for fast replay, filtering, annotation, and dataset checks.

Camera-enabled replay task
  Use for camera rendering, visual calibration, and VLA evaluation.
```

When running any camera-enabled task, pass:

```bash
--enable_cameras
```

---

## End-to-end workflow

```text
LeRobot source dataset
  ↓
Isaac HDF5 replay dataset
  ↓
Isaac Lab replay / calibration
  ↓
successful replay HDF5
  ↓
Isaac Mimic annotation
  ↓
Isaac Mimic synthetic generation
  ↓
camera rendering replay
  ↓
camera-augmented Isaac HDF5
  ↓
LeRobot / VLA Foundry-compatible export
  ↓
VLA Foundry preprocessing
  ↓
VLA training
  ↓
offline VLA diagnostics
  ↓
live Isaac Lab VLA evaluation
```

---

## Step 0: configure paths

For copy-paste use, set shell variables once per terminal.

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export SO101_REPO=/path/to/so101_IsaacLab
export VLA_FOUNDRY_ROOT=/path/to/vla_foundry

export SOURCE_LEROBOT=/path/to/source_lerobot_dataset
export REFERENCE_INFO=/path/to/source_lerobot_dataset/meta/info.json

export DATASET_DIR=$SO101_REPO/datasets
mkdir -p "$DATASET_DIR"
```

For Isaac Lab commands:

```bash
cd "$SO101_REPO"
```

For VLA Foundry commands:

```bash
cd "$VLA_FOUNDRY_ROOT"
```

---

## Step 1: convert source LeRobot data to Isaac HDF5

### One episode

```bash
python "$SO101_REPO/tools/data/convert_lerobot_to_isaac_hdf5_so101.py" \
  --repo-id "$SOURCE_LEROBOT" \
  --root "$SOURCE_LEROBOT" \
  --env-name Isaac-SO-ARM101-Cube-I4H-Replay-v0 \
  --episode-index 0 \
  --root-pos="-0.02079,-0.01576,-0.03248" \
  --root-rot-wxyz="0.707,0.0,0.0,0.707" \
  --wrist-roll-offset-deg -155 \
  --output-file "$DATASET_DIR/source_episode_000000_i4h.hdf5"
```

### Full dataset

```bash
python "$SO101_REPO/tools/data/convert_lerobot_to_isaac_hdf5_so101.py" \
  --repo-id "$SOURCE_LEROBOT" \
  --root "$SOURCE_LEROBOT" \
  --env-name Isaac-SO-ARM101-Cube-I4H-Replay-v0 \
  --root-pos="-0.02079,-0.01576,-0.03248" \
  --root-rot-wxyz="0.707,0.0,0.0,0.707" \
  --wrist-roll-offset-deg -155 \
  --output-file "$DATASET_DIR/source_all_i4h.hdf5"
```

---

## Step 2: calibrate and replay the scene

Use the interactive calibration script to verify base pose, object pose, gripper pose, goal region, and camera views.

### Direct pose check

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/debug/calibrate_cube_scene.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --dataset_file "$DATASET_DIR/source_episode_000000_i4h.hdf5" \
  --episode_index 0 \
  --sample_index 100 \
  --mode direct_pose \
  --save_camera_debug \
  --enable_cameras
```

### Action-step replay check

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/debug/calibrate_cube_scene.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --dataset_file "$DATASET_DIR/source_episode_000000_i4h.hdf5" \
  --episode_index 0 \
  --sample_index 200 \
  --mode action_step \
  --step_size 20 \
  --save_camera_debug \
  --enable_cameras
```

Use `action_step` when you want to reset to the episode start and replay actions sequentially up to a target frame.

---

## Step 3: filter successful replays

Use the no-camera task for speed.

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/debug/filter_successful_replays.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-v0 \
  --input-hdf5 "$DATASET_DIR/source_all_i4h.hdf5" \
  --output-hdf5 "$DATASET_DIR/source_successful_i4h.hdf5" \
  --check-mode ever_success
```

---

## Step 4: annotate demonstrations for Isaac Mimic

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/mimic/annotate_demos_so101.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0 \
  --input_file "$DATASET_DIR/source_successful_i4h.hdf5" \
  --output_file "$DATASET_DIR/source_successful_i4h_annotated_$(date +%Y%m%d_%H%M%S).hdf5" \
  --headless
```

Inspect the result:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/data/inspect_hdf5.py" \
  "$DATASET_DIR/source_successful_i4h_annotated.hdf5" \
  --episode 0 \
  --stats
```

---

## Step 5: generate synthetic Mimic trajectories

### Small debug run

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/mimic/generate_dataset_so101.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0 \
  --input_file "$DATASET_DIR/source_successful_i4h_annotated.hdf5" \
  --output_file "$DATASET_DIR/generated_mimic_debug_$(date +%Y%m%d_%H%M%S).hdf5" \
  --num_envs 1 \
  --generation_num_trials 1 \
  --debug
```

### Larger headless run

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/mimic/generate_dataset_so101.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0 \
  --input_file "$DATASET_DIR/source_successful_i4h_annotated.hdf5" \
  --output_file "$DATASET_DIR/generated_mimic_$(date +%Y%m%d_%H%M%S).hdf5" \
  --num_envs 32 \
  --generation_num_trials 96 \
  --headless
```

---

## Step 6: render camera observations for generated trajectories

Generate trajectories without cameras first, then replay them in the camera-enabled task to record images.

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/data/replay_dataset_with_cameras.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --input_file "$DATASET_DIR/generated_mimic.hdf5" \
  --output_file "$DATASET_DIR/generated_mimic_with_cameras.hdf5" \
  --headless \
  --enable_cameras \
  --overwrite
```

Smoke test only a few episodes:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/data/replay_dataset_with_cameras.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --input_file "$DATASET_DIR/generated_mimic.hdf5" \
  --output_file "$DATASET_DIR/generated_mimic_with_cameras_smoke.hdf5" \
  --max_episodes 2 \
  --headless \
  --enable_cameras \
  --overwrite
```

Inspect camera datasets:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/data/inspect_hdf5.py" \
  "$DATASET_DIR/generated_mimic_with_cameras_smoke.hdf5" \
  --check-cameras \
  --stats \
  --episode 0
```

Expected camera datasets:

```text
data/demo_X/camera_obs/wrist  [T, H, W, 3] uint8
data/demo_X/camera_obs/up     [T, H, W, 3] uint8
```

---

## Step 7: export camera-augmented Isaac HDF5 to LeRobot

### Standard LeRobot export

```bash
python "$SO101_REPO/tools/data/convert_isaac_hdf5_to_lerobot_so101.py" \
  --input_file "$DATASET_DIR/generated_mimic_with_cameras.hdf5" \
  --out "$DATASET_DIR/lerobot_mimic_camera" \
  --reference-info "$REFERENCE_INFO" \
  --fallback-source-fps 50 \
  --task "Pick up the cube and place it in the goal region." \
  --only-with-cameras \
  --video-codec libx264 \
  --video-pix-fmt yuv420p \
  --overwrite
```

### VLA Foundry-compatible LeRobot export

```bash
python "$SO101_REPO/tools/data/convert_isaac_hdf5_to_lerobot_so101.py" \
  --input_file "$DATASET_DIR/generated_mimic_with_cameras.hdf5" \
  --out "$DATASET_DIR/lerobot_mimic_vla_compat" \
  --reference-info "$REFERENCE_INFO" \
  --fallback-source-fps 50 \
  --task "Pick up the cube and place it in the goal region." \
  --only-with-cameras \
  --video-codec libx264 \
  --video-pix-fmt yuv420p \
  --vla-foundry-compat \
  --overwrite
```

For a small subset:

```bash
python "$SO101_REPO/tools/data/convert_isaac_hdf5_to_lerobot_so101.py" \
  --input_file "$DATASET_DIR/generated_mimic_with_cameras.hdf5" \
  --out "$DATASET_DIR/lerobot_mimic_vla_compat_subset" \
  --reference-info "$REFERENCE_INFO" \
  --fallback-source-fps 50 \
  --task "Pick up the cube and place it in the goal region." \
  --episode-indices 0,1 \
  --only-with-cameras \
  --video-codec libx264 \
  --video-pix-fmt yuv420p \
  --vla-foundry-compat \
  --overwrite
```

Inspect exported columns:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

dataset = Path("datasets/lerobot_mimic_vla_compat_subset")
parquet = sorted((dataset / "data").glob("**/*.parquet"))[0]
df = pd.read_parquet(parquet)

print(parquet)
print(df.columns.tolist())
print(df.head())
PY
```

Expected image columns:

```text
observation.images.wrist
observation.images.up
```

---

## Step 8: preprocess LeRobot data for VLA Foundry

Use the VLA Foundry environment.

```bash
cd "$VLA_FOUNDRY_ROOT"
```

Preprocess the exported LeRobot dataset:

```bash
uv run python "$SO101_REPO/tools/vla_foundry_so101/preprocess_so101.py" \
  --compat-root "$DATASET_DIR/lerobot_mimic_vla_compat_subset" \
  --output-root "$VLA_FOUNDRY_ROOT/tutorials/data/so101_mimic_preprocessed" \
  --past-lowdim-steps 2 \
  --future-lowdim-steps 60 \
  --resize 224 224 \
  --num-workers 1
```

The converter uses full feature names during preprocessing:

```text
observation.images.wrist
observation.images.up
```

---

## Step 9: train a VLA Foundry diffusion policy

Use the VLA Foundry environment.

```bash
cd "$VLA_FOUNDRY_ROOT"
```

Train:

```bash
uv run python "$SO101_REPO/tools/vla_foundry_so101/train_vla.py" \
  --preproc-root "$VLA_FOUNDRY_ROOT/tutorials/data/so101_mimic_preprocessed" \
  --past-lowdim-timesteps 2 \
  --future-lowdim-timesteps 60 \
  --per-gpu-batch-size 16 \
  --global-batch-size 16 \
  --total-train-samples 300000 \
  --num-checkpoints 10 \
  --max-checkpoint-limit 10
```

The training script uses short camera names:

```text
wrist
up
```

The future low-dimensional timestep value controls the future action horizon. The number of diffusion denoising steps is controlled later during inference/evaluation.

---

## Step 10: run offline VLA diagnostics

### Single-sample action prediction check

```bash
cd "$VLA_FOUNDRY_ROOT"

uv run python "$SO101_REPO/tools/vla_foundry_so101/vla_vibe_check.py" \
  --checkpoint-dir /path/to/vla_checkpoint_dir \
  --preproc-root "$VLA_FOUNDRY_ROOT/tutorials/data/so101_mimic_preprocessed" \
  --sample-index 0 \
  --num-inference-steps 10 \
  --out-dir "$VLA_FOUNDRY_ROOT/tutorials/diagnostics/vla_vibe_check"
```

This plots:

```text
ground-truth normalized action vs predicted normalized action
ground-truth denormalized action vs predicted denormalized action
camera views used by the policy
```

### Anchor sweep

```bash
cd "$VLA_FOUNDRY_ROOT"

uv run python "$SO101_REPO/tools/vla_foundry_so101/vla_anchor_sweep.py" \
  --checkpoint-dir /path/to/vla_checkpoint_dir \
  --preproc-root "$VLA_FOUNDRY_ROOT/tutorials/data/so101_mimic_preprocessed" \
  --max-samples 20 \
  --num-inference-steps 10 \
  --out-dir "$VLA_FOUNDRY_ROOT/tutorials/diagnostics/vla_anchor_sweep"
```

Use this to determine whether the model predicts well across approach, grasp, lift, and place phases before running live simulation.

---

## Step 11: run live Isaac Lab VLA evaluation

Use the Isaac Lab environment, but expose the VLA Foundry repository on `PYTHONPATH`.

```bash
export PYTHONUNBUFFERED=1
export VLA_FOUNDRY_ROOT=/path/to/vla_foundry
export PYTHONPATH="$VLA_FOUNDRY_ROOT:$PYTHONPATH"
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
```

Run:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$SO101_REPO/tools/eval/eval_vla_foundry_so101.py" \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --checkpoint_dir /path/to/vla_checkpoint_dir \
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

Key runtime settings:

```text
replan_steps
  Number of predicted future actions to execute before asking the model for a new action chunk.

execute_start_offset
  Offset after the past-action slots. Use this to skip the anchor/current slot if needed.

num_inference_steps
  Number of diffusion denoising steps used when generating an action chunk.

respect_policy_fps
  Repeats policy actions as needed so training FPS and simulation control FPS are not accidentally mismatched.

action_smoothing
  Low-pass smoothing applied to executed denormalized actions.
```

---

## Common pitfalls

### Negative command-line values

Bad:

```bash
--root-pos -0.02079,-0.01576,-0.03248
```

Good:

```bash
--root-pos="-0.02079,-0.01576,-0.03248"
```

### Camera-enabled tasks require `--enable_cameras`

If a camera-enabled task is launched without this flag, Isaac Lab can fail during camera sensor initialization.

### Do not assume video count from directory count

Use:

```bash
find /path/to/dataset/videos -type f | wc -l
```

not:

```bash
find /path/to/dataset/videos | wc -l
```

because the second form counts directories too.

### LeRobot subsets must use compact episode indices

If a subset contains only two episodes, make sure episode indices are compact:

```text
0, 1
```

not sparse original IDs such as:

```text
27, 65
```

Some preprocessing code indexes episode metadata using `entries[episode_index]`, so sparse episode IDs can produce `IndexError`.

### VLA Foundry relative checkpoint paths

VLA Foundry model configs may contain relative checkpoint paths. Run offline diagnostics from the VLA Foundry repository root, or ensure the evaluator changes current working directory to the VLA Foundry root before model construction.

### Large batch size can reduce optimizer steps

For overfit diagnostics, count optimizer steps:

```text
optimizer steps ≈ total_train_samples / global_batch_size
```

A large batch can make a run look long in sample count but short in optimizer updates.

---

## Git and data hygiene

Commit:

```text
src/
tools/
docs/
README.md
pyproject.toml
LICENSE
CITATION.cff
.gitignore
```

Do not commit:

```text
datasets/
logs/
outputs/
*.hdf5
*.mp4
*.pt
*.pth
*.ckpt
*.tar
*.parquet
*.png
*.jpg
*.jpeg
.local_env_snapshots/
README_old*.md
```

For environment reproducibility, prefer curated files under:

```text
docs/env/
```

such as:

```text
env_from_history.yml
env_full_export.yml
key_versions.txt
notes.md
```

Avoid committing large or noisy full `pip freeze` dumps unless they are explicitly needed for debugging.
