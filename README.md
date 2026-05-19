# SO101 Isaac Lab → Isaac Mimic → LeRobot/VLA Foundry Pipeline

This repository contains a working Isaac Lab pipeline for the SO-ARM101 / SO101 cube pick-and-place task using the Isaac 4 Healthcare SO101 USD asset.

The current pipeline supports:

1. Converting SO101 LeRobot demonstrations into Isaac HDF5.
2. Replaying and filtering successful Isaac trajectories.
3. Annotating demonstrations for Isaac Mimic.
4. Generating synthetic trajectories with Isaac Mimic.
5. Replaying generated trajectories with camera sensors.
6. Exporting camera-augmented Isaac HDF5 datasets back to LeRobot format.
7. Creating VLA Foundry-compatible LeRobot exports for robotics preprocessing/training smoke tests.

> This repository should track code, configs, and documentation only. Large datasets, generated HDF5 files, rendered videos, logs, and model checkpoints should stay out of Git.

---

## 1. Repository layout

```text
src/isaac_so_arm101/
  robots/
    i4h_so101/                     # Isaac 4 Healthcare SO101 USD robot config
    trs_so101/                     # Original/URDF-based SO101 support
  tasks/
    cube_replay_i4h/               # I4H replay environments and success checks
    cube_mimic_i4h/                # I4H Isaac Mimic environments
    camera_config/                 # camera sensor configuration
  scripts/
    calibrate_cube_scene.py        # visual/debug calibration helper

tools/
  data/
    convert_lerobot_to_isaac_hdf5_so101.py
    inspect_hdf5.py
    replay_dataset_with_cameras.py
    convert_isaac_hdf5_to_lerobot_so101.py
  mimic/
    annotate_demos_so101.py
    generate_dataset_so101.py
  debug/
    filter_successful_replays.py
    debug_so101_frame_audit.py
    debug_replay_annotated_eef_targets.py
```

---

## 2. Important I4H SO101 conventions

### 2.1 Calibrated robot root pose

For the I4H SO101 USD asset, use:

```text
root position:   -0.02079, -0.01576, -0.03248
root quaternion:  0.707,    0.0,      0.0,      0.707   # wxyz
```

This compensates for the USD asset/base-frame offset and places the robot correctly in the cube pick-and-place scene.

### 2.2 Wrist-roll offset

When converting the original LeRobot demonstrations into Isaac HDF5, use:

```text
--wrist-roll-offset-deg -155
```

This aligns the wrist/gripper convention of the source demonstrations with the I4H USD asset convention.

### 2.3 Physics/control rate

The current pick/place configuration uses:

```text
sim.dt      = 0.005  # 200 Hz physics
decimation  = 4      # one action/env step every 4 physics steps
env step dt = 0.02   # 50 Hz control/data rate
```

The Isaac HDF5 trajectory rows are recorded per `env.step(...)`, not per raw PhysX substep. So the source FPS for generated state/action datasets is normally:

```text
source_fps = 50
```

### 2.4 Camera vs no-camera tasks

Use separate task variants for speed and reproducibility:

```text
Isaac-SO-ARM101-Cube-I4H-Replay-v0
  no active camera tensors
  use for fast filtering, annotation, and no-camera replay

Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0
  active wrist/up camera sensors
  use for replaying successful trajectories and saving camera observations

Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0
  Isaac Mimic environment using 6D joint-position actions
  use for annotation and generation aligned with original teleop data

Isaac-SO-ARM101-Cube-I4H-Diff-IK-Mimic-v0
  Isaac Mimic environment using differential IK action semantics
  useful for future experiments, but Pinocchio generation is currently the safer path
```

The gripper USD may contain an embedded camera prim that appears in the Isaac Sim viewport. That does not mean Isaac Lab is actively rendering camera tensors. The reliable check is:

```python
print(env.scene.keys())
```

---

## 3. End-to-end pipeline

```text
LeRobot source dataset
  ↓
convert_lerobot_to_isaac_hdf5_so101.py
  ↓
Isaac HDF5 replay dataset
  ↓
calibrate_cube_scene.py
  ↓
filter_successful_replays.py
  ↓
successful replay HDF5
  ↓
annotate_demos_so101.py
  ↓
Isaac Mimic annotated HDF5
  ↓
generate_dataset_so101.py
  ↓
synthetic no-camera Mimic HDF5
  ↓
replay_dataset_with_cameras.py
  ↓
synthetic HDF5 with camera_obs/{wrist,up}
  ↓
convert_isaac_hdf5_to_lerobot_so101.py
  ↓
LeRobot / VLA Foundry-compatible dataset
```

---

## 4. Convert source LeRobot data to Isaac HDF5

### One episode

```bash
python /home/insol02/IH_ws/so101_IsaacLab/tools/data/convert_lerobot_to_isaac_hdf5_so101.py \
  --repo-id /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place \
  --root /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place \
  --env-name Isaac-SO-ARM101-Cube-I4H-Replay-v0 \
  --episode-index 0 \
  --root-pos="-0.02079,-0.01576,-0.03248" \
  --root-rot-wxyz="0.707,0.0,0.0,0.707" \
  --wrist-roll-offset-deg -155 \
  --output-file /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_ep0_calibroot_wr-155.hdf5
```

### Full dataset

```bash
python /home/insol02/IH_ws/so101_IsaacLab/tools/data/convert_lerobot_to_isaac_hdf5_so101.py \
  --repo-id /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place \
  --root /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place \
  --env-name Isaac-SO-ARM101-Cube-I4H-Replay-v0 \
  --root-pos="-0.02079,-0.01576,-0.03248" \
  --root-rot-wxyz="0.707,0.0,0.0,0.707" \
  --wrist-roll-offset-deg -155 \
  --output-file /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_all_calibroot_wr-155.hdf5
```

---

## 5. Calibrate scene/replay

Use calibration before filtering or Mimic generation. Verify that the robot base, wrist/gripper, cube, and goal region visually align.

### Direct pose camera check

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/src/isaac_so_arm101/scripts/calibrate_cube_scene.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --dataset_file /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_ep0_calibroot_wr-155.hdf5 \
  --episode_index 0 \
  --sample_index 300 \
  --mode direct_pose \
  --enable_cameras
```

### Action-step replay check

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/src/isaac_so_arm101/scripts/calibrate_cube_scene.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --dataset_file /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_ep0_calibroot_wr-155.hdf5 \
  --episode_index 0 \
  --sample_index 340 \
  --mode action_step \
  --step_size 20 \
  --enable_cameras
```

---

## 6. Filter successful replays

Use the no-camera replay task for speed.

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/debug/filter_successful_replays.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-v0 \
  --input-hdf5 /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_all_calibroot_wr-155.hdf5 \
  --output-hdf5 /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_successful.hdf5 \
  --check-mode ever_success
```

Recommended:

```text
Use num_envs=1 unless the filtering script has been rewritten for true vectorized episode filtering.
```

Important multi-env success note: success checks must compare object positions in env-local coordinates, not raw world coordinates. If `num_envs > 1` works in GUI but success does not trigger for envs other than env 0, check for `root_state_w` vs `root_state_w - env.scene.env_origins`.

---

## 7. Annotate demos for Isaac Mimic

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/mimic/annotate_demos_so101.py \
  --task Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0 \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_successful.hdf5 \
  --output_file /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_successful_annotated_$(date +%Y%m%d_%H%M%S).hdf5 \
  --headless
```

Avoid overwriting the same annotation file while debugging. If annotation crashes, HDF5 files can be left as tiny invalid files.

### Inspect HDF5

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/data/inspect_hdf5.py \
  /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_successful_annotated.hdf5 \
  --episode 0 \
  --stats
```

### Optional safe copy

```bash
SAFE_ANNOTATED=/home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_successful_annotated_SAFE_$(date +%Y%m%d_%H%M%S).hdf5

cp /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_i4h_successful_annotated.hdf5 "$SAFE_ANNOTATED"
chmod a-w "$SAFE_ANNOTATED"
```

---

## 8. Generate synthetic Mimic data

### Small debug run

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/mimic/generate_dataset_so101.py \
  --task Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0 \
  --input_file "$SAFE_ANNOTATED" \
  --output_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_debug_1env_1trial_$(date +%Y%m%d_%H%M%S).hdf5 \
  --num_envs 1 \
  --generation_num_trials 1 \
  --debug
```

### Moderate headless run

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/mimic/generate_dataset_so101.py \
  --task Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0 \
  --input_file "$SAFE_ANNOTATED" \
  --output_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_32env_96trial_$(date +%Y%m%d_%H%M%S).hdf5 \
  --num_envs 32 \
  --generation_num_trials 96 \
  --headless
```

For small real datasets, start with roughly 2x to 5x synthetic expansion. For 29 teleop episodes, generate about 60–150 synthetic episodes first and validate quality before scaling.

---

## 9. Replay generated trajectories with cameras

Generate synthetic trajectories without cameras first, then replay successful generated trajectories with the camera task. This keeps generation fast and makes rendering a separate deterministic post-process.

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/data/replay_dataset_with_cameras.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_32env_96trial.hdf5 \
  --output_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_32env_96trial_with_cameras.hdf5 \
  --headless \
  --enable_cameras \
  --overwrite
```

Smoke test only two episodes:

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/data/replay_dataset_with_cameras.py \
  --task Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0 \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_32env_96trial.hdf5 \
  --output_file /tmp/generated_with_cameras_smoke.hdf5 \
  --headless \
  --enable_cameras \
  --max_episodes 2 \
  --overwrite
```

Inspect camera presence:

```bash
/home/insol02/IH_ws/IsaacLab/isaaclab.sh -p \
  /home/insol02/IH_ws/so101_IsaacLab/tools/data/inspect_hdf5.py \
  /tmp/generated_with_cameras_smoke.hdf5 \
  --check-cameras \
  --stats \
  --episode 0
```

Expected camera datasets:

```text
data/demo_X/camera_obs/wrist  [T, 480, 640, 3] uint8
data/demo_X/camera_obs/up     [T, 480, 640, 3] uint8
```

---

## 10. Export Isaac HDF5 with cameras to LeRobot

The converter uses a reference LeRobot `meta/info.json` as the schema authority, preserves camera feature names, resamples from Isaac 50 Hz to the reference FPS, and writes videos.

### Standard LeRobot export

```bash
python /home/insol02/IH_ws/so101_IsaacLab/tools/data/convert_isaac_hdf5_to_lerobot_so101.py \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_32env_96trial_with_cameras.hdf5 \
  --out /home/insol02/IH_ws/so101_IsaacLab/datasets/lerobot_mimic_i4h_camera \
  --reference-info /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place/meta/info.json \
  --fallback-source-fps 50 \
  --task "Pick up the cube and place it in the goal region." \
  --only-with-cameras \
  --video-codec libx264 \
  --video-pix-fmt yuv420p \
  --overwrite
```

### VLA Foundry compatibility export

Some VLA Foundry tutorial preprocessors expect `episode_*.parquet` filenames and image/video reference columns inside the parquet. Use:

```bash
python /home/insol02/IH_ws/so101_IsaacLab/tools/data/convert_isaac_hdf5_to_lerobot_so101.py \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/generated_mimic_i4h_32env_96trial_with_cameras.hdf5 \
  --out /home/insol02/IH_ws/so101_IsaacLab/datasets/lerobot_mimic_i4h_vla_compat \
  --reference-info /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place/meta/info.json \
  --fallback-source-fps 50 \
  --task "Pick up the cube and place it in the goal region." \
  --only-with-cameras \
  --video-codec libx264 \
  --video-pix-fmt yuv420p \
  --vla-foundry-compat \
  --overwrite
```

For a 2-episode test:

```bash
python /home/insol02/IH_ws/so101_IsaacLab/tools/data/convert_isaac_hdf5_to_lerobot_so101.py \
  --input_file /home/insol02/IH_ws/so101_IsaacLab/datasets/successful_mimics_with_cameras_added_post_2trials_test.hdf5 \
  --out /home/insol02/IH_ws/so101_IsaacLab/datasets/lerobot_mimic_i4h_2ep_vla_compat \
  --reference-info /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_pickplace_cube_1020_same_place/meta/info.json \
  --fallback-source-fps 50 \
  --task "Pick up the cube and place it in the goal region." \
  --episode-indices 0,1 \
  --only-with-cameras \
  --video-codec libx264 \
  --video-pix-fmt yuv420p \
  --vla-foundry-compat \
  --overwrite
```

Inspect:

```bash
python - <<'PY'
import pandas as pd
p = "/home/insol02/IH_ws/so101_IsaacLab/datasets/lerobot_mimic_i4h_2ep_vla_compat/data/chunk-000/episode_000000.parquet"
df = pd.read_parquet(p)
print(df.columns.tolist())
print(df[["observation.images.wrist", "observation.images.up"]].head())
PY
```

Expected image columns:

```text
observation.images.wrist
observation.images.up
```

---

## 11. VLA Foundry smoke test notes

Inside the VLA Foundry repo:

```bash
cd /home/insol02/IH_ws/vla_foundry

mkdir -p tutorials/data/so101_i4h_mimic_2ep

ln -sfn \
  /home/insol02/IH_ws/so101_IsaacLab/datasets/lerobot_mimic_i4h_2ep_vla_compat \
  tutorials/data/so101_i4h_mimic_2ep/lerobot_vla_compat
```

Preprocessing uses full LeRobot feature names:

```python
SO101_CAMERAS = [
    "observation.images.wrist",
    "observation.images.up",
]
```

The training cell uses short names:

```python
cameras = '["wrist","up"]'
```

For the tutorial model, resize preprocessing images to 224x224 because the Stage-2 VLM/VLA config uses `--data.image_size 224`:

```text
--resize_images_size "[224,224]"
```

A 2-episode run only validates the data path and training loop. It is not a meaningful policy training result.

---

## 12. Common pitfalls

### Negative command-line values

Bad:

```bash
--root-pos -0.02079,-0.01576,-0.03248
```

Good:

```bash
--root-pos="-0.02079,-0.01576,-0.03248"
```

### Tiny/truncated HDF5 files

If you see:

```text
OSError: truncated file: eof = 96, stored_eof = 2048
```

then a script probably opened an output HDF5 and crashed before writing valid content. Use timestamped output files and safe read-only copies.

### Multi-env success checks

If `num_envs=1` succeeds but `num_envs>1` fails, inspect coordinate frames. Success regions are usually env-local, while `root_state_w` is world-frame. Use:

```python
pos_local = obj.data.root_state_w[:, 0:3] - env.scene.env_origins
```

### Cameras missing from HDF5

Camera tensors only appear if the camera-enabled replay task is used and camera observations are explicitly recorded.

### VLA Foundry cannot discover cameras

If VLA Foundry says:

```text
No image columns found in parquet files
```

export with:

```text
--vla-foundry-compat
```

This writes `episode_*.parquet` files and image reference columns such as `observation.images.wrist`.

---

## 13. Git / data hygiene

Do not commit:

```text
datasets/
logs/
outputs/
*.hdf5
*.mp4
*.pt
*.tar
*.parquet
```

Commit:

```text
src/
tools/
README.md
pyproject.toml
LICENSE
CITATION.cff
.gitignore
```

For reproducible experiments, store commands and dataset filenames in README notes or small text files, not the datasets themselves.

---

## 14. Next work

1. Render cameras for all successful generated Mimic episodes.
2. Export the full synthetic dataset to LeRobot and VLA Foundry-compatible formats.
3. Train/evaluate real-only, synthetic-only, and mixed real+synthetic datasets.
4. Add Isaac Lab closed-loop evaluation for the fine-tuned VLA.
5. Compare Pinocchio Mimic generation against Diff IK Mimic after annotation fixes.
6. Add dataset metadata attributes such as `source_fps`, `sim_dt`, `decimation`, and env/task name directly into HDF5 outputs.