# SO101 VLA diagnostics scripts

## Install

```bash
cd /home/insol02/IH_ws/so101_IsaacLab
mkdir -p tools/vla_foundry_so101
cp nearest_gt_compare_preprocessed.py tools/vla_foundry_so101/
cp run_live_vla_tests.sh tools/vla_foundry_so101/
cp train_vla_and_shutdown.sh tools/vla_foundry_so101/
chmod +x tools/vla_foundry_so101/*.py tools/vla_foundry_so101/*.sh
```

## Nearest GT diagnostic

Use this with a live action/q printed by the evaluator:

```bash
cd /home/insol02/IH_ws/so101_IsaacLab

python tools/vla_foundry_so101/nearest_gt_compare_preprocessed.py \
  --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_i4h_mimic_2ep_27_65_h60/preprocessed \
  --live-q="-0.08047047,-1.21297,1.299576,0.8238884,-1.7594925,0.06221569" \
  --top-k 8 \
  --lookahead 20 \
  --save-json outputs/nearest_gt_debug.json
```

Interpretation:
- If nearest GT continues into contact/lift but VLA retreats, the policy is phase/contact wrong.
- If nearest GT also retreats, the live state is close to a demonstration retreat/recovery phase.
- If the nearest distance is large, the live rollout has left the training state manifold.

## Live tests

```bash
conda activate env_so101_vla_isaaclab
bash /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/run_live_vla_tests.sh
```

## Long training and shutdown

Default is no shutdown:

```bash
bash /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/train_vla_and_shutdown.sh
```

Shutdown on success:

```bash
SHUTDOWN_ON_SUCCESS=1 bash /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/train_vla_and_shutdown.sh
```

Train a different preprocessed root:

```bash
PREPROC_ROOT=/path/to/preprocessed \
TOTAL_TRAIN_SAMPLES=2000000 \
GLOBAL_BATCH_SIZE=16 \
PER_GPU_BATCH_SIZE=16 \
SHUTDOWN_ON_SUCCESS=1 \
bash /home/insol02/IH_ws/so101_IsaacLab/tools/vla_foundry_so101/train_vla_and_shutdown.sh
```
