# SO101 VLA Foundry Diagnostics and Training Scripts

Recommended diagnostic order:

1. **GT replay in Isaac Lab**: prove dataset actions still succeed in the camera eval env.
2. **Preprocess with the target horizon**: `future_lowdim_steps` must match VLA training.
3. **Train VLA**: use the matching manifest/stats for the same horizon.
4. **Offline vibe check**: GT vs predicted action sequence for one sample.
5. **Offline anchor sweep**: GT vs predicted at many anchors / phases.
6. **Live Isaac Lab eval**: only after the offline checks are sane.

Key meanings:
- `future_lowdim_steps` / `--data.lowdim_future_timesteps`: trained action horizon.
- `past_lowdim_steps` / `--data.lowdim_past_timesteps`: low-dimensional history context.
- `num_inference_steps`: diffusion denoising iterations.
- `replan_steps`: how many generated actions to execute before replanning in live sim.
- `reset_steps`: settling steps after env reset; not horizon and not chunk size.

Example h60 preprocessing:

```bash
cd /home/insol02/IH_ws/vla_foundry
uv run python /path/to/03_preprocess_so101.py   --compat-root /home/insol02/IH_ws/so101_IsaacLab/datasets/so101_i4h_lerobot_mimic_vla_compat   --output-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_i4h_mimic_h60/preprocessed   --past-lowdim-steps 2   --future-lowdim-steps 60   --resize 224 224   --num-workers 1
```

Example h60 VLA overfit training:

```bash
cd /home/insol02/IH_ws/vla_foundry
uv run python /path/to/04_train_vla.py   --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_i4h_mimic_h60/preprocessed   --future-lowdim-timesteps 60   --past-lowdim-timesteps 2   --per-gpu-batch-size 16   --global-batch-size 16   --total-train-samples 300000   --num-checkpoints 10   --max-checkpoint-limit 10
```

Example vibe check:

```bash
cd /home/insol02/IH_ws/vla_foundry
uv run python /path/to/05_vla_vibe_check.py   --checkpoint-dir /home/insol02/IH_ws/vla_foundry/tutorials/checkpoints/<diffusion_run_dir>   --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_i4h_mimic_h60/preprocessed   --sample-index 0   --num-inference-steps 10   --out-dir /home/insol02/IH_ws/vla_foundry/tutorials/diagnostics/h60_vibe
```

Example anchor sweep:

```bash
cd /home/insol02/IH_ws/vla_foundry
uv run python /path/to/07_vla_anchor_sweep.py   --checkpoint-dir /home/insol02/IH_ws/vla_foundry/tutorials/checkpoints/<diffusion_run_dir>   --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_i4h_mimic_h60/preprocessed   --episode-index 27   --max-samples 12   --num-inference-steps 10   --out-dir /home/insol02/IH_ws/vla_foundry/tutorials/diagnostics/h60_anchor_sweep
```
