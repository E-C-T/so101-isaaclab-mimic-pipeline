#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, os, subprocess
from pathlib import Path

def latest_checkpoint(exp_glob: str) -> str:
    dirs = sorted(glob.glob(exp_glob))
    if not dirs:
        raise FileNotFoundError(f"No dirs matched {exp_glob}")
    ckpts = sorted(glob.glob(f"{dirs[-1]}/checkpoints/checkpoint_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_*.pt under {dirs[-1]}/checkpoints")
    return ckpts[-1]

def main():
    p = argparse.ArgumentParser(description="Train VLA diffusion policy on SO101 robotics shards.")
    p.add_argument("--vla-root", default="/home/insol02/IH_ws/vla_foundry")
    p.add_argument("--preproc-root", required=True)
    p.add_argument("--vlm-ckpt", default=None)
    p.add_argument("--past-lowdim-timesteps", type=int, default=2)
    p.add_argument("--future-lowdim-timesteps", type=int, default=60)
    p.add_argument("--per-gpu-batch-size", type=int, default=16)
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--total-train-samples", type=int, default=300000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-checkpoints", type=int, default=10)
    p.add_argument("--max-checkpoint-limit", type=int, default=10)
    p.add_argument("--save-path", default="./tutorials/checkpoints")
    p.add_argument("--master-port", default="0")
    args = p.parse_args()

    os.chdir(Path(args.vla_root).resolve())
    preproc_root = Path(args.preproc_root).resolve()
    manifest = preproc_root / "shards/manifest.jsonl"
    stats = preproc_root / "shards/stats.json"
    if not manifest.exists(): raise FileNotFoundError(manifest)
    if not stats.exists(): raise FileNotFoundError(stats)
    vlm_ckpt = args.vlm_ckpt or latest_checkpoint("tutorials/checkpoints/*vlm*")

    cmd = [
        "uv", "run", "torchrun",
        "--nproc_per_node=1", "--nnodes=1", f"--master_port={args.master_port}",
        "vla_foundry/main.py",
        "--model", "include vla_foundry/config_presets/models/vla_diffusion_100m.yaml",
        "--model.vision_language_backbone.resume_from_checkpoint", vlm_ckpt,
        "--distributed.fsdp", "False",
        "--data.type", "robotics",
        "--data.processor", "simple_vlm",
        "--data.image_size", "224",
        "--data.img_num_tokens", "256",
        "--data.seq_len", "2048",
        "--data.dataset_manifest", f'["{manifest}"]',
        "--data.dataset_statistics", f'["{stats}"]',
        "--data.dataset_modality", '["robotics"]',
        "--data.dataset_weighting", "[1.0]",
        "--data.camera_names", '["wrist","up"]',
        "--data.action_fields", '["action"]',
        "--data.proprioception_fields", '["observation.state"]',
        "--data.language_instruction_types", '["original"]',
        "--data.pose_groups", "[]",
        "--data.intrinsics_fields", "[]",
        "--data.extrinsics_fields", "[]",
        "--data.lowdim_past_timesteps", str(args.past_lowdim_timesteps),
        "--data.lowdim_future_timesteps", str(args.future_lowdim_timesteps),
        "--data.allow_multiple_epochs", "True",
        "--data.num_workers", str(args.num_workers),
        "--hparams", "include vla_foundry/config_presets/hparams/diffusion_policy.yaml",
        "--hparams.per_gpu_batch_size", str(args.per_gpu_batch_size),
        "--hparams.global_batch_size", str(args.global_batch_size),
        "--hparams.warmup", str(args.warmup),
        "--total_train_samples", str(args.total_train_samples),
        "--num_checkpoints", str(args.num_checkpoints),
        "--max_checkpoint_limit", str(args.max_checkpoint_limit),
        "--save_path", args.save_path,
        "--wandb", "False",
        "--db_logging", "False",
    ]
    print("Running SO101 VLA training:\n" + " ".join(cmd))
    print(f"[INFO] optimizer steps ≈ {args.total_train_samples / args.global_batch_size:.1f}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
