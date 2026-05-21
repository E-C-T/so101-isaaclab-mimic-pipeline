#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, shutil, subprocess
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Preprocess SO101 LeRobot data into VLA Foundry robotics tar shards.")
    p.add_argument("--vla-root", default="/home/insol02/IH_ws/vla_foundry")
    p.add_argument("--compat-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--past-lowdim-steps", type=int, default=2)
    p.add_argument("--future-lowdim-steps", type=int, default=60)
    p.add_argument("--resize", nargs=2, type=int, default=[224, 224], metavar=("W", "H"))
    p.add_argument("--samples-per-shard", type=int, default=32)
    p.add_argument("--max-episodes-to-process", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.chdir(Path(args.vla_root).resolve())
    compat_root = Path(args.compat_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not compat_root.exists():
        raise FileNotFoundError(f"Missing compat root: {compat_root}")
    if output_root.exists():
        if args.overwrite:
            print(f"[WARN] Removing existing output root: {output_root}")
            shutil.rmtree(output_root)
        else:
            raise RuntimeError(f"{output_root} already exists. Use --overwrite only if intentional.")

    cameras = ["observation.images.wrist", "observation.images.up"]
    cmd = [
        "uv", "run", "--group", "preprocessing", "python",
        "vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py",
        "--type", "lerobot",
        "--source_episodes", f"['{compat_root}']",
        "--output_dir", str(output_root),
        "--camera_names", str(cameras),
        "--observation_keys", "['observation.state']",
        "--action_keys", "['action']",
        "--past_lowdim_steps", str(args.past_lowdim_steps),
        "--future_lowdim_steps", str(args.future_lowdim_steps),
        "--resize_images_size", f"[{args.resize[0]},{args.resize[1]}]",
        "--samples_per_shard", str(args.samples_per_shard),
        "--config_path", "tutorials/tutorial_utils/robotics_preprocessing_params_tutorial.yaml",
        "--num_workers", str(args.num_workers),
        "--db_logging", "False",
    ]
    if args.max_episodes_to_process is not None:
        cmd += ["--max_episodes_to_process", str(args.max_episodes_to_process)]
    print("Running preprocessing:\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[DONE] Manifest:", output_root / "shards/manifest.jsonl")
    print("[DONE] Stats:   ", output_root / "shards/stats.json")

if __name__ == "__main__":
    main()
