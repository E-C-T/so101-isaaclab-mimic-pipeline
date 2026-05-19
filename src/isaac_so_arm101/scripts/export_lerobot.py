from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Export Isaac Lab trajectories to LeRobot format.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    print(f"[INFO] Exporting from {args.input_dir} to {args.output_dir}")
    print("[TODO] Hook this to src.isaac_so_arm101.data.export.to_lerobot")


if __name__ == "__main__":
    main()