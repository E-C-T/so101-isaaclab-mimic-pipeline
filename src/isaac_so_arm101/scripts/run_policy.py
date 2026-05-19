from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run a local policy inside Isaac Lab.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--policy", type=str, default="zero", choices=["zero", "random", "lerobot", "pi0"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Policy: {args.policy}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print("[TODO] Build env, construct policy wrapper, and run local inference loop.")


if __name__ == "__main__":
    main()