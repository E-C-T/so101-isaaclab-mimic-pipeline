from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.5:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        default="/home/insol02/IH_ws/so101_IsaacLab/datasets/debug_generated_camera_obs",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        demo = f["data"][args.demo]

        for key in ["camera_obs/up", "camera_obs/wrist"]:
            if key not in demo:
                print(f"MISSING: data/{args.demo}/{key}")
                continue

            arr = demo[key]
            idx = min(args.frame, arr.shape[0] - 1)

            print(f"FOUND: data/{args.demo}/{key} shape={arr.shape} dtype={arr.dtype}")

            img = to_uint8(np.asarray(arr[idx]))

            # Drop alpha if present.
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]

            cam_name = key.split("/")[-1]
            out_path = out_dir / f"{args.demo}_{cam_name}_{idx:06d}.png"
            iio.imwrite(out_path, img)
            print(f"WROTE: {out_path}")


if __name__ == "__main__":
    main()