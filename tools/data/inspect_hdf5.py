#!/usr/bin/env python3
"""
Inspect Isaac/Isaac Mimic HDF5 datasets.

Examples:

python tools/data/inspect_hdf5.py dataset.hdf5
python tools/data/inspect_hdf5.py dataset.hdf5 --episode 5
python tools/data/inspect_hdf5.py dataset.hdf5 --list-demos
python tools/data/inspect_hdf5.py dataset.hdf5 --check-cameras
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def sorted_demo_keys(h5_file: h5py.File) -> list[str]:
    if "data" not in h5_file:
        return []
    keys = list(h5_file["data"].keys())

    def sort_key(name: str):
        try:
            return (int(name.split("_")[-1]), name)
        except Exception:
            return (10**12, name)

    return sorted(keys, key=sort_key)


def print_attrs(obj: h5py.Group | h5py.Dataset, indent: str = "  ") -> None:
    if not obj.attrs:
        return
    print(f"{indent}attrs:")
    for k, v in obj.attrs.items():
        print(f"{indent}  {k}: {v}")


def print_tree(name: str, obj: Any, *, show_attrs: bool = False, filter_text: str | None = None) -> None:
    if filter_text is not None and filter_text not in name:
        return
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name}")
        print(f"  shape = {obj.shape}")
        print(f"  dtype = {obj.dtype}")
        if show_attrs:
            print_attrs(obj)
    elif isinstance(obj, h5py.Group):
        print(f"[GROUP]   {name}")
        if show_attrs:
            print_attrs(obj)


def dataset_exists(group: h5py.Group, path: str) -> bool:
    try:
        return isinstance(group[path], h5py.Dataset)
    except KeyError:
        return False


def summarize_demo(demo: h5py.Group) -> None:
    print("\n--- quick summary ---")
    common_paths = [
        "actions",
        "processed_actions",
        "obs/actions",
        "obs/joint_pos",
        "obs/joint_vel",
        "obs/object_position",
        "obs/target_object_position",
        "states/articulation/robot/joint_position",
        "states/rigid_object/object/root_pose",
        "camera_obs/wrist",
        "camera_obs/up",
    ]
    for path in common_paths:
        if dataset_exists(demo, path):
            ds = demo[path]
            print(f"{path:<52} shape={ds.shape} dtype={ds.dtype}")

    if "camera_obs" in demo and isinstance(demo["camera_obs"], h5py.Group):
        print("\n--- camera_obs ---")
        for key, obj in demo["camera_obs"].items():
            if isinstance(obj, h5py.Dataset):
                print(f"camera_obs/{key:<20} shape={obj.shape} dtype={obj.dtype}")


def check_cameras(h5: h5py.File, demos: list[str]) -> None:
    print("\n--- camera check ---")
    missing = []
    present = []
    for demo_name in demos:
        demo = h5["data"][demo_name]
        if "camera_obs" not in demo:
            missing.append(demo_name)
            continue
        cams = []
        for key, obj in demo["camera_obs"].items():
            if isinstance(obj, h5py.Dataset):
                cams.append((key, obj.shape, str(obj.dtype)))
        present.append((demo_name, cams))

    print(f"demos with camera_obs    : {len(present)}")
    print(f"demos missing camera_obs : {len(missing)}")
    if present:
        demo_name, cams = present[0]
        print(f"\nfirst camera demo: {demo_name}")
        for key, shape, dtype in cams:
            print(f"  {key}: shape={shape}, dtype={dtype}")
    if missing:
        print("\nfirst missing demos:")
        for name in missing[:10]:
            print(f"  {name}")


def episode_length_stats(h5: h5py.File, demos: list[str]) -> None:
    lengths = []
    for demo_name in demos:
        demo = h5["data"][demo_name]
        if "actions" in demo:
            lengths.append(len(demo["actions"]))
        elif "processed_actions" in demo:
            lengths.append(len(demo["processed_actions"]))
    if not lengths:
        return
    arr = np.asarray(lengths)
    print("\n--- episode length stats ---")
    print(f"count: {len(arr)}")
    print(f"min  : {arr.min()}")
    print(f"max  : {arr.max()}")
    print(f"mean : {arr.mean():.2f}")
    print(f"std  : {arr.std():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Isaac/Isaac Mimic HDF5 dataset contents.")
    parser.add_argument("path", type=str, help="Path to HDF5 dataset.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to inspect. Default: 0.")
    parser.add_argument("--demo", type=str, default=None, help="Demo key to inspect directly, e.g. demo_12.")
    parser.add_argument("--list-demos", action="store_true", help="List demo names and exit.")
    parser.add_argument("--check-cameras", action="store_true", help="Check which demos have camera_obs.")
    parser.add_argument("--stats", action="store_true", help="Print episode length stats.")
    parser.add_argument("--attrs", action="store_true", help="Print HDF5 attributes.")
    parser.add_argument("--filter", type=str, default=None, help="Only print tree entries containing this substring.")
    parser.add_argument("--no-tree", action="store_true", help="Do not print full selected demo tree.")
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    with h5py.File(path, "r") as f:
        print("=" * 80)
        print("HDF5 INSPECTION")
        print("=" * 80)
        print("file:", path)
        if args.attrs:
            print_attrs(f, indent="")

        demos = sorted_demo_keys(f)
        if not demos:
            print("No /data group or no demos found.")
            print("=" * 80)
            return

        print("num demos:", len(demos))
        if args.list_demos:
            print("\n--- demos ---")
            for i, name in enumerate(demos):
                print(f"{i:04d}: {name}")
            print("=" * 80)
            return

        if args.stats:
            episode_length_stats(f, demos)
        if args.check_cameras:
            check_cameras(f, demos)

        if args.demo is not None:
            demo_name = args.demo
            if demo_name not in f["data"]:
                raise KeyError(f"Demo {demo_name!r} not found in /data")
        else:
            if args.episode < 0 or args.episode >= len(demos):
                raise IndexError(f"--episode {args.episode} out of range [0, {len(demos)-1}]")
            demo_name = demos[args.episode]

        demo = f["data"][demo_name]
        print(f"\n--- selected demo: {demo_name} ---")
        summarize_demo(demo)
        if not args.no_tree:
            print(f"\n--- {demo_name} tree ---")
            demo.visititems(lambda name, obj: print_tree(name, obj, show_attrs=args.attrs, filter_text=args.filter))
        print("=" * 80)


if __name__ == "__main__":
    main()