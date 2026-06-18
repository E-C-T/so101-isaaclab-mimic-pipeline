#!/usr/bin/env python3
"""
Nearest-GT diagnostic for VLA Foundry preprocessed SO101 tar shards.

Purpose:
  Given a live joint/action vector q, find the closest observation.state windows
  in a VLA Foundry preprocessed dataset and print the GT action continuation.

This helps answer:
  "From the state my live policy reached, what did the demonstrations do next?"

Example:
  python nearest_gt_compare_preprocessed.py \
    --preproc-root /home/insol02/IH_ws/vla_foundry/tutorials/data/so101_i4h_mimic_2ep_27_65_h60/preprocessed \
    --live-q="-0.08047047,-1.21297,1.299576,0.8238884,-1.7594925,0.06221569" \
    --top-k 8 \
    --lookahead 20
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Match:
    dist: float
    shard: str
    sample_id: str
    metadata: dict[str, Any]
    timestep: int
    state: np.ndarray
    action: np.ndarray
    next_actions: np.ndarray


def parse_vec(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(vals, dtype=np.float32)


def unique_prefixes(members: list[tarfile.TarInfo]) -> list[str]:
    prefixes: list[str] = []
    last = None
    for m in members:
        prefix = m.name.split(".")[0]
        if prefix != last:
            prefixes.append(prefix)
            last = prefix
    return prefixes


def read_one_sample(tar: tarfile.TarFile, members: list[tarfile.TarInfo], sample_id: str) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    prefix = sample_id + "."
    for m in members:
        if not m.name.startswith(prefix):
            continue
        suffix = m.name[len(prefix):]
        f = tar.extractfile(m)
        if f is None:
            continue
        payload = f.read()
        if suffix.endswith(".npz"):
            raw[suffix] = dict(np.load(io.BytesIO(payload)))
        elif suffix.endswith(".json"):
            raw[suffix] = json.load(io.BytesIO(payload))
    return raw


def iter_samples(preproc_root: Path):
    shard_paths = sorted((preproc_root / "shards").glob("shard_*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.tar files found under {preproc_root / 'shards'}")

    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            for sample_id in unique_prefixes(members):
                raw = read_one_sample(tar, members, sample_id)
                lowdim = raw.get("lowdim.npz", {})
                metadata = raw.get("metadata.json", {})
                if "observation.state" not in lowdim or "action" not in lowdim:
                    continue
                yield shard_path.name, sample_id, metadata, lowdim["observation.state"], lowdim["action"]


def find_nearest(
    preproc_root: Path,
    live_q: np.ndarray,
    top_k: int,
    lookahead: int,
    compare_dims: int | None,
) -> list[Match]:
    matches: list[Match] = []

    for shard_name, sample_id, metadata, states, actions in iter_samples(preproc_root):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if states.ndim != 2 or actions.ndim != 2:
            continue

        d = live_q.shape[0] if compare_dims is None else min(compare_dims, live_q.shape[0], states.shape[1])
        q = live_q[:d]
        s = states[:, :d]

        dist = np.linalg.norm(s - q[None, :], axis=1)
        # Find local best few from this sample, then global top-k.
        candidate_ts = np.argsort(dist)[: min(top_k, len(dist))]

        for t in candidate_ts:
            t_int = int(t)
            end = min(actions.shape[0], t_int + lookahead)
            matches.append(
                Match(
                    dist=float(dist[t_int]),
                    shard=shard_name,
                    sample_id=sample_id,
                    metadata=metadata,
                    timestep=t_int,
                    state=states[t_int],
                    action=actions[t_int],
                    next_actions=actions[t_int:end],
                )
            )

    matches.sort(key=lambda m: m.dist)
    return matches[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc-root", required=True, type=Path)
    ap.add_argument("--live-q", required=True, help="Comma-separated 6D live joint/action vector.")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lookahead", type=int, default=20)
    ap.add_argument("--compare-dims", type=int, default=None)
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    live_q = parse_vec(args.live_q)
    print("[INFO] live_q:", live_q)
    print("[INFO] preproc_root:", args.preproc_root)

    matches = find_nearest(
        preproc_root=args.preproc_root,
        live_q=live_q,
        top_k=args.top_k,
        lookahead=args.lookahead,
        compare_dims=args.compare_dims,
    )

    serializable = []
    for rank, m in enumerate(matches):
        meta_short = {
            k: m.metadata.get(k)
            for k in ["episode_index", "frame_index", "timestamp", "index", "anchor_relative_idx"]
            if k in m.metadata
        }
        print("\n" + "=" * 100)
        print(f"[MATCH {rank}] dist={m.dist:.6f} shard={m.shard} sample_id={m.sample_id} t={m.timestep}")
        print("[META]", meta_short)
        print("[GT state[t]] ", np.array2string(m.state, precision=6, floatmode="fixed"))
        print("[GT action[t]]", np.array2string(m.action, precision=6, floatmode="fixed"))
        print(f"[GT next {len(m.next_actions)} actions]")
        for i, a in enumerate(m.next_actions):
            print(f"  +{i:02d}: {np.array2string(a, precision=6, floatmode='fixed')}")

        serializable.append(
            {
                "rank": rank,
                "dist": m.dist,
                "shard": m.shard,
                "sample_id": m.sample_id,
                "timestep": m.timestep,
                "metadata": m.metadata,
                "state": m.state.tolist(),
                "action": m.action.tolist(),
                "next_actions": m.next_actions.tolist(),
            }
        )

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(serializable, indent=2))
        print("\n[INFO] saved:", args.save_json)


if __name__ == "__main__":
    main()
