from pathlib import Path
import argparse, json, h5py, numpy as np, pandas as pd, imageio.v3 as iio

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--task", default="Pick up the cube and place it in the goal region.")
    ap.add_argument("--dummy_h", type=int, default=224)
    ap.add_argument("--dummy_w", type=int, default=224)
    args = ap.parse_args()

    out = Path(args.out)
    meta = out / "meta"
    data_dir = out / "data" / "chunk-000"
    video_dir = out / "videos" / "chunk-000" / "observation.images.front"
    meta.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    total_frames = 0

    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for ep_idx, demo_name in enumerate(demos):
            g = f["data"][demo_name]
            q = np.asarray(g["obs/joint_pos"], dtype=np.float32)
            qd = np.asarray(g["obs/joint_vel"], dtype=np.float32)
            obj = np.asarray(g["obs/object_position"], dtype=np.float32)
            action = np.asarray(g["processed_actions"] if "processed_actions" in g else g["actions"], dtype=np.float32)

            T = min(len(q), len(qd), len(obj), len(action))
            state = np.concatenate([q[:T], qd[:T], obj[:T]], axis=1).astype(np.float32)

            df = pd.DataFrame({
                "observation.state": list(state),
                "action": list(action[:T]),
                "timestamp": np.arange(T, dtype=np.float64) / args.fps,
                "episode_index": np.full(T, ep_idx, dtype=np.int64),
                "task_index": np.zeros(T, dtype=np.int64),
                "index": np.arange(total_frames, total_frames + T, dtype=np.int64),
                "frame_index": np.arange(T, dtype=np.int64),
                "next.reward": np.r_[np.zeros(T-1), 1.0].astype(np.float32),
                "next.done": np.r_[np.zeros(T-1, dtype=bool), True],
            })
            df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet")

            frames = np.zeros((T, args.dummy_h, args.dummy_w, 3), dtype=np.uint8)
            iio.imwrite(video_dir / f"episode_{ep_idx:06d}.mp4", frames, fps=args.fps)

            episodes.append({"episode_index": ep_idx, "tasks": [args.task], "length": T})
            total_frames += T

    write_jsonl(meta / "episodes.jsonl", episodes)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": args.task}])

    info = {
        "codebase_version": "so101_dummy_lerobot_v0",
        "robot_type": "so101",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [15]},
            "action": {"dtype": "float32", "shape": [6]},
            "observation.images.front": {"dtype": "video", "shape": [args.dummy_h, args.dummy_w, 3]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))

    modality = {
        "state": {"observation.state": {"start": 0, "end": 15}},
        "action": {"action": {"start": 0, "end": 6}},
        "video": {"observation.images.front": {}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2))
    print(f"Wrote LeRobot-style dataset to: {out}")

if __name__ == "__main__":
    main()
