#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, os, tarfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from vla_foundry.data.pipelines.robotics import extract_robotics_fields
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml

def read_sample_from_tar(tar_path: Path, sample_index: int = 0) -> dict:
    raw = {}
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        prefixes, last = [], None
        for m in members:
            prefix = m.name.split(".")[0]
            if prefix != last:
                prefixes.append(prefix); last = prefix
        sample_id = prefixes[sample_index]
        for m in members:
            if not m.name.startswith(sample_id + "."): continue
            suffix = m.name[len(sample_id) + 1:]
            f = tar.extractfile(m)
            if f is None: continue
            payload = f.read()
            if suffix.endswith((".jpg", ".jpeg", ".png")):
                raw[suffix] = torch.from_numpy(np.array(Image.open(io.BytesIO(payload)).convert("RGB"))).permute(2,0,1)
            elif suffix.endswith(".npz"):
                raw[suffix] = dict(np.load(io.BytesIO(payload)))
            elif suffix.endswith(".json"):
                raw[suffix] = json.load(io.BytesIO(payload))
    return raw

def main():
    p = argparse.ArgumentParser(description="Offline SO101 VLA vibe check: GT vs predicted action sequence.")
    p.add_argument("--vla-root", default="/home/insol02/IH_ws/vla_foundry")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--preproc-root", required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--tar-index", type=int, default=0)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--out-dir", default="./tutorials/diagnostics/vla_vibe_check")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    os.chdir(Path(args.vla_root).resolve())
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    preproc_root = Path(args.preproc_root).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = sorted((checkpoint_dir / "checkpoints").glob("checkpoint_*.pt"))[-1]
    print("[INFO] checkpoint_dir:", checkpoint_dir)
    print("[INFO] checkpoint:", ckpt)
    model_params = load_params_from_yaml(ModelParams, str(checkpoint_dir / "config_model.yaml"))
    model = create_model(model_params)
    load_model_checkpoint(model, str(ckpt))
    model.eval().cuda()

    processor = RoboticsProcessor.from_pretrained(str(checkpoint_dir))
    data_params = processor.data_params
    normalizer = processor.normalizer
    print("image_names:", list(data_params.image_names))
    print("action_fields:", list(data_params.action_fields))
    print("proprioception_fields:", list(data_params.proprioception_fields))
    print("past:", data_params.lowdim_past_timesteps)
    print("future:", data_params.lowdim_future_timesteps)

    tar_path = sorted((preproc_root / "shards").glob("shard_*.tar"))[args.tar_index]
    raw_sample = read_sample_from_tar(tar_path, args.sample_index)
    metadata = raw_sample.get("metadata.json", {})
    lowdim = raw_sample.get("lowdim.npz", {})
    anchor = metadata.get("original_anchor_relative_idx", metadata.get("anchor_relative_idx", None))
    print("metadata:", metadata)
    print("anchor:", anchor)
    for k in list(data_params.action_fields) + list(data_params.proprioception_fields):
        arr = lowdim.get(k); print(k, None if arr is None else arr.shape)
    if anchor is not None:
        print("required lowdim length:", int(anchor) + int(data_params.lowdim_future_timesteps) + 1)

    fields = extract_robotics_fields(
        raw_sample,
        language_instruction_types=list(data_params.language_instruction_types),
        action_fields=list(data_params.action_fields),
        proprioception_fields=list(data_params.proprioception_fields),
        lowdim_past_timesteps=data_params.lowdim_past_timesteps,
        lowdim_future_timesteps=data_params.lowdim_future_timesteps,
    )
    batch = processor.process_inputs({k: [v] for k,v in fields.items()}, image_names=list(data_params.image_names))
    batch = processor.add_action_and_proprioception_fields(
        batch,
        action_fields=list(data_params.action_fields),
        proprioception_fields=list(data_params.proprioception_fields),
    )
    actions = batch["actions"].cuda()
    past_mask = torch.as_tensor(np.stack(fields["past_mask"][None]), dtype=torch.bool).cuda()
    with torch.no_grad():
        predicted = model.generate_actions(
            input_ids=batch["input_ids"].cuda(),
            pixel_values=batch["pixel_values"].cuda(),
            actions=actions,
            attention_mask=batch["attention_mask"].cuda().bool(),
            attention_mask_images=batch["attention_mask_images"].cuda().bool(),
            past_mask=past_mask,
            proprioception=batch["proprioception"].cuda() if "proprioception" in batch else None,
            num_inference_steps=args.num_inference_steps,
        )

    gt_norm = actions.detach().cpu().numpy()[0]
    pred_norm = predicted.detach().cpu().numpy()[0]
    future_start = int(past_mask.detach().cpu().numpy()[0].sum())
    action_field = list(data_params.action_fields)[0]
    anchor_timestep = int(data_params.lowdim_past_timesteps)
    gt_denorm = normalizer.denormalize_tensor(actions.detach().cpu(), action_field, anchor_timestep=anchor_timestep).numpy()[0]
    pred_denorm = normalizer.denormalize_tensor(predicted.detach().cpu(), action_field, anchor_timestep=anchor_timestep).numpy()[0]
    print("GT denorm shape:", gt_denorm.shape)
    print("Pred denorm shape:", pred_denorm.shape)
    print("future_start:", future_start)
    print("Pred future min:", pred_denorm[future_start:].min(axis=0))
    print("Pred future max:", pred_denorm[future_start:].max(axis=0))

    np.savez(out_dir / f"vibe_sample_{args.sample_index:06d}.npz",
             gt_norm=gt_norm, pred_norm=pred_norm, gt_denorm=gt_denorm, pred_denorm=pred_denorm,
             future_start=future_start, metadata=json.dumps(metadata))

    cam_keys = sorted([k for k in raw_sample if k.endswith("_t0.jpg")])
    T, D = gt_denorm.shape; t = np.arange(T); num_dims = min(D, 6)
    fig, axes = plt.subplots(num_dims + 1, 2, figsize=(14, 3 * (num_dims + 1)), sharex=False)
    for i in range(2):
        ax = axes[0, i]
        if i < len(cam_keys):
            img = raw_sample[cam_keys[i]].permute(1,2,0).numpy()
            ax.imshow(img); ax.set_title(cam_keys[i].replace("observation.images.", "").replace("_t0.jpg", ""))
        ax.axis("off")
    for dim in range(num_dims):
        ax = axes[dim+1,0]
        ax.plot(t, gt_norm[:,dim], "-", label="GT norm"); ax.plot(t, pred_norm[:,dim], "--", label="Pred norm")
        ax.axvline(future_start-1, ls=":", alpha=0.6); ax.set_title(f"Action dim {dim} normalized"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        ax = axes[dim+1,1]
        ax.plot(t, gt_denorm[:,dim], "-", label="GT denorm"); ax.plot(t, pred_denorm[:,dim], "--", label="Pred denorm")
        ax.axvline(future_start-1, ls=":", alpha=0.6); ax.set_title(f"Action dim {dim} denormalized"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.suptitle("SO101 VLA Vibe Check: GT vs Predicted Actions", fontsize=14)
    plt.tight_layout()
    fig_path = out_dir / f"vibe_sample_{args.sample_index:06d}.png"
    plt.savefig(fig_path, dpi=150); print("[DONE] wrote", fig_path)
    if args.show: plt.show()
    else: plt.close(fig)

if __name__ == "__main__":
    main()
