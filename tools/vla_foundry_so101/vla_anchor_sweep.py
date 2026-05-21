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

def iter_samples(preproc_root):
    for tar_path in sorted((preproc_root/"shards").glob("shard_*.tar")):
        with tarfile.open(tar_path,"r") as tar:
            members=tar.getmembers(); prefixes=[]; last=None
            for m in members:
                pref=m.name.split(".")[0]
                if pref!=last: prefixes.append(pref); last=pref
            for sid in prefixes:
                raw={}
                for m in members:
                    if not m.name.startswith(sid+"."): continue
                    suffix=m.name[len(sid)+1:]; f=tar.extractfile(m)
                    if f is None: continue
                    payload=f.read()
                    if suffix.endswith((".jpg",".jpeg",".png")):
                        raw[suffix]=torch.from_numpy(np.array(Image.open(io.BytesIO(payload)).convert("RGB"))).permute(2,0,1)
                    elif suffix.endswith(".npz"): raw[suffix]=dict(np.load(io.BytesIO(payload)))
                    elif suffix.endswith(".json"): raw[suffix]=json.load(io.BytesIO(payload))
                yield tar_path, sid, raw

def infer_one(model, processor, data_params, normalizer, raw, steps):
    fields=extract_robotics_fields(raw,
        language_instruction_types=list(data_params.language_instruction_types),
        action_fields=list(data_params.action_fields),
        proprioception_fields=list(data_params.proprioception_fields),
        lowdim_past_timesteps=data_params.lowdim_past_timesteps,
        lowdim_future_timesteps=data_params.lowdim_future_timesteps)
    batch=processor.process_inputs({k:[v] for k,v in fields.items()}, image_names=list(data_params.image_names))
    batch=processor.add_action_and_proprioception_fields(batch, action_fields=list(data_params.action_fields), proprioception_fields=list(data_params.proprioception_fields))
    actions=batch["actions"].cuda(); past_mask=torch.as_tensor(np.stack(fields["past_mask"][None]), dtype=torch.bool).cuda()
    with torch.no_grad():
        pred=model.generate_actions(input_ids=batch["input_ids"].cuda(), pixel_values=batch["pixel_values"].cuda(), actions=actions,
            attention_mask=batch["attention_mask"].cuda().bool(), attention_mask_images=batch["attention_mask_images"].cuda().bool(),
            past_mask=past_mask, proprioception=batch["proprioception"].cuda() if "proprioception" in batch else None, num_inference_steps=steps)
    field=list(data_params.action_fields)[0]; anchor=int(data_params.lowdim_past_timesteps)
    gt=normalizer.denormalize_tensor(actions.detach().cpu(), field, anchor_timestep=anchor).numpy()[0]
    pr=normalizer.denormalize_tensor(pred.detach().cpu(), field, anchor_timestep=anchor).numpy()[0]
    return gt, pr, int(past_mask.detach().cpu().numpy()[0].sum())

def main():
    p=argparse.ArgumentParser(description="Offline anchor sweep for SO101 VLA.")
    p.add_argument("--vla-root", default="/home/insol02/IH_ws/vla_foundry")
    p.add_argument("--checkpoint-dir", required=True); p.add_argument("--preproc-root", required=True)
    p.add_argument("--episode-index", type=int, default=None); p.add_argument("--max-samples", type=int, default=12)
    p.add_argument("--num-inference-steps", type=int, default=10); p.add_argument("--out-dir", default="./tutorials/diagnostics/vla_anchor_sweep")
    args=p.parse_args(); os.chdir(Path(args.vla_root).resolve())
    ckpt_dir=Path(args.checkpoint_dir).resolve(); preproc=Path(args.preproc_root).resolve(); out=Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    ckpt=sorted((ckpt_dir/"checkpoints").glob("checkpoint_*.pt"))[-1]
    model_params=load_params_from_yaml(ModelParams, str(ckpt_dir/"config_model.yaml")); model=create_model(model_params); load_model_checkpoint(model, str(ckpt)); model.eval().cuda()
    processor=RoboticsProcessor.from_pretrained(str(ckpt_dir)); data_params=processor.data_params; normalizer=processor.normalizer
    rows=[]; count=0
    for tar_path,sid,raw in iter_samples(preproc):
        meta=raw.get("metadata.json",{}); ep=int(meta.get("episode_index",-1))
        if args.episode_index is not None and ep!=args.episode_index: continue
        gt, pred, fs=infer_one(model, processor, data_params, normalizer, raw, args.num_inference_steps)
        mse=((gt[fs:]-pred[fs:])**2).mean(axis=0); mae=np.abs(gt[fs:]-pred[fs:]).mean(axis=0)
        frame=int(meta.get("frame_index",-1))
        row={"sample_id":sid,"tar":str(tar_path),"episode_index":ep,"frame_index":frame,"timestamp":float(meta.get("timestamp",-1.0)),
             "mse":mse.tolist(),"mae":mae.tolist(),"mse_mean":float(mse.mean()),"mae_mean":float(mae.mean())}
        rows.append(row)
        T,D=gt.shape; t=np.arange(T); nd=min(D,6); fig,axs=plt.subplots(nd,1,figsize=(10,2.4*nd),sharex=True)
        if nd==1: axs=[axs]
        for d in range(nd):
            axs[d].plot(t,gt[:,d],"-",label="GT"); axs[d].plot(t,pred[:,d],"--",label="Pred"); axs[d].axvline(fs-1,ls=":",alpha=.6)
            axs[d].set_title(f"ep={ep} frame={frame} dim={d} mae={mae[d]:.4g}"); axs[d].grid(True,alpha=.3); axs[d].legend(fontsize=8)
        plt.tight_layout(); fig.savefig(out/f"anchor_ep{ep:06d}_frame{frame:06d}.png", dpi=130); plt.close(fig)
        count+=1
        if count>=args.max_samples: break
    if not rows: raise RuntimeError("No samples matched selection.")
    with (out/"anchor_sweep_summary.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print("[DONE] wrote", out/"anchor_sweep_summary.jsonl")
    for r in rows: print(f"ep={r['episode_index']} frame={r['frame_index']} mse_mean={r['mse_mean']:.6g} mae_mean={r['mae_mean']:.6g}")

if __name__=="__main__": main()
