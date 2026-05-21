#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay GT HDF5 actions in Isaac Lab and log object/EEF state.")
parser.add_argument("--task", default="Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0")
parser.add_argument("--dataset-file", required=True)
parser.add_argument("--episode-index", type=int, default=0)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--end-index", type=int, default=None)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--debug-every", type=int, default=10)
parser.add_argument("--save-camera-debug", action="store_true")
parser.add_argument("--camera-debug-dir", default="/home/insol02/IH_ws/so101_IsaacLab/datasets/gt_replay_camera_debug")
parser.add_argument("--preserve-cfg-root-pose", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py, gymnasium as gym, imageio.v3 as iio, numpy as np, torch
import isaac_so_arm101.tasks  # noqa
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def load_episode_arrays(path, ep):
    with h5py.File(path, "r") as f:
        demo=f["data"][f"demo_{ep}"]
        return np.asarray(demo["obs"]["joint_pos"][:], dtype=np.float32), np.asarray(demo["actions"][:], dtype=np.float32)

def load_episode_data_object(dataset_file, episode_index, device):
    h=HDF5DatasetFileHandler(); h.open(dataset_file)
    names=list(h.get_episode_names()); data=h.load_episode(names[episode_index], device); h.close(); return data

def inject_default_rigid_object_state(initial_state, env, env_id):
    if "rigid_object" in initial_state: return initial_state
    runtime_state=env.scene.get_state(is_relative=True)
    if "rigid_object" not in runtime_state: return initial_state
    initial_state["rigid_object"]={}
    for asset_name, asset_state_dict in runtime_state["rigid_object"].items():
        initial_state["rigid_object"][asset_name]={}
        for state_name, state_tensor in asset_state_dict.items():
            initial_state["rigid_object"][asset_name][state_name]=state_tensor[env_id:env_id+1].clone().detach()
    return initial_state

def get_cfg_robot_root_pose_tensor(env): return env.scene["robot"].data.root_state_w[0:1,:7].clone().detach()

def apply_robot_root_pose(env, root_pose_w):
    robot=env.scene["robot"]; root_pose_w=root_pose_w.to(device=env.device, dtype=torch.float32)
    robot.write_root_pose_to_sim(root_pose_w); robot.write_root_velocity_to_sim(torch.zeros((root_pose_w.shape[0],6),device=env.device))
    robot.reset(); env.sim.render()

def reset_env_to_episode_start(env, episode_data, root_pose_w=None):
    s=inject_default_rigid_object_state(episode_data.get_initial_state(), env, 0)
    if root_pose_w is not None and "articulation" in s and "robot" in s["articulation"]:
        s["articulation"]["robot"]["root_pose"]=root_pose_w.clone().detach()
        s["articulation"]["robot"]["root_velocity"]=torch.zeros((1,6),device=env.device)
    env.reset_to(s, torch.tensor([0], device=env.device), is_relative=True)
    if root_pose_w is not None: apply_robot_root_pose(env, root_pose_w)
    env.sim.render()

def get_object_pos_local(env):
    if "object" not in env.scene.keys(): return None
    p=env.scene["object"].data.root_state_w[0,:3].detach()
    if hasattr(env.scene,"env_origins"): p=p-env.scene.env_origins[0]
    return p.cpu().numpy()

def get_eef_pos_local(env):
    if "ee_frame" not in env.scene.keys(): return None
    ee=env.scene["ee_frame"]
    if not hasattr(ee.data,"target_pos_w"): return None
    p=ee.data.target_pos_w[0,0,:].detach()
    if hasattr(env.scene,"env_origins"): p=p-env.scene.env_origins[0]
    return p.cpu().numpy()

def get_goal_region(env):
    r=getattr(getattr(env,"cfg",None),"goal_region",None)
    return r if isinstance(r,dict) else {"x_min":0.025,"x_max":0.175,"y_min":0.125,"y_max":0.275,"z_min":0.0,"z_max":0.04}

def is_object_in_goal(env):
    p=get_object_pos_local(env); g=get_goal_region(env)
    return False if p is None else (g["x_min"]<=p[0]<=g["x_max"] and g["y_min"]<=p[1]<=g["y_max"] and g["z_min"]<=p[2]<=g["z_max"])

def camera_rgb(env, name):
    rgb=env.scene[name].data.output["rgb"]
    if isinstance(rgb, torch.Tensor): rgb=rgb.detach().cpu().numpy()
    if rgb.ndim==4: rgb=rgb[0]
    if rgb.shape[-1]==4: rgb=rgb[...,:3]
    if rgb.dtype!=np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and rgb.max()<=1.5: rgb=rgb*255.0
        rgb=np.clip(rgb,0,255).astype(np.uint8)
    return rgb

def save_cameras(env, out_dir, step):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); env.sim.render()
    for key, short in [("up_camera","up"),("wrist_camera","wrist")]:
        if key in env.scene.keys(): iio.imwrite(out_dir/f"{short}_{step:06d}.png", camera_rgb(env,key))

def main():
    cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.recorders={}; cfg.terminations={}; cfg.seed=0
    env=gym.make(args_cli.task, cfg=cfg).unwrapped; env.reset()
    root_pose=get_cfg_robot_root_pose_tensor(env) if args_cli.preserve_cfg_root_pose else None
    _, actions=load_episode_arrays(args_cli.dataset_file, args_cli.episode_index)
    episode_data=load_episode_data_object(args_cli.dataset_file, args_cli.episode_index, env.device)
    reset_env_to_episode_start(env, episode_data, root_pose)
    end=args_cli.end_index if args_cli.end_index is not None else len(actions)-1
    end=min(end, len(actions)-1); ever=False
    for t in range(args_cli.start_index, end+1):
        a=torch.zeros(env.action_space.shape, device=env.device)
        a[0]=torch.tensor(actions[t], dtype=torch.float32, device=env.device)
        env.step(a)
        if root_pose is not None: apply_robot_root_pose(env, root_pose)
        success=is_object_in_goal(env); ever=ever or success
        if args_cli.save_camera_debug and (t==args_cli.start_index or t==end or t%args_cli.debug_every==0):
            save_cameras(env, args_cli.camera_debug_dir, t)
        if t%args_cli.debug_every==0 or success or t==end:
            obj=get_object_pos_local(env); ee=get_eef_pos_local(env)
            dist=None if obj is None or ee is None else float(np.linalg.norm(ee-obj))
            print(f"[GT STEP {t:04d}] success={success} ever_success={ever} obj={obj} ee={ee} dist={dist} action={actions[t]}")
        if success:
            print(f"[DONE] Success at step {t}"); break
    print("[RESULT] ever_success:", ever)
    print("[RESULT] final_object_pos:", get_object_pos_local(env))
    env.close()

if __name__=="__main__":
    try: main()
    finally: simulation_app.close()
