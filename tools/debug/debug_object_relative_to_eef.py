from __future__ import annotations

import argparse
import h5py
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Cube-Joint-Pos-Mimic-v0")
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--episode", type=str, default="demo_0")
parser.add_argument("--start", type=int, default=250)
parser.add_argument("--end", type=int, default=460)
parser.add_argument("--stride", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler


def invert_T(T: torch.Tensor) -> torch.Tensor:
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ p)
    return T_inv


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.dataset_file)
    ep = handler.load_episode(args_cli.episode, env.device)
    env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
    env.sim.render()

    with h5py.File(args_cli.dataset_file, "r") as f:
        actions = f[f"data/{args_cli.episode}/actions"][()]

    samples = []

    for t in range(min(args_cli.end + 1, actions.shape[0])):
        action = torch.as_tensor(actions[t], device=env.device, dtype=torch.float32).unsqueeze(0)
        env.step(action)

        if t >= args_cli.start and t % args_cli.stride == 0:
            T_w_eef = env.get_robot_eef_pose("end_effector")[0]
            p_obj_w = env.scene["object"].data.root_state_w[0, 0:3]

            T_eef_w = invert_T(T_w_eef)
            p_obj_h = torch.cat([p_obj_w, torch.ones(1, device=env.device)])
            p_obj_eef = (T_eef_w @ p_obj_h)[:3]

            signals = env.get_subtask_term_signals()
            signal_dict = {k: bool(v[0].detach().cpu()) for k, v in signals.items()}

            samples.append(p_obj_eef.detach().cpu().numpy())

            print(
                "[OBJ REL EEF]",
                "t=", t,
                "obj_w=", p_obj_w.detach().cpu().numpy(),
                "eef_w=", T_w_eef[:3, 3].detach().cpu().numpy(),
                "obj_minus_eef_world=", (p_obj_w - T_w_eef[:3, 3]).detach().cpu().numpy(),
                "obj_in_eef_frame=", p_obj_eef.detach().cpu().numpy(),
                "signals=", signal_dict,
            )

    if samples:
        samples = np.stack(samples, axis=0)
        print("\n[SUMMARY]")
        print("mean object position in current EEF frame:", samples.mean(axis=0))
        print("std  object position in current EEF frame:", samples.std(axis=0))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
