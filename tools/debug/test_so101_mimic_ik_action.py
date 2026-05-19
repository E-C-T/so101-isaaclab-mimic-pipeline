from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-Cube-Joint-Pos-Mimic-v0",
)

# Do NOT add --device manually. AppLauncher adds it.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    T0 = env.get_robot_eef_pose("end_effector")[0]
    print("[TEST] initial EEF pose:")
    print(T0.detach().cpu().numpy())

    T_target = T0.clone()
    T_target[0, 3] += 0.02
    T_target[1, 3] += 0.02

    action = env.target_eef_pose_to_action(
        target_eef_pose=T_target,
        eef_name="end_effector",
        env_id=0,
    )

    print("[TEST] IK action:")
    print(action.detach().cpu().numpy())

    for _ in range(80):
        env.step(action.unsqueeze(0))

    T1 = env.get_robot_eef_pose("end_effector")[0]
    print("[TEST] final EEF pose:")
    print(T1.detach().cpu().numpy())

    print("[TEST] initial pos:", T0[:3, 3].detach().cpu().numpy())
    print("[TEST] target  pos:", T_target[:3, 3].detach().cpu().numpy())
    print("[TEST] final   pos:", T1[:3, 3].detach().cpu().numpy())

    initial_dist = torch.linalg.norm(T_target[:3, 3] - T0[:3, 3])
    final_dist = torch.linalg.norm(T_target[:3, 3] - T1[:3, 3])
    print("[TEST] initial target distance:", float(initial_dist.detach().cpu()))
    print("[TEST] final target distance:  ", float(final_dist.detach().cpu()))

    if final_dist < initial_dist:
        print("[TEST RESULT] PASS: EEF moved closer to target.")
    else:
        print("[TEST RESULT] FAIL: EEF did not move closer to target.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
