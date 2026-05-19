from __future__ import annotations

import argparse
import math
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Cube-Joint-Pos-Mimic-v0")
parser.add_argument("--dataset_file", type=str, default="")
parser.add_argument("--episode", type=str, default="demo_0")
parser.add_argument("--reset_to_demo_initial_state", action="store_true")
parser.add_argument("--step_actions", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from pxr import UsdGeom, Gf


def quat_wxyz_to_rotmat_np(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def make_T_np(pos, quat_wxyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rotmat_np(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def rot_error_deg(R_a, R_b):
    R = R_a.T @ R_b
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def pos_rot_err(T_a, T_b):
    pos_err = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    ang_err = rot_error_deg(T_a[:3, :3], T_b[:3, :3])
    return pos_err, ang_err


def gf_matrix_to_np(M):
    arr = np.eye(4, dtype=np.float64)
    # Gf.Matrix4d is row-major indexable.
    for r in range(4):
        for c in range(4):
            arr[r, c] = float(M[r][c])
    return arr


def get_usd_world_T(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    M = cache.GetLocalToWorldTransform(prim)
    return gf_matrix_to_np(M)


def local_offset_T(offset_xyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(offset_xyz, dtype=np.float64)
    return T


def find_paths(stage, contains):
    out = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if contains in p:
            out.append(p)
    return out


def print_T(label, T):
    print(f"\n[{label}]")
    print("pos =", np.array2string(T[:3, 3], precision=7, suppress_small=False))
    print("rot =")
    print(np.array2string(T[:3, :3], precision=7, suppress_small=False))


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    if args_cli.reset_to_demo_initial_state:
        if not args_cli.dataset_file:
            raise ValueError("--dataset_file is required with --reset_to_demo_initial_state")
        handler = HDF5DatasetFileHandler()
        handler.open(args_cli.dataset_file)
        ep = handler.load_episode(args_cli.episode, env.device)
        initial_state = ep.get_initial_state()
        env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
        env.sim.render()
        print("[INFO] reset_to demo initial_state")
    else:
        print("[INFO] env.reset() default initial_state")

    if args_cli.step_actions > 0:
        if not args_cli.dataset_file:
            raise ValueError("--dataset_file is required with --step_actions")
        with h5py.File(args_cli.dataset_file, "r") as f:
            actions = f[f"data/{args_cli.episode}/actions"][()]
        for t in range(min(args_cli.step_actions, actions.shape[0])):
            a = torch.as_tensor(actions[t], device=env.device, dtype=torch.float32).unsqueeze(0)
            env.step(a)
        env.sim.render()
        print(f"[INFO] stepped original actions for {args_cli.step_actions} steps")

    stage = env.sim.stage

    print("\n[PATH SEARCH: gripper]")
    for p in find_paths(stage, "gripper"):
        print(" ", p)

    print("\n[PATH SEARCH: jaw]")
    for p in find_paths(stage, "jaw"):
        print(" ", p)

    robot = env.scene["robot"]
    q = robot.data.joint_pos[0, :6].detach().cpu().numpy()
    root = robot.data.root_state_w[0].detach().cpu().numpy()

    root_pos = root[0:3]
    root_quat = root[3:7]

    T_world_base_tensor = make_T_np(root_pos, root_quat)

    print("\n[ROBOT STATE]")
    print("root_pos  =", root_pos)
    print("root_quat =", root_quat, "  # wxyz")
    print("joint_pos =", q)

    T_world_eef_tensor = env.get_robot_eef_pose("end_effector")[0].detach().cpu().numpy()
    print_T("Isaac env.get_robot_eef_pose('end_effector')", T_world_eef_tensor)

    # Candidate USD paths. Adjust only if your printed path search shows different paths.
    candidates = [
        "/World/envs/env_0/Robot/base_link",
        "/World/envs/env_0/Robot/gripper_link",
        "/World/envs/env_0/Robot/gripper_link/gripper_frame_link",
        "/World/envs/env_0/Robot/moving_jaw_so101_v1_link",
    ]

    print("\n[USD CANDIDATE FRAMES]")
    T_usd = {}
    for p in candidates:
        T = get_usd_world_T(stage, p)
        if T is None:
            print(f"{p}: INVALID")
            continue
        T_usd[p] = T
        print_T(f"USD {p}", T)

    offset = np.array([0.01, 0.0, -0.09], dtype=np.float64)
    T_link_eef = local_offset_T(offset)

    gripper_link_path = "/World/envs/env_0/Robot/gripper_link"
    if gripper_link_path in T_usd:
        T_world_eef_from_usd_offset = T_usd[gripper_link_path] @ T_link_eef
        print_T("USD gripper_link * offset [0.01, 0, -0.09]", T_world_eef_from_usd_offset)

        pe, ae = pos_rot_err(T_world_eef_tensor, T_world_eef_from_usd_offset)
        print("\n[COMPARE] env EEF vs USD gripper_link+offset")
        print(f"pos_err = {pe:.9f} m")
        print(f"rot_err = {ae:.9f} deg")

    # Pinocchio comparison through your existing IK helper.
    from isaac_so_arm101.tasks.cube_mimic.pinocchio_ik import So101PinocchioIK

    ik = So101PinocchioIK()
    T_base_eef_pin = ik.forward_eef_pose(q)
    T_world_eef_pin = T_world_base_tensor @ T_base_eef_pin

    print_T("Pinocchio base->EEF from q", T_base_eef_pin)
    print_T("World base * Pinocchio base->EEF", T_world_eef_pin)

    pe, ae = pos_rot_err(T_world_eef_tensor, T_world_eef_pin)
    print("\n[COMPARE] env EEF vs world_base * Pinocchio FK")
    print(f"pos_err = {pe:.9f} m")
    print(f"rot_err = {ae:.9f} deg")

    # Compare USD gripper_frame_link to env EEF, if valid.
    gf_path = "/World/envs/env_0/Robot/gripper_link/gripper_frame_link"
    if gf_path in T_usd:
        pe, ae = pos_rot_err(T_world_eef_tensor, T_usd[gf_path])
        print("\n[COMPARE] env EEF vs USD gripper_frame_link")
        print(f"pos_err = {pe:.9f} m")
        print(f"rot_err = {ae:.9f} deg")

    print("\n[INTERPRETATION]")
    print("If env EEF ≈ USD gripper_link+offset and env EEF ≈ world_base*Pinocchio FK, frame math is correct.")
    print("If gripper_frame_link differs, do not use it as the Mimic/IK frame.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
