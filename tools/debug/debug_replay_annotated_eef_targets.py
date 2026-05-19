from __future__ import annotations

import argparse
import h5py
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Cube-Joint-Pos-Mimic-v0")
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--episode", type=str, default="demo_0")
parser.add_argument("--max_steps", type=int, default=120)
parser.add_argument("--reset_to_demo_initial_state", action="store_true")

# Debug/tuning options.
parser.add_argument(
    "--use_source_posture_prior",
    action="store_true",
    help="Pass source demo action[t] as q_nominal to target_eef_pose_to_action().",
)
parser.add_argument("--q_compare_start", type=int, default=280)
parser.add_argument("--q_compare_end", type=int, default=430)
parser.add_argument("--print_every", type=int, default=10)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler


def find_dataset(group, suffix):
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith(suffix):
            found.append(name)

    group.visititems(visit)
    if not found:
        raise KeyError(f"Could not find dataset ending with: {suffix}")
    return found[0]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    if args_cli.reset_to_demo_initial_state:
        handler = HDF5DatasetFileHandler()
        handler.open(args_cli.dataset_file)
        ep = handler.load_episode(args_cli.episode, env.device)
        initial_state = ep.get_initial_state()
        env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
        env.sim.render()
        print("[DEBUG] reset_to demo initial_state")
    else:
        print("[DEBUG] used env.reset() default initial_state")

    with h5py.File(args_cli.dataset_file, "r") as f:
        demo = f[f"data/{args_cli.episode}"]

        print("[DEBUG] available datasets ending with useful names:")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and (
                "target_eef_pose" in name
                or "eef_pose" in name
                or name.endswith("actions")
                or "gripper" in name
            ):
                print("   ", name, obj.shape)

        demo.visititems(visit)

        target_path = find_dataset(demo, "obs/datagen_info/target_eef_pose/end_effector")
        target_poses = demo[target_path][()]
        print("[DEBUG] target_path =", target_path)
        print("[DEBUG] target_poses shape =", target_poses.shape)

        if "actions" not in demo:
            raise KeyError("Could not find demo/actions. This script needs full source actions for q_nominal.")
        actions = demo["actions"][()]
        print("[DEBUG] actions shape =", actions.shape)

        try:
            gripper_path = find_dataset(demo, "obs/datagen_info/gripper_action/end_effector")
            gripper_actions = demo[gripper_path][()]
            print("[DEBUG] gripper_path =", gripper_path)
            print("[DEBUG] gripper_actions shape =", gripper_actions.shape)
        except KeyError:
            print("[DEBUG] no datagen_info/gripper_action found; falling back to actions[:, -1]")
            gripper_actions = actions[:, -1:]

    n = min(args_cli.max_steps, target_poses.shape[0], actions.shape[0])
    print("[DEBUG] n steps =", n)
    print("[DEBUG] use_source_posture_prior =", args_cli.use_source_posture_prior)

    for t in range(n):
        T = torch.as_tensor(target_poses[t], device=env.device, dtype=torch.float32)

        g = torch.as_tensor(gripper_actions[t], device=env.device, dtype=torch.float32)
        if g.ndim == 0:
            g = g.reshape(1)

        q_source_t = torch.as_tensor(actions[t], device=env.device, dtype=torch.float32)

        current = env.get_robot_eef_pose("end_effector")[0]
        cur_pos = current[:3, 3]
        target_pos = T[:3, 3]
        before_dist = torch.linalg.norm(target_pos - cur_pos)

        if args_cli.use_source_posture_prior:
            action = env.target_eef_pose_to_action(
                target_eef_pose=T,
                gripper_action={"end_effector": g},
                env_id=0,
                q_nominal=q_source_t,
            )
        else:
            action = env.target_eef_pose_to_action(
                target_eef_pose=T,
                gripper_action={"end_effector": g},
                env_id=0,
            )

        env.step(action.unsqueeze(0))

        if t % args_cli.print_every == 0:
            obj = env.scene["object"].data.root_state_w[0, 0:3]
            new_pos = env.get_robot_eef_pose("end_effector")[0, :3, 3]
            after_dist = torch.linalg.norm(target_pos - new_pos)

            signals = {}
            if hasattr(env, "get_subtask_term_signals"):
                try:
                    raw_signals = env.get_subtask_term_signals()
                    signals = {k: bool(v[0].detach().cpu()) for k, v in raw_signals.items()}
                except Exception as exc:
                    signals = {"signal_error": str(exc)}

            print(
                "[TARGET REPLAY]",
                "t=", t,
                "target_pos=", target_pos.detach().cpu().numpy(),
                "before_dist=", float(before_dist.detach().cpu()),
                "after_dist=", float(after_dist.detach().cpu()),
                "eef_pos=", new_pos.detach().cpu().numpy(),
                "object_pos=", obj.detach().cpu().numpy(),
                "signals=", signals,
                "action=", action.detach().cpu().numpy(),
            )

        if args_cli.q_compare_start <= t <= args_cli.q_compare_end and t % args_cli.print_every == 0:
            q_current = env.scene["robot"].data.joint_pos[0, :6]
            q_ik = action

            print(
                "[Q COMPARE]",
                "t=", t,
                "q_source=", q_source_t.detach().cpu().numpy(),
                "q_current=", q_current.detach().cpu().numpy(),
                "q_ik=", q_ik.detach().cpu().numpy(),
                "q_ik_minus_source=", (q_ik - q_source_t).detach().cpu().numpy(),
            )

    final_obj = env.scene["object"].data.root_state_w[0, 0:3]
    final_eef = env.get_robot_eef_pose("end_effector")[0, :3, 3]
    print("[FINAL] object_pos=", final_obj.detach().cpu().numpy())
    print("[FINAL] eef_pos=", final_eef.detach().cpu().numpy())

    if hasattr(env, "get_subtask_term_signals"):
        try:
            raw_signals = env.get_subtask_term_signals()
            print("[FINAL] subtask_signals=", {k: bool(v[0].detach().cpu()) for k, v in raw_signals.items()})
        except Exception as exc:
            print("[FINAL] subtask signal error:", exc)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
