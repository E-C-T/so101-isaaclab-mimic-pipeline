from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.managers import SceneEntityCfg

from isaac_so_arm101.tasks.cube_mimic_i4h.pinocchio_mimic_env import (
    SoArm101CubePinocchioMimicI4HEnv,
)
import isaac_so_arm101.tasks.cube_mimic_i4h.mimic_mdp as mimic_mdp


class SoArm101CubeDiffIKMimicI4HEnv(SoArm101CubePinocchioMimicI4HEnv):
    """SO101 I4H Mimic env using USD Diff IK for Mimic generation.

    Action convention:
        action[0:3] = absolute target EEF position for DiffIK
        action[3]   = gripper joint target

    This intentionally bypasses Pinocchio IK.
    """

    def __init__(self, cfg=None, render_mode=None, **kwargs):
        """Create env from either a cfg object or an env_cfg_entry_point string."""
        if cfg is None:
            env_cfg_entry_point = kwargs.pop("env_cfg_entry_point", None)
            if env_cfg_entry_point is None:
                raise ValueError("Missing cfg. Expected cfg=... or env_cfg_entry_point=...")

            module_name, class_name = env_cfg_entry_point.split(":")
            module = __import__(module_name, fromlist=[class_name])
            cfg = getattr(module, class_name)()

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _goal_region(self) -> dict:
        return getattr(
            self.cfg,
            "goal_region",
            {
                "x_min": 0.075,
                "x_max": 0.225,
                "y_min": 0.175,
                "y_max": 0.325,
                "z_min": 0.0,
                "z_max": 0.15,
                "max_lin_vel": 0.15,
            },
        )

    def _above_goal_region(self) -> dict:
        region = dict(self._goal_region())
        region.update({"z_min": 0.04, "z_max": 0.14, "max_lin_vel": 1.0})
        return region


    def _resolve_env_ids(self, env_ids=None):
        """Convert Isaac Mimic env_ids argument into tensor/slice indexing.

        Isaac Mimic may pass env_ids as:
            None
            int
            list[int]
            tuple[int]
            torch.Tensor
            slice

        This helper normalizes those formats for indexing Isaac Lab tensors.
        """
        if env_ids is None:
            return slice(None)

        if isinstance(env_ids, slice):
            return env_ids

        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)

        if isinstance(env_ids, int):
            return torch.tensor([env_ids], device=self.device, dtype=torch.long)

        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def get_subtask_term_signals(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return subtask-completion signals for Isaac Mimic.

        This supports both the base Mimic style with no env_ids and call-sites
        that pass a subset of env ids.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals_all = {
            "object_lifted": mimic_mdp.object_lifted(
                self,
                asset_cfg=SceneEntityCfg("object"),
                min_height=0.04,
            ),
            "object_above_goal": mimic_mdp.object_above_goal(
                self,
                asset_name="object",
                **self._above_goal_region(),
            ),
            "object_in_goal": mimic_mdp.object_in_goal(
                self,
                asset_name="object",
                **self._goal_region(),
            ),
        }
        signals = {name: value[env_ids] for name, value in signals_all.items()}

        if getattr(self.cfg, "debug_mimic_signals", False) and self.common_step_counter % 50 == 0:
            obj = self.scene["object"]
            root_state = obj.data.root_state_w[0]
            print(
                "[I4H DIFFIK DIFFIK DEBUG]",
                "step=", self.common_step_counter,
                "object_pos=", root_state[0:3].detach().cpu().numpy(),
                "object_vel=", root_state[7:10].detach().cpu().numpy(),
                "lifted=", bool(signals_all["object_lifted"][0].detach().cpu()),
                "above_goal=", bool(signals_all["object_above_goal"][0].detach().cpu()),
                "in_goal=", bool(signals_all["object_in_goal"][0].detach().cpu()),
            )

        return signals

    def _coerce_target_pose_tensor(self, target_eef_pose, eef_name: str, **kwargs) -> torch.Tensor:
        """Accept dict/tensor/list target EEF pose inputs and return [N, 4, 4]."""
        if target_eef_pose is None:
            target_eef_pose = kwargs.get("target_eef_pose", None)
        if target_eef_pose is None:
            target_eef_pose = kwargs.get("target_eef_pose_dict", None)
        if target_eef_pose is None:
            target_eef_pose = kwargs.get("eef_pose", None)
        if target_eef_pose is None:
            raise ValueError("Missing target_eef_pose / target_eef_pose_dict / eef_pose.")

        if isinstance(target_eef_pose, dict):
            target_eef_pose = target_eef_pose.get(eef_name, next(iter(target_eef_pose.values())))

        if isinstance(target_eef_pose, torch.Tensor):
            target = target_eef_pose.to(device=self.device, dtype=torch.float32)
        else:
            target = torch.as_tensor(target_eef_pose, device=self.device, dtype=torch.float32)

        if target.ndim == 2:
            target = target.unsqueeze(0)
        if target.shape[-2:] != (4, 4):
            raise ValueError(f"Expected EEF pose as [..., 4, 4], got {tuple(target.shape)}")
        return target

    def _coerce_gripper_tensor(
        self,
        gripper_action,
        eef_name: str,
        batch_size: int,
        **kwargs,
    ) -> torch.Tensor:
        """Return gripper command as [batch_size, 1]."""
        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action", None)
        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action_dict", None)

        if gripper_action is None:
            robot = self.scene["robot"]
            joint_names = list(robot.data.joint_names)
            try:
                gripper_idx = joint_names.index("gripper")
                values = robot.data.joint_pos[:, gripper_idx : gripper_idx + 1]
            except ValueError:
                values = robot.data.joint_pos[:, -1:]
        else:
            if isinstance(gripper_action, dict):
                gripper_action = gripper_action.get(eef_name, next(iter(gripper_action.values())))

            if isinstance(gripper_action, torch.Tensor):
                values = gripper_action.to(device=self.device, dtype=torch.float32)
            else:
                values = torch.as_tensor(gripper_action, device=self.device, dtype=torch.float32)

            if values.ndim == 0:
                values = values.reshape(1, 1)
            elif values.ndim == 1:
                values = values.reshape(-1, 1)
            elif values.ndim > 2:
                values = values.reshape(values.shape[0], -1)[:, :1]

        if values.shape[0] == 1 and batch_size > 1:
            values = values.repeat(batch_size, 1)
        elif values.shape[0] != batch_size:
            values = values[:batch_size]

        return values
    
    def _world_eef_pose_to_root_position_action(
        self,
        target_pose_w: torch.Tensor,
        env_id=None,
        env_ids=None,
    ) -> torch.Tensor:
        """Convert world-frame EEF target pose matrix to robot-root-frame xyz.

        target_pose_w:
            Tensor [N, 4, 4], world-frame EEF pose.

        Returns:
            Tensor [N, 3], target EEF position expressed in robot root frame.

        Important:
            We only transform position here. Do not pass the EEF orientation into
            subtract_frame_transforms(), because target_pose_w stores orientation
            as a 3x3 rotation matrix while subtract_frame_transforms expects
            quaternions.
        """

        robot = self.scene["robot"]
        batch_size = target_pose_w.shape[0]

        if env_id is not None:
            selected_env_ids = torch.tensor([int(env_id)], device=self.device, dtype=torch.long)
        elif env_ids is not None and not isinstance(env_ids, slice):
            selected_env_ids = env_ids
            if not isinstance(selected_env_ids, torch.Tensor):
                selected_env_ids = torch.as_tensor(selected_env_ids, device=self.device, dtype=torch.long)
            else:
                selected_env_ids = selected_env_ids.to(device=self.device, dtype=torch.long)
        else:
            selected_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if selected_env_ids.numel() != batch_size:
            if batch_size == 1:
                selected_env_ids = selected_env_ids[:1]
            elif selected_env_ids.numel() == 1:
                selected_env_ids = selected_env_ids.repeat(batch_size)
            else:
                selected_env_ids = selected_env_ids[:batch_size]

        root_pos_w = robot.data.root_pos_w[selected_env_ids]
        root_quat_w = robot.data.root_quat_w[selected_env_ids]

        # Homogeneous matrix -> world position only.
        target_pos_w = target_pose_w[:, 0:3, 3]

        # Position-only world -> root transform.
        # Do NOT pass target orientation here.
        target_pos_b, _ = PoseUtils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            target_pos_w,
        )

        if getattr(self.cfg, "debug_diffik_frame_conversion", False):
            if not hasattr(self, "_diffik_frame_debug_count"):
                self._diffik_frame_debug_count = 0
            if self._diffik_frame_debug_count < 20:
                print(
                    "[DIFFIK FRAME CONVERSION]",
                    "count=", self._diffik_frame_debug_count,
                    "root_pos_w=", root_pos_w[0].detach().cpu().numpy(),
                    "target_pos_w=", target_pos_w[0].detach().cpu().numpy(),
                    "target_pos_b=", target_pos_b[0].detach().cpu().numpy(),
                )
                self._diffik_frame_debug_count += 1

        return target_pos_b

    def _quat_from_matrix_wxyz(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices [N, 3, 3] to quaternions [N, 4] in wxyz order."""

        q = torch.empty((R.shape[0], 4), device=R.device, dtype=R.dtype)
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cond = trace > 0.0

        if cond.any():
            s = torch.sqrt(torch.clamp(trace[cond] + 1.0, min=1e-8)) * 2.0
            q[cond, 0] = 0.25 * s
            q[cond, 1] = (R[cond, 2, 1] - R[cond, 1, 2]) / s
            q[cond, 2] = (R[cond, 0, 2] - R[cond, 2, 0]) / s
            q[cond, 3] = (R[cond, 1, 0] - R[cond, 0, 1]) / s

        not_cond = ~cond
        if not_cond.any():
            idx = torch.nonzero(not_cond, as_tuple=False).squeeze(-1)
            Rn = R[idx]
            qn = torch.empty((Rn.shape[0], 4), device=R.device, dtype=R.dtype)

            c0 = (Rn[:, 0, 0] > Rn[:, 1, 1]) & (Rn[:, 0, 0] > Rn[:, 2, 2])
            c1 = (~c0) & (Rn[:, 1, 1] > Rn[:, 2, 2])
            c2 = (~c0) & (~c1)

            if c0.any():
                s0 = torch.sqrt(torch.clamp(1.0 + Rn[c0, 0, 0] - Rn[c0, 1, 1] - Rn[c0, 2, 2], min=1e-8)) * 2.0
                qn[c0, 0] = (Rn[c0, 2, 1] - Rn[c0, 1, 2]) / s0
                qn[c0, 1] = 0.25 * s0
                qn[c0, 2] = (Rn[c0, 0, 1] + Rn[c0, 1, 0]) / s0
                qn[c0, 3] = (Rn[c0, 0, 2] + Rn[c0, 2, 0]) / s0

            if c1.any():
                s1 = torch.sqrt(torch.clamp(1.0 + Rn[c1, 1, 1] - Rn[c1, 0, 0] - Rn[c1, 2, 2], min=1e-8)) * 2.0
                qn[c1, 0] = (Rn[c1, 0, 2] - Rn[c1, 2, 0]) / s1
                qn[c1, 1] = (Rn[c1, 0, 1] + Rn[c1, 1, 0]) / s1
                qn[c1, 2] = 0.25 * s1
                qn[c1, 3] = (Rn[c1, 1, 2] + Rn[c1, 2, 1]) / s1

            if c2.any():
                s2 = torch.sqrt(torch.clamp(1.0 + Rn[c2, 2, 2] - Rn[c2, 0, 0] - Rn[c2, 1, 1], min=1e-8)) * 2.0
                qn[c2, 0] = (Rn[c2, 1, 0] - Rn[c2, 0, 1]) / s2
                qn[c2, 1] = (Rn[c2, 0, 2] + Rn[c2, 2, 0]) / s2
                qn[c2, 2] = (Rn[c2, 1, 2] + Rn[c2, 2, 1]) / s2
                qn[c2, 3] = 0.25 * s2

            q[idx] = qn

        q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)
        # Stabilize quaternion sign. q and -q represent the same orientation.
        q = q * torch.where(q[:, 0:1] < 0.0, -1.0, 1.0)
        return q

    def _world_eef_pose_to_root_pose_action(
        self,
        target_pose_w: torch.Tensor,
        env_id=None,
        env_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world-frame EEF target pose matrix to root-frame position + quaternion.

        Returns:
            target_pos_b: [N, 3]
            target_quat_b: [N, 4] in wxyz order
        """

        robot = self.scene["robot"]
        batch_size = target_pose_w.shape[0]

        if env_id is not None:
            selected_env_ids = torch.tensor([int(env_id)], device=self.device, dtype=torch.long)
        elif env_ids is not None and not isinstance(env_ids, slice):
            selected_env_ids = env_ids
            if not isinstance(selected_env_ids, torch.Tensor):
                selected_env_ids = torch.as_tensor(selected_env_ids, device=self.device, dtype=torch.long)
            else:
                selected_env_ids = selected_env_ids.to(device=self.device, dtype=torch.long)
        else:
            selected_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if selected_env_ids.numel() != batch_size:
            if batch_size == 1:
                selected_env_ids = selected_env_ids[:1]
            elif selected_env_ids.numel() == 1:
                selected_env_ids = selected_env_ids.repeat(batch_size)
            else:
                selected_env_ids = selected_env_ids[:batch_size]

        root_pos_w = robot.data.root_pos_w[selected_env_ids]
        root_quat_w = robot.data.root_quat_w[selected_env_ids]

        target_pos_w = target_pose_w[:, 0:3, 3]
        R_w_eef = target_pose_w[:, 0:3, 0:3]

        target_pos_b, _ = PoseUtils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            target_pos_w,
        )

        R_w_root = PoseUtils.matrix_from_quat(root_quat_w)
        R_root_w = R_w_root.transpose(1, 2)
        R_root_eef = R_root_w @ R_w_eef
        target_quat_b = self._quat_from_matrix_wxyz(R_root_eef)

        if getattr(self.cfg, "debug_diffik_frame_conversion", False):
            if not hasattr(self, "_diffik_pose_debug_count"):
                self._diffik_pose_debug_count = 0
            if self._diffik_pose_debug_count < 20:
                print(
                    "[DIFFIK POSE CONVERSION]",
                    "count=", self._diffik_pose_debug_count,
                    "target_pos_w=", target_pos_w[0].detach().cpu().numpy(),
                    "target_pos_b=", target_pos_b[0].detach().cpu().numpy(),
                    "target_quat_b=", target_quat_b[0].detach().cpu().numpy(),
                )
                self._diffik_pose_debug_count += 1

        return target_pos_b, target_quat_b


    def target_eef_pose_to_action(
        self,
        target_eef_pose=None,
        eef_name: str = "end_effector",
        gripper_action=None,
        action_noise_dict=None,
        env_id=None,
        env_ids=None,
        q_nominal=None,
        **kwargs,
    ) -> torch.Tensor:
        """Convert Mimic target EEF pose to DiffIK pose + gripper action.

        Output action layout:
            [target_eef_x_b, target_eef_y_b, target_eef_z_b, qw_b, qx_b, qy_b, qz_b, gripper]

        where the pose is expressed in the robot root frame.

        This intentionally bypasses Pinocchio. Isaac Lab DiffIK solves the arm
        motion using the USD articulation Jacobian.
        """

        if target_eef_pose is None:
            target_eef_pose = kwargs.get("target_eef_pose", None)
        if target_eef_pose is None:
            target_eef_pose = kwargs.get("target_eef_pose_dict", None)
        if target_eef_pose is None:
            target_eef_pose = kwargs.get("eef_pose", None)

        if isinstance(target_eef_pose, dict):
            target_eef_pose = target_eef_pose.get(
                eef_name,
                next(iter(target_eef_pose.values())),
            )

        if target_eef_pose is None:
            raise ValueError(
                "target_eef_pose_to_action requires target_eef_pose, "
                "target_eef_pose_dict, or eef_pose."
            )

        target = target_eef_pose
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        else:
            target = target.to(device=self.device, dtype=torch.float32)

        # Accept [4, 4], [1, 4, 4], or [N, 4, 4].
        if target.ndim == 2:
            target = target.unsqueeze(0)

        if target.shape[-2:] != (4, 4):
            raise ValueError(
                f"Expected target EEF pose as 4x4 matrix, got shape {tuple(target.shape)}"
            )

        # Resolve selected env ids for batch alignment.
        if env_ids is not None:
            resolved_env_ids = self._resolve_env_ids(env_ids)
            if isinstance(resolved_env_ids, slice):
                num_selected = self.num_envs
            else:
                num_selected = len(resolved_env_ids)
        elif env_id is not None:
            resolved_env_ids = torch.tensor([int(env_id)], device=self.device, dtype=torch.long)
            num_selected = 1
        else:
            resolved_env_ids = slice(None)
            num_selected = self.num_envs

        if target.shape[0] == 1 and num_selected > 1:
            target = target.repeat(num_selected, 1, 1)

        # Mimic target EEF pose is world-frame.
        # DiffIK absolute pose action expects robot-root-frame xyz + quaternion.
        target_pos, target_quat = self._world_eef_pose_to_root_pose_action(
            target_pose_w=target,
            env_id=env_id,
            env_ids=env_ids,
        )

        if action_noise_dict is not None:
            noise = action_noise_dict.get(eef_name, None) if isinstance(action_noise_dict, dict) else action_noise_dict
            if noise is not None:
                if not isinstance(noise, torch.Tensor):
                    noise = torch.as_tensor(noise, device=self.device, dtype=torch.float32)
                else:
                    noise = noise.to(device=self.device, dtype=torch.float32)
                target_pos = target_pos + noise.reshape(-1)[0:3] * torch.randn_like(target_pos)

        # Extract gripper action. If missing, preserve current gripper.
        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action", None)
        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action_dict", None)

        robot = self.scene["robot"]

        if gripper_action is None:
            current_q = robot.data.joint_pos
            gripper_values = current_q[:, 5:6]
            if not isinstance(resolved_env_ids, slice):
                gripper_values = gripper_values[resolved_env_ids]
            if gripper_values.shape[0] == 1 and target_pos.shape[0] > 1:
                gripper_values = gripper_values.repeat(target_pos.shape[0], 1)
        else:
            if isinstance(gripper_action, dict):
                if eef_name in gripper_action:
                    g = gripper_action[eef_name]
                else:
                    g = next(iter(gripper_action.values()))
            else:
                g = gripper_action

            if not isinstance(g, torch.Tensor):
                gripper_values = torch.as_tensor(g, device=self.device, dtype=torch.float32)
            else:
                gripper_values = g.to(device=self.device, dtype=torch.float32)

            if gripper_values.ndim == 0:
                gripper_values = gripper_values.reshape(1, 1)
            elif gripper_values.ndim == 1:
                gripper_values = gripper_values.reshape(-1, 1)
            else:
                gripper_values = gripper_values.reshape(gripper_values.shape[0], -1)[:, -1:]

            if gripper_values.shape[0] == 1 and target_pos.shape[0] > 1:
                gripper_values = gripper_values.repeat(target_pos.shape[0], 1)

        # Optional carry clamp: keep gripper closed while lifted but not in goal.
        if getattr(self.cfg, "hold_gripper_during_carry", False):
            goal_region = getattr(
                self.cfg,
                "goal_region",
                {
                    "x_min": 0.075,
                    "x_max": 0.225,
                    "y_min": 0.175,
                    "y_max": 0.325,
                    "z_min": 0.0,
                    "z_max": 0.15,
                    "max_lin_vel": 0.15,
                },
            )
            lifted = mimic_mdp.object_lifted(
                self,
                asset_cfg=SceneEntityCfg("object"),
                min_height=0.04,
            )

            above_goal = mimic_mdp.object_above_goal(
                self,
                asset_name="object",
                **self._above_goal_region(),
            )

            obj = self.scene["object"]
            root_state = obj.data.root_state_w

            object_z = root_state[:, 2]
            object_speed = torch.linalg.norm(root_state[:, 7:10], dim=-1)

            # Require:
            # - not yet sufficiently lowered
            # OR
            # - still moving too fast
            release_not_ready = (
                (object_z > 0.045)
                | (object_speed > 0.08)
            )

            if not isinstance(resolved_env_ids, slice):
                above_goal = above_goal[resolved_env_ids]
                release_not_ready = release_not_ready[resolved_env_ids]

            carry_closed_mask = lifted & (
                (~above_goal) | release_not_ready
            )

            if carry_closed_mask.any():
                closed_value = float(
                    getattr(self.cfg, "closed_gripper_value", gripper_values[0, 0].item())
                )
                gripper_values[carry_closed_mask, 0] = closed_value

        action = torch.cat([target_pos, target_quat, gripper_values], dim=-1)

        # If Mimic called this for one env, return a flat action vector.
        if action.shape[0] == 1:
            return action[0]

        return action

    def action_to_target_eef_pose(self, action: torch.Tensor, env_ids=None) -> dict[str, torch.Tensor]:
        """Convert DiffIK pose action back into a world-frame target EEF pose.

        Action layout:
            action[:, 0:3] = target EEF position in robot root frame
            action[:, 3:7] = target EEF quaternion in robot root frame, wxyz
            action[:, 7]   = gripper

        Mimic bookkeeping expects world-frame EEF pose matrices.
        """

        eef_name = list(self.cfg.subtask_configs.keys())[0]

        if action is None:
            return {eef_name: self.get_robot_eef_pose(eef_name, env_ids=env_ids)}

        action = action.to(device=self.device, dtype=torch.float32)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        if env_ids is not None:
            resolved_env_ids = self._resolve_env_ids(env_ids)
            if action.shape[0] == self.num_envs:
                action = action[resolved_env_ids]
        else:
            resolved_env_ids = torch.arange(action.shape[0], device=self.device, dtype=torch.long)

        target_pos_b = action[:, 0:3]
        target_quat_b = action[:, 3:7]
        target_quat_b = target_quat_b / torch.clamp(
            torch.linalg.norm(target_quat_b, dim=-1, keepdim=True),
            min=1e-8,
        )

        robot = self.scene["robot"]

        if isinstance(resolved_env_ids, slice):
            root_pos_w = robot.data.root_pos_w[: action.shape[0]]
            root_quat_w = robot.data.root_quat_w[: action.shape[0]]
        else:
            root_pos_w = robot.data.root_pos_w[resolved_env_ids]
            root_quat_w = robot.data.root_quat_w[resolved_env_ids]

        target_pos_w, target_quat_w = PoseUtils.combine_frame_transforms(
            root_pos_w,
            root_quat_w,
            target_pos_b,
            target_quat_b,
        )
        target_rot_w = PoseUtils.matrix_from_quat(target_quat_w)

        target_poses_w = torch.eye(
            4,
            device=self.device,
            dtype=target_pos_w.dtype,
        ).unsqueeze(0).repeat(action.shape[0], 1, 1)

        target_poses_w[:, 0:3, 0:3] = target_rot_w
        target_poses_w[:, 0:3, 3] = target_pos_w

        return {eef_name: target_poses_w}


    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from action trajectories.

        Supports both:
            [T, A]          from annotated HDF5 source demos
            [N, T, A]       from batched generated trajectories

        The gripper command is always the last action dimension.

        For this project:
            Pinocchio/source demos: [T, 6]
            DiffIK/generated demos: [T, 8] or [N, T, 8] when using pose mode
        """

        eef_name = list(self.cfg.subtask_configs.keys())[0]

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.device, dtype=torch.float32)

        if actions.ndim == 2:
            # [T, A] -> [T, 1]
            return {eef_name: actions[:, -1:]}

        if actions.ndim == 3:
            # [N, T, A] -> [N, T, 1]
            return {eef_name: actions[:, :, -1:]}

        raise ValueError(
            f"Expected actions with shape [T, A] or [N, T, A], got {tuple(actions.shape)}"
        )