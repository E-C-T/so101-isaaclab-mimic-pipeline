from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import SceneEntityCfg

import isaac_so_arm101.tasks.cube_mimic_i4h.mimic_mdp as mimic_mdp
from isaac_so_arm101.tasks.cube_mimic_i4h.pinocchio_ik import So101PinocchioIK


class SoArm101CubePinocchioMimicI4HEnv(ManagerBasedRLMimicEnv):

    def _resolve_env_ids(self, env_ids=None):
        """Normalize env_ids for indexing Isaac Lab tensors."""
        if env_ids is None:
            return slice(None)

        if isinstance(env_ids, slice):
            return env_ids

        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)

        if isinstance(env_ids, int):
            return torch.tensor([env_ids], device=self.device, dtype=torch.long)

        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    """SO-101 cube pick/place Mimic environment."""

    def get_subtask_term_signals(self) -> dict[str, torch.Tensor]:
        """Return subtask-completion signals for automatic Isaac Mimic annotation/generation.

        The task is internally segmented into phases:
            1. object_lifted
            2. object_above_goal
            3. object_in_goal

        These signals are shared by the Pinocchio and Diff IK Mimic envs.
        """

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

        above_goal_region = dict(goal_region)
        above_goal_region["z_min"] = 0.04
        above_goal_region["z_max"] = 0.25
        above_goal_region["max_lin_vel"] = 1.0

        signals = {
            "object_lifted": mimic_mdp.object_lifted(
                self,
                asset_cfg=SceneEntityCfg("object"),
                min_height=0.04,
            ),
            "object_above_goal": mimic_mdp.object_above_goal(
                self,
                asset_name="object",
                **above_goal_region,
            ),
            "object_in_goal": mimic_mdp.object_in_goal(
                self,
                asset_name="object",
                **goal_region,
            ),
        }

        if getattr(self.cfg, "debug_mimic_signals", False) and self.common_step_counter % 50 == 0:
            obj = self.scene["object"]
            root_state = obj.data.root_state_w[0]
            print(
                "[I4H PINOCCHIO MIMIC DEBUG]",
                "step=", self.common_step_counter,
                "object_pos=", root_state[0:3].detach().cpu().numpy(),
                "object_vel=", root_state[7:10].detach().cpu().numpy(),
                "object_lifted=", bool(signals["object_lifted"][0].detach().cpu()),
                "object_above_goal=", bool(signals["object_above_goal"][0].detach().cpu()),
                "object_in_goal=", bool(signals["object_in_goal"][0].detach().cpu()),
            )

        return signals

    def _get_ik_solver(self):
        if not hasattr(self, "_so101_ik_solver"):
            self._so101_ik_solver = So101PinocchioIK()
        return self._so101_ik_solver
    

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
        """Convert a Mimic world-frame EEF target pose to a 6D joint action.

        This is the Pinocchio path, so the target pose must be converted from
        Isaac world coordinates into the Pinocchio/URDF base frame before
        solving IK.

        Output action layout:
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """

        # ------------------------------------------------------------------
        # Resolve target EEF pose input.
        # Isaac Mimic may pass the pose as a tensor directly, or inside a dict.
        # The pose is expected to be a homogeneous matrix [4, 4] or [N, 4, 4]
        # in Isaac world coordinates.
        # ------------------------------------------------------------------
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

        if not isinstance(target_eef_pose, torch.Tensor):
            target_eef_pose_w = torch.as_tensor(
                target_eef_pose,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            target_eef_pose_w = target_eef_pose.to(device=self.device, dtype=torch.float32)

        if target_eef_pose_w.ndim == 2:
            target_eef_pose_w = target_eef_pose_w.unsqueeze(0)

        if target_eef_pose_w.shape[-2:] != (4, 4):
            raise ValueError(
                f"Expected target EEF pose as [4, 4] or [N, 4, 4], "
                f"got {tuple(target_eef_pose_w.shape)}"
            )

        # ------------------------------------------------------------------
        # Resolve env ids and batch size.
        # ------------------------------------------------------------------
        if env_id is not None:
            env_ids_resolved = torch.tensor([int(env_id)], device=self.device, dtype=torch.long)
        elif env_ids is not None:
            env_ids_resolved = self._resolve_env_ids(env_ids)
        else:
            env_ids_resolved = slice(None)

        if isinstance(env_ids_resolved, slice):
            selected_count = self.num_envs
        else:
            selected_count = int(env_ids_resolved.numel())

        # If Mimic gives one target pose but several envs are selected, repeat it.
        if target_eef_pose_w.shape[0] == 1 and selected_count > 1:
            target_eef_pose_w = target_eef_pose_w.repeat(selected_count, 1, 1)

        # ------------------------------------------------------------------
        # CRITICAL I4H PINOCCHIO BRIDGE:
        # target_eef_pose_w is in Isaac world frame.
        # Pinocchio IK expects target pose expressed in the Pinocchio URDF base.
        #
        # _get_robot_root_pose_matrix() returns T_world_pinbase, computed from
        # current measured Isaac EEF pose and Pinocchio FK at the same q.
        # ------------------------------------------------------------------
        T_world_pinbase = self._get_robot_root_pose_matrix(env_ids=env_ids_resolved)
        T_pinbase_world = self._invert_transform(T_world_pinbase)

        if target_eef_pose_w.shape[0] != T_pinbase_world.shape[0]:
            if target_eef_pose_w.shape[0] == 1:
                target_eef_pose_w = target_eef_pose_w.repeat(T_pinbase_world.shape[0], 1, 1)
            else:
                T_pinbase_world = T_pinbase_world[: target_eef_pose_w.shape[0]]

        target_eef_pose_pinbase = T_pinbase_world @ target_eef_pose_w

        # ------------------------------------------------------------------
        # Resolve gripper command.
        # ------------------------------------------------------------------
        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action", None)
        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action_dict", None)

        if isinstance(gripper_action, dict):
            gripper_action = gripper_action.get(
                eef_name,
                next(iter(gripper_action.values())),
            )

        robot = self.scene["robot"]

        if gripper_action is None:
            q_for_gripper = robot.data.joint_pos
            if not isinstance(env_ids_resolved, slice):
                q_for_gripper = q_for_gripper[env_ids_resolved]
            gripper_values = q_for_gripper[:, 5:6]
        else:
            if not isinstance(gripper_action, torch.Tensor):
                gripper_values = torch.as_tensor(
                    gripper_action,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                gripper_values = gripper_action.to(device=self.device, dtype=torch.float32)

            if gripper_values.ndim == 0:
                gripper_values = gripper_values.reshape(1, 1)
            elif gripper_values.ndim == 1:
                gripper_values = gripper_values.reshape(-1, 1)
            else:
                gripper_values = gripper_values.reshape(gripper_values.shape[0], -1)[:, -1:]

            if gripper_values.shape[0] == 1 and target_eef_pose_pinbase.shape[0] > 1:
                gripper_values = gripper_values.repeat(target_eef_pose_pinbase.shape[0], 1)

        # ------------------------------------------------------------------
        # Current joint seed.
        # ------------------------------------------------------------------
        q_current = robot.data.joint_pos
        if not isinstance(env_ids_resolved, slice):
            q_current = q_current[env_ids_resolved]

        if q_current.shape[0] != target_eef_pose_pinbase.shape[0]:
            if q_current.shape[0] == 1:
                q_current = q_current.repeat(target_eef_pose_pinbase.shape[0], 1)
            else:
                q_current = q_current[: target_eef_pose_pinbase.shape[0]]

        # Optional nominal posture. Keep it batch-compatible but do not force it.
        q_nominal_batch = None
        if q_nominal is not None:
            if not isinstance(q_nominal, torch.Tensor):
                q_nominal_batch = torch.as_tensor(q_nominal, device=self.device, dtype=torch.float32)
            else:
                q_nominal_batch = q_nominal.to(device=self.device, dtype=torch.float32)

            if q_nominal_batch.ndim == 1:
                q_nominal_batch = q_nominal_batch.unsqueeze(0)

            if q_nominal_batch.shape[0] == 1 and target_eef_pose_pinbase.shape[0] > 1:
                q_nominal_batch = q_nominal_batch.repeat(target_eef_pose_pinbase.shape[0], 1)

        ik_solver = self._get_ik_solver()
        actions = []

        for i in range(target_eef_pose_pinbase.shape[0]):
            target_np = target_eef_pose_pinbase[i].detach().cpu().numpy()
            q_current_np = q_current[i].detach().cpu().numpy()
            gripper_np = float(gripper_values[min(i, gripper_values.shape[0] - 1), 0].detach().cpu())

            q_nominal_np = None
            if q_nominal_batch is not None:
                q_nominal_np = q_nominal_batch[min(i, q_nominal_batch.shape[0] - 1)].detach().cpu().numpy()

            # The So101PinocchioIK wrapper expects the target matrix in its own
            # model frame. The argument name in the base class is historical.
            q_target_np = ik_solver.solve(
                q_current_6=q_current_np,
                target_T_world_eef=target_np,
                gripper_value=gripper_np,
                q_nominal_6=q_nominal_np,
                debug=False,
            )

            actions.append(
                torch.as_tensor(q_target_np, device=self.device, dtype=torch.float32)
            )

        action = torch.stack(actions, dim=0)

        if action.shape[0] == 1:
            return action[0]

        return action

    def action_to_target_eef_pose(
        self,
        action: torch.Tensor,
        env_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert Pinocchio Mimic actions into target EEF poses.

        Isaac Mimic uses this for:
        - trajectory reconstruction
        - annotation visualization
        - subtask stitching

        Input action format:
            [shoulder_pan,
            shoulder_lift,
            elbow_flex,
            wrist_flex,
            wrist_roll,
            gripper]
        """

        env_ids_resolved = self._resolve_env_ids(env_ids)

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(
                action,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            action = action.to(device=self.device, dtype=torch.float32)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        q_arm = action[:, :5]

        # ------------------------------------------------------------------
        # Reconstruct 6DOF robot joint vector
        # ------------------------------------------------------------------
        q_full = torch.zeros(
            (q_arm.shape[0], 6),
            device=self.device,
            dtype=torch.float32,
        )

        q_full[:, :5] = q_arm

        if action.shape[1] >= 6:
            q_full[:, 5] = action[:, 5]

        ik_solver = self._get_ik_solver()

        T_pinbase_eef_list = []

        for i in range(q_full.shape[0]):
            q_np = q_full[i].detach().cpu().numpy()

            T_np = ik_solver.forward_eef_pose(q_np)

            T_pinbase_eef_list.append(
                torch.as_tensor(
                    T_np,
                    device=self.device,
                    dtype=torch.float32,
                )
            )

        T_pinbase_eef = torch.stack(T_pinbase_eef_list, dim=0)

        # ------------------------------------------------------------------
        # Convert Pinocchio base frame -> Isaac world frame
        # ------------------------------------------------------------------
        T_world_pinbase = self._get_robot_root_pose_matrix(
            env_ids=env_ids_resolved
        )

        if T_world_pinbase.shape[0] != T_pinbase_eef.shape[0]:
            if T_world_pinbase.shape[0] == 1:
                T_world_pinbase = T_world_pinbase.repeat(
                    T_pinbase_eef.shape[0],
                    1,
                    1,
                )

        T_world_eef = T_world_pinbase @ T_pinbase_eef

        return {
            "end_effector": T_world_eef
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from action trajectories.

        Supports both:
            [T, A]
            [N, T, A]

        The gripper command is always the last action dimension.
        """

        eef_name = list(self.cfg.subtask_configs.keys())[0]

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.device, dtype=torch.float32)

        if actions.ndim == 2:
            return {eef_name: actions[:, -1:]}

        if actions.ndim == 3:
            return {eef_name: actions[:, :, -1:]}

        raise ValueError(
            f"Expected actions with shape [T, A] or [N, T, A], got {tuple(actions.shape)}"
        )
            
    def get_robot_eef_pose(self, eef_name: str = "end_effector", env_ids=None) -> torch.Tensor:
        """Return world-frame EEF pose as homogeneous matrix [N, 4, 4]."""
        env_ids = self._resolve_env_ids(env_ids)

        ee_frame = self.scene["ee_frame"]
        target_index = 0

        if hasattr(ee_frame.data, "target_frame_names"):
            names = list(ee_frame.data.target_frame_names)
            if eef_name in names:
                target_index = names.index(eef_name)

        pos_w = ee_frame.data.target_pos_w[env_ids, target_index, :]
        quat_w = ee_frame.data.target_quat_w[env_ids, target_index, :]

        pose_vec = torch.cat([pos_w, quat_w], dim=-1)
        return self._pose_vec_wxyz_to_matrix(pose_vec)
    
    def get_object_poses(self, env_ids=None) -> dict[str, torch.Tensor]:
        """Return object poses as homogeneous matrices for Isaac Mimic datagen.

        Returns:
            Dict mapping object name -> Tensor [N, 4, 4].
        """
        env_ids = self._resolve_env_ids(env_ids)

        obj = self.scene["object"]
        root_state = obj.data.root_state_w

        pose_vec = root_state[env_ids, 0:7].clone()
        return {
            "object": self._pose_vec_wxyz_to_matrix(pose_vec)
        }
    

    def _quat_wxyz_to_rotmat(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w, x, y, z] to rotation matrix.

        Args:
            quat: Tensor [..., 4] in wxyz order.

        Returns:
            Tensor [..., 3, 3].
        """
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)

        w = quat[..., 0]
        x = quat[..., 1]
        y = quat[..., 2]
        z = quat[..., 3]

        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z

        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        rot = torch.zeros((*quat.shape[:-1], 3, 3), device=quat.device, dtype=quat.dtype)

        rot[..., 0, 0] = ww + xx - yy - zz
        rot[..., 0, 1] = 2.0 * (xy - wz)
        rot[..., 0, 2] = 2.0 * (xz + wy)

        rot[..., 1, 0] = 2.0 * (xy + wz)
        rot[..., 1, 1] = ww - xx + yy - zz
        rot[..., 1, 2] = 2.0 * (yz - wx)

        rot[..., 2, 0] = 2.0 * (xz - wy)
        rot[..., 2, 1] = 2.0 * (yz + wx)
        rot[..., 2, 2] = ww - xx - yy + zz

        return rot

    def _pose_vec_wxyz_to_matrix(self, pose_vec: torch.Tensor) -> torch.Tensor:
        """Convert pose [x, y, z, qw, qx, qy, qz] to homogeneous matrix.

        Args:
            pose_vec: Tensor [N, 7].

        Returns:
            Tensor [N, 4, 4].
        """
        if pose_vec.ndim == 1:
            pose_vec = pose_vec.unsqueeze(0)

        pos = pose_vec[..., 0:3]
        quat = pose_vec[..., 3:7]

        pose_mat = torch.eye(4, device=pose_vec.device, dtype=pose_vec.dtype).repeat(pose_vec.shape[0], 1, 1)
        pose_mat[..., 0:3, 0:3] = self._quat_wxyz_to_rotmat(quat)
        pose_mat[..., 0:3, 3] = pos

        return pose_mat
    
    def _invert_transform(self, T: torch.Tensor) -> torch.Tensor:
        """Invert homogeneous transform matrix/matrices."""
        if T.ndim == 2:
            T = T.unsqueeze(0)

        T_inv = torch.eye(4, device=T.device, dtype=T.dtype).repeat(T.shape[0], 1, 1)
        R = T[:, :3, :3]
        p = T[:, :3, 3]

        R_inv = R.transpose(-1, -2)
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3] = -(R_inv @ p.unsqueeze(-1)).squeeze(-1)
        return T_inv

    def _get_robot_root_pose_matrix(self, env_ids=None) -> torch.Tensor:
        """Return world-from-Pinocchio-base transform [N, 4, 4].

        This is intentionally not the raw USD articulation root pose.

        For the I4H USD asset, the Isaac articulation root/base frame and the
        Pinocchio URDF base frame are not guaranteed to be identical. Build a
        dynamic bridge from the currently measured Isaac EEF pose and the
        Pinocchio FK EEF pose at the same q:

            T_world_pinbase = T_world_eef_isaac(q) @ inv(T_pinbase_eef_pin(q))
        """

        env_ids_resolved = self._resolve_env_ids(env_ids)

        if isinstance(env_ids_resolved, slice):
            selected_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            selected_env_ids = env_ids_resolved

        robot = self.scene["robot"]
        q_selected = robot.data.joint_pos[selected_env_ids, :6]

        # Isaac measured EEF pose in world frame.
        T_world_eef_isaac = self.get_robot_eef_pose(
            eef_name="end_effector",
            env_ids=selected_env_ids,
        )

        # Pinocchio FK EEF pose in Pinocchio base frame.
        ik_solver = self._get_ik_solver()
        T_pinbase_eef_list = []
        for i in range(q_selected.shape[0]):
            q_np = q_selected[i].detach().cpu().numpy()
            T_np = ik_solver.forward_eef_pose(q_np)
            T_pinbase_eef_list.append(
                torch.as_tensor(T_np, device=self.device, dtype=torch.float32)
            )

        T_pinbase_eef = torch.stack(T_pinbase_eef_list, dim=0)

        # Dynamic bridge:
        #   T_world_pinbase @ T_pinbase_eef = T_world_eef_isaac
        T_world_pinbase = T_world_eef_isaac @ self._invert_transform(T_pinbase_eef)

        if getattr(self.cfg, "debug_i4h_frame_bridge", False):
            if not hasattr(self, "_i4h_bridge_debug_count"):
                self._i4h_bridge_debug_count = 0
            if self._i4h_bridge_debug_count < 20:
                reconstructed = T_world_pinbase @ T_pinbase_eef
                pos_err = torch.linalg.norm(
                    reconstructed[:, :3, 3] - T_world_eef_isaac[:, :3, 3],
                    dim=-1,
                )
                print(
                    "[I4H PINOCCHIO FRAME BRIDGE]",
                    "count=", self._i4h_bridge_debug_count,
                    "T_world_pinbase_pos=", T_world_pinbase[0, :3, 3].detach().cpu().numpy(),
                    "eef_reconstruction_pos_err=", pos_err.detach().cpu().numpy(),
                )
                self._i4h_bridge_debug_count += 1

        return T_world_pinbase

# Backward-compatible alias for older scripts/imports.
SoArm101CubeJointPosMimicI4HEnv = SoArm101CubePinocchioMimicI4HEnv
