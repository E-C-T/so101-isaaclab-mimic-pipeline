from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import SceneEntityCfg

import isaac_so_arm101.tasks.cube_mimic.mimic_mdp as mimic_mdp
from isaac_so_arm101.tasks.cube_mimic.pinocchio_ik import So101PinocchioIK


class SoArm101CubeJointPosMimicEnv(ManagerBasedRLMimicEnv):
    """SO-101 cube pick/place Mimic environment."""

    def get_subtask_term_signals(self) -> dict[str, torch.Tensor]:
        """Return subtask-completion signals for automatic Isaac Mimic annotation."""

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

        # signals = {
        #     "object_in_goal": mimic_mdp.object_in_goal(
        #         self,
        #         asset_name="object",
        #         **goal_region,
        #     ),
        # }

        signals = {
            "object_lifted": mimic_mdp.object_lifted(
                self,
                asset_cfg=SceneEntityCfg("object"),
                min_height=0.04,
            ),
            "object_in_goal": mimic_mdp.object_in_goal(
                self,
                asset_name="object",
                **goal_region,
            ),
        }


        if self.common_step_counter % 50 == 0:
            obj = self.scene["object"]
            root_state = obj.data.root_state_w[0]
            print(
                "[MIMIC DEBUG]",
                "step=", self.common_step_counter,
                "object_pos=", root_state[0:3].detach().cpu().numpy(),
                "object_vel=", root_state[7:10].detach().cpu().numpy(),
                "object_in_goal=", bool(signals["object_in_goal"][0].detach().cpu()),
            )

        return signals
    

    def _resolve_env_ids(self, env_ids=None):
        """Convert Isaac Mimic env_ids argument into an index usable on tensors."""
        if env_ids is None:
            return slice(None)

        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)

        if isinstance(env_ids, int):
            return torch.tensor([env_ids], device=self.device, dtype=torch.long)

        # Handles list/tuple like [0], [1], etc.
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)
    

    # def action_to_target_eef_pose(self, actions: torch.Tensor, env_ids=None) -> dict[str, torch.Tensor]:
    #     """Convert absolute joint-position actions to target EEF poses.

    #     For this SO-101 JointPos Mimic env, actions are absolute joint targets:
    #         [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

    #     Therefore the target EEF pose should be FK(action), not the current measured EEF pose.
    #     Isaac Mimic stores this in datagen_info and later uses it for waypoint generation.
    #     """
    #     if actions is None:
    #         return {
    #             "end_effector": self.get_robot_eef_pose("end_effector", env_ids=env_ids)
    #         }

    #     actions = actions.to(device=self.device, dtype=torch.float32)
    #     if actions.ndim == 1:
    #         actions = actions.unsqueeze(0)

    #     if env_ids is not None:
    #         resolved_env_ids = self._resolve_env_ids(env_ids)
    #         selected_env_ids = resolved_env_ids
    #         if actions.shape[0] == self.num_envs:
    #             actions = actions[resolved_env_ids]
    #     else:
    #         selected_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    #     ik_solver = self._get_ik_solver()

    #     # Convert action-space FK from robot base frame into Isaac world frame.
    #     T_world_base = self._get_robot_root_pose_matrix(env_ids=selected_env_ids)

    #     output_poses = []
    #     for i in range(actions.shape[0]):
    #         q_action_np = actions[i, :6].detach().cpu().numpy()
    #         T_base_eef_np = ik_solver.forward_eef_pose(q_action_np)

    #         T_base_eef = torch.as_tensor(
    #             T_base_eef_np,
    #             device=self.device,
    #             dtype=torch.float32,
    #         )

    #         T_world_eef = T_world_base[i] @ T_base_eef
    #         output_poses.append(T_world_eef)

    #     return {
    #         "end_effector": torch.stack(output_poses, dim=0)
    #     }

    def action_to_target_eef_pose(self, actions: torch.Tensor, env_ids=None) -> dict[str, torch.Tensor]:
        """Return the realized/measured EEF pose as the target EEF pose.

        For this absolute joint-position SO-101 environment, FK(action) is the
        commanded joint-space target, but it is not necessarily the realized EEF
        trajectory under Isaac physics/actuator dynamics.

        Isaac Mimic needs a task-space trajectory to transform. For this bridge,
        use the measured EEF pose from the replayed demonstration.
        """
        return {
            "end_effector": self.get_robot_eef_pose("end_effector", env_ids=env_ids)
        }


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
        """Convert Mimic target EEF pose to absolute SO-101 joint-position action.

        Returns:
            action with layout:
                [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

        Notes:
            q_nominal is optional and is mainly used for direct replay/debugging.
            It provides a source-demo joint posture prior to the position-only IK solver,
            helping the IK stay on the same joint-space branch as the successful demo.
        """

        if target_eef_pose is None:
            target_eef_pose = kwargs.get("target_eef_pose", None)
        if target_eef_pose is None:
            target_eef_pose = kwargs.get("target_eef_pose_dict", None)
        if target_eef_pose is None:
            target_eef_pose = kwargs.get("eef_pose", None)

        if isinstance(target_eef_pose, dict):
            target_eef_pose = target_eef_pose.get(eef_name, next(iter(target_eef_pose.values())))

        if target_eef_pose is None:
            raise ValueError("target_eef_pose_to_action requires target_eef_pose or target_eef_pose_dict.")

        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action", None)
        if gripper_action is None:
            gripper_action = kwargs.get("gripper_action_dict", None)

        # Optional source-demo posture prior.
        # Support either explicit q_nominal argument or kwargs fallback.
        if q_nominal is None:
            q_nominal = kwargs.get("q_nominal", None)
        if q_nominal is None:
            q_nominal = kwargs.get("q_nominal_6", None)
        if q_nominal is None:
            q_nominal = kwargs.get("source_action", None)

        robot = self.scene["robot"]
        q_all = robot.data.joint_pos[:, :6].clone()

        if env_ids is not None:
            resolved_env_ids = self._resolve_env_ids(env_ids)
            selected_q = q_all[resolved_env_ids]
            selected_env_ids = resolved_env_ids
            single_env = not isinstance(resolved_env_ids, slice) and len(resolved_env_ids) == 1
        elif env_id is not None:
            selected_env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
            selected_q = q_all[selected_env_ids]
            single_env = True
        else:
            selected_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            selected_q = q_all
            single_env = False

        target = target_eef_pose
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        else:
            target = target.to(device=self.device, dtype=torch.float32)

        # Accept [4, 4], [1, 4, 4], or [N, 4, 4].
        if target.ndim == 2:
            target = target.unsqueeze(0)

        if target.shape[-2:] != (4, 4):
            raise ValueError(f"Expected target EEF pose as 4x4 matrix, got shape {tuple(target.shape)}")

        if target.shape[0] == 1 and selected_q.shape[0] > 1:
            target = target.repeat(selected_q.shape[0], 1, 1)

        # Normalize q_nominal if provided.
        q_nominal_tensor = None
        if q_nominal is not None:
            if not isinstance(q_nominal, torch.Tensor):
                q_nominal_tensor = torch.as_tensor(q_nominal, device=self.device, dtype=torch.float32)
            else:
                q_nominal_tensor = q_nominal.to(device=self.device, dtype=torch.float32)

            if q_nominal_tensor.ndim == 1:
                q_nominal_tensor = q_nominal_tensor.unsqueeze(0)

            if q_nominal_tensor.shape[0] == 1 and selected_q.shape[0] > 1:
                q_nominal_tensor = q_nominal_tensor.repeat(selected_q.shape[0], 1)

        # Extract gripper command if provided. Otherwise preserve current gripper.
        gripper_values = selected_q[:, 5].clone()

        if gripper_action is not None:
            if isinstance(gripper_action, dict):
                if eef_name in gripper_action:
                    g = gripper_action[eef_name]
                else:
                    g = next(iter(gripper_action.values()))
            else:
                g = gripper_action

            if not isinstance(g, torch.Tensor):
                g = torch.as_tensor(g, device=self.device, dtype=selected_q.dtype)
            else:
                g = g.to(device=self.device, dtype=selected_q.dtype)

            if g.ndim == 0:
                g = g.reshape(1)
            elif g.ndim > 1:
                g = g.reshape(g.shape[0], -1)[:, -1]

            if g.shape[0] == 1 and selected_q.shape[0] > 1:
                g = g.repeat(selected_q.shape[0])

            gripper_values = g[: selected_q.shape[0]]

        ik_solver = self._get_ik_solver()

        # Compute root transform once per call.
        # Mimic target EEF pose is in Isaac world frame.
        # Pinocchio IK expects robot-base/URDF frame.
        T_world_base = self._get_robot_root_pose_matrix(env_ids=selected_env_ids)
        T_base_world = self._invert_transform(T_world_base)

        # Current EEF poses for debug and optional target step limiting.
        current_eef_world_all = self.get_robot_eef_pose(eef_name, env_ids=selected_env_ids)

        output_actions = []
        for i in range(selected_q.shape[0]):
            q_current_np = selected_q[i].detach().cpu().numpy()
            gripper_np = float(gripper_values[i].detach().cpu().item())

            current_eef_world = current_eef_world_all[i]
            target_eef_world = target[i].clone()

            current_pos = current_eef_world[:3, 3]
            target_pos = target_eef_world[:3, 3]
            delta = target_pos - current_pos
            dist = torch.linalg.norm(delta)

            # if not hasattr(self, "_ik_debug_counter"):
            #     self._ik_debug_counter = 0

            # if self._ik_debug_counter < 50:
            #     print(
            #         "[IK TARGET DEBUG]",
            #         "call=", self._ik_debug_counter,
            #         "current_pos=", current_pos.detach().cpu().numpy(),
            #         "requested_target_pos=", target_pos.detach().cpu().numpy(),
            #         "used_target_pos=", target_eef_world[:3, 3].detach().cpu().numpy(),
            #         "dist=", float(dist.detach().cpu()),
            #         "q_current=", q_current_np,
            #         "has_q_nominal=", q_nominal_tensor is not None,
            #     )

            # Prevent one Mimic waypoint from requesting a huge jump.
            # Keep permissive for now because the measured EEF trajectory is already reasonable.
            max_eef_step = 1.00  # meters
            if dist > max_eef_step:
                target_eef_world[:3, 3] = current_pos + delta / dist.clamp_min(1e-8) * max_eef_step

            # Convert target EEF pose from Isaac world frame to robot base/URDF frame.
            T_base_eef = T_base_world[i] @ target_eef_world
            target_np = T_base_eef.detach().cpu().numpy()

            q_nominal_np = None
            if q_nominal_tensor is not None:
                q_nominal_np = q_nominal_tensor[i, :6].detach().cpu().numpy()

            q_target_np = ik_solver.solve(
                q_current_6=q_current_np,
                target_T_world_eef=target_np,
                gripper_value=gripper_np,
                q_nominal_6=q_nominal_np,
                debug=False,
            )

            # Clamp returned absolute joint target relative to current joint state.
            # This prevents large jumps from IK when targets are difficult or unreachable.
            max_joint_delta = torch.tensor(
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                device=self.device,
                dtype=selected_q.dtype,
            ).detach().cpu().numpy()

            dq = q_target_np - q_current_np
            dq = dq.clip(-max_joint_delta, max_joint_delta)
            q_target_np = q_current_np + dq

            # if self._ik_debug_counter < 50:
            #     print(
            #         "[IK ACTION DEBUG]",
            #         "q_target=", q_target_np,
            #         "dq=", q_target_np - q_current_np,
            #     )
            # self._ik_debug_counter += 1

            output_actions.append(torch.as_tensor(q_target_np, device=self.device, dtype=selected_q.dtype))

        action = torch.stack(output_actions, dim=0)

        # Optional action noise. Keep disabled unless you intentionally want it.
        # Mimic may pass per-EEF noise dictionaries.
        if action_noise_dict is not None:
            pass

        if single_env:
            return action[0]

        return action


    def actions_to_gripper_actions(self, actions: torch.Tensor, env_ids=None) -> dict[str, torch.Tensor]:
        """Extract gripper action from the absolute joint-position action tensor.

        Current action layout:
            [5 arm joint targets, 1 gripper joint target]
        """
        if actions is None:
            env_ids_resolved = self._resolve_env_ids(env_ids)
            if isinstance(env_ids_resolved, slice):
                n = self.num_envs
            else:
                n = len(env_ids_resolved)
            gripper = torch.zeros((n, 1), device=self.device)
        else:
            actions = actions.to(self.device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)

            if env_ids is not None and actions.shape[0] == self.num_envs:
                env_ids_resolved = self._resolve_env_ids(env_ids)
                actions = actions[env_ids_resolved]

            gripper = actions[:, -1:].clone()

        return {
            "end_effector": gripper
        }
    
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
        """Return robot root pose as world-from-base transform [N, 4, 4]."""
        env_ids = self._resolve_env_ids(env_ids)
        robot = self.scene["robot"]
        root_state = robot.data.root_state_w
        root_pose = root_state[env_ids, 0:7]
        return self._pose_vec_wxyz_to_matrix(root_pose)