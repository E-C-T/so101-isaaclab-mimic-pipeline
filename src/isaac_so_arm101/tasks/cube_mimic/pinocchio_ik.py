from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class So101IKConfig:
    urdf_path: str = (
        "/home/insol02/IH_ws/so101_IsaacLab/"
        "src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
    )
    arm_joint_names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    gripper_joint_name: str = "gripper"
    ee_frame_name: str = "gripper_link"

    # Isaac FrameTransformer offset:
    # end_effector = gripper_link * offset
    ee_offset_xyz: tuple[float, float, float] = (0.01, 0.0, -0.09)

    # Weak scalar fallback posture prior.
    # Kept for backwards compatibility and as a simple fallback, but the preferred
    # fallback behavior is fallback_posture_weights below.
    # During direct replay debugging, prefer passing q_nominal_6 into solve(...),
    # so the IK follows the source demo's joint-space branch.
    posture_weight: float = 0.10

    # Per-joint fallback posture prior used when q_nominal_6 is NOT provided.
    # This is the regime normally used during standard Mimic generation unless
    # the generator is modified to pass source-demo posture metadata.
    #
    # Joint order:
    # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
    #
    # Tune this for generation. Keep it moderate: too high causes lag, too low
    # allows overshoot.
    fallback_posture_weights: tuple[float, float, float, float, float] = (
        0.02,  # shoulder_pan
        0.05,  # shoulder_lift
        0.12,  # elbow_flex
        0.12,  # wrist_flex
        0.02,  # wrist_roll
    )

    # Per-joint dynamic source-demo posture prior used when q_nominal_6 IS provided.
    # This is the regime used by direct replay with --use_source_posture_prior.
    #
    # Joint order:
    # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
    #
    # These weights bias the position-only IK toward the same joint-space branch
    # as the successful source demonstration.
    dynamic_posture_weights: tuple[float, float, float, float, float] = (
        0.05,  # shoulder_pan
        0.35,  # shoulder_lift
        0.35,  # elbow_flex
        0.35,  # wrist_flex
        0.05,  # wrist_roll
    )

    # Final blend toward the dynamic source-demo posture.
    # Only used when q_nominal_6 is passed.
    #
    # Keep this low for normal tuning. High values can pull the final solution
    # away from the EEF target after IK has solved the position objective.
    dynamic_posture_blend: float = 0.10

    nominal_arm_q: tuple[float, float, float, float, float] = (
        -0.0316,   # shoulder_pan
        -1.6728,   # shoulder_lift
        1.6850,    # elbow_flex, inside Isaac's limit
        1.3147,    # wrist_flex
        1.55,      # wrist_roll
    )

    # IK Params (DLS)
    damping: float = 2e-3 # high damp is smooth and slow
    step_size: float = 1.05  # IK update step, high step faster conv., might overshoot
    max_iters: int = 120  # Number of internal iterations per call
    pos_tol: float = 5e-4  # dist to target (m)
    max_delta_per_iter: float = 0.32  # limits each internal IK iteration’s joint-space update norm
    max_total_delta: float = 1.70  # max total IK solution movement away from q_current in one action call

    # Use finite differences first to avoid Jacobian row convention issues.
    use_finite_difference_jacobian: bool = True
    finite_difference_eps: float = 1e-5


class So101PinocchioIK:
    """Position-first IK bridge for SO-101.

    Input target is Mimic's end_effector pose.
    Solver target is Pinocchio's gripper_link pose after removing the Isaac
    FrameTransformer offset.
    """

    def __init__(self, cfg: So101IKConfig | None = None):
        self.cfg = cfg or So101IKConfig()

        try:
            import pinocchio as pin
        except Exception as exc:
            raise RuntimeError(
                "Failed to import pinocchio. Run through IsaacLab's Python environment."
            ) from exc

        self.pin = pin
        self.model = pin.buildModelFromUrdf(self.cfg.urdf_path)
        self.data = self.model.createData()

        self.frame_id = self.model.getFrameId(self.cfg.ee_frame_name)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"Frame not found in Pinocchio model: {self.cfg.ee_frame_name}")

        self.arm_q_ids: list[int] = []
        self.arm_v_ids: list[int] = []
        for joint_name in self.cfg.arm_joint_names:
            joint_id = self.model.getJointId(joint_name)
            if joint_id == 0 or joint_id >= len(self.model.joints):
                raise ValueError(f"Joint not found in Pinocchio model: {joint_name}")
            joint = self.model.joints[joint_id]
            self.arm_q_ids.append(joint.idx_q)
            self.arm_v_ids.append(joint.idx_v)

        gripper_joint_id = self.model.getJointId(self.cfg.gripper_joint_name)
        if gripper_joint_id == 0 or gripper_joint_id >= len(self.model.joints):
            raise ValueError(f"Joint not found in Pinocchio model: {self.cfg.gripper_joint_name}")
        self.gripper_q_id = self.model.joints[gripper_joint_id].idx_q

        self.lower = np.array(self.model.lowerPositionLimit, dtype=np.float64)
        self.upper = np.array(self.model.upperPositionLimit, dtype=np.float64)

        self.lower = np.where(np.isfinite(self.lower), self.lower, -np.pi)
        self.upper = np.where(np.isfinite(self.upper), self.upper, np.pi)

        self.T_link_eef = self.pin.SE3(
            np.eye(3),
            np.array(self.cfg.ee_offset_xyz, dtype=np.float64),
        )
        self.T_eef_link = self.T_link_eef.inverse()

    def _se3_to_matrix(self, T) -> np.ndarray:
        """Convert Pinocchio SE3 to 4x4 numpy matrix."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = T.rotation
        M[:3, 3] = T.translation
        return M

    def forward_eef_pose(self, q_6: np.ndarray) -> np.ndarray:
        """Compute base-frame end_effector pose from a 6D SO-101 joint vector.

        Args:
            q_6:
                [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

        Returns:
            T_base_eef as [4, 4].
        """
        q_6 = np.asarray(q_6, dtype=np.float64).reshape(-1)
        if q_6.shape[0] < 6:
            raise ValueError(f"Expected at least 6 joint values, got {q_6.shape}")

        q = self.pin.neutral(self.model)
        q[:6] = q_6[:6]

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

        T_base_link = self.data.oMf[self.frame_id]

        # end_effector = gripper_link * Isaac FrameTransformer offset
        T_base_eef = T_base_link * self.T_link_eef

        return self._se3_to_matrix(T_base_eef)

    def _matrix_to_se3(self, T: np.ndarray):
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Expected target pose shape (4, 4), got {T.shape}")
        return self.pin.SE3(T[:3, :3], T[:3, 3])

    def _fk_link_pose(self, q: np.ndarray):
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id]

    def _fk_link_translation(self, q: np.ndarray) -> np.ndarray:
        return self._fk_link_pose(q).translation.copy()

    def _finite_difference_position_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Numerically compute d p_link / d q_arm in world coordinates.

        Returns:
            J_pos: [3, num_arm_joints]
        """
        eps = self.cfg.finite_difference_eps
        J = np.zeros((3, len(self.arm_q_ids)), dtype=np.float64)

        for local_i, q_idx in enumerate(self.arm_q_ids):
            q_plus = q.copy()
            q_minus = q.copy()

            q_plus[q_idx] += eps
            q_minus[q_idx] -= eps

            p_plus = self._fk_link_translation(q_plus)
            p_minus = self._fk_link_translation(q_minus)

            J[:, local_i] = (p_plus - p_minus) / (2.0 * eps)

        return J

    def _analytic_position_jacobian_candidates(self, q: np.ndarray):
        """Return both common row candidates for debugging analytic Jacobian convention."""
        J_full = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return {
            "rows_0_3": J_full[0:3, :][:, self.arm_v_ids],
            "rows_3_6": J_full[3:6, :][:, self.arm_v_ids],
        }

    def _get_posture_weights_and_nominal(
        self,
        q_nominal_arm: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Select posture target and per-joint weights.

        If q_nominal_arm is provided, use the dynamic source-demo prior.
        Otherwise, use the fallback nominal posture prior.
        """
        if q_nominal_arm is not None:
            q_nom = q_nominal_arm
            posture_weights = np.array(self.cfg.dynamic_posture_weights, dtype=np.float64)
            used_dynamic_prior = True
        else:
            q_nom = np.array(self.cfg.nominal_arm_q, dtype=np.float64)
            q_nom = np.clip(q_nom, self.lower[self.arm_q_ids], self.upper[self.arm_q_ids])

            if hasattr(self.cfg, "fallback_posture_weights"):
                posture_weights = np.array(self.cfg.fallback_posture_weights, dtype=np.float64)
            else:
                posture_weights = np.full(len(self.arm_q_ids), self.cfg.posture_weight, dtype=np.float64)

            used_dynamic_prior = False

        if posture_weights.shape[0] != len(self.arm_q_ids):
            raise ValueError(
                f"Expected {len(self.arm_q_ids)} posture weights, got {posture_weights.shape}"
            )

        return q_nom, posture_weights, used_dynamic_prior

    def solve(
        self,
        q_current_6: np.ndarray,
        target_T_world_eef: np.ndarray,
        gripper_value: float | None = None,
        q_nominal_6: np.ndarray | None = None,
        debug: bool = False,
    ) -> np.ndarray:
        """Solve position IK and return absolute 6D joint target.

        Args:
            q_current_6:
                [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].
            target_T_world_eef:
                Desired Mimic end_effector pose as [4, 4].
            gripper_value:
                Optional gripper target. If None, preserve current gripper.
            q_nominal_6:
                Optional source-demo joint posture used as a dynamic prior.
                This biases the position-only IK toward the same joint-space branch
                that generated the successful demonstration.
            debug:
                Print convergence diagnostics.

        Returns:
            q_target_6:
                Absolute joint target in the current env action layout.
        """
        q_current_6 = np.asarray(q_current_6, dtype=np.float64).reshape(-1)
        if q_current_6.shape[0] < 6:
            raise ValueError(f"Expected at least 6 joint values, got {q_current_6.shape}")

        q_nominal_arm = None
        if q_nominal_6 is not None:
            q_nominal_6 = np.asarray(q_nominal_6, dtype=np.float64).reshape(-1)
            if q_nominal_6.shape[0] < 5:
                raise ValueError(f"Expected q_nominal_6 to have at least 5 values, got {q_nominal_6.shape}")

            q_nominal_arm = q_nominal_6[:5].copy()
            q_nominal_arm = np.clip(
                q_nominal_arm,
                self.lower[self.arm_q_ids],
                self.upper[self.arm_q_ids],
            )

        q = self.pin.neutral(self.model)
        q[:6] = q_current_6[:6]
        q_start = q.copy()

        T_world_eef_target = self._matrix_to_se3(target_T_world_eef)

        # target_eef = target_link * offset, so target_link = target_eef * inv(offset)
        T_world_link_target = T_world_eef_target * self.T_eef_link
        p_target = T_world_link_target.translation.copy()

        initial_p = self._fk_link_translation(q)
        initial_err = float(np.linalg.norm(p_target - initial_p))
        last_err = initial_err

        q_nom, posture_weights, used_dynamic_prior = self._get_posture_weights_and_nominal(q_nominal_arm)

        for it in range(self.cfg.max_iters):
            p_current = self._fk_link_translation(q)
            pos_err = p_target - p_current
            err_norm = float(np.linalg.norm(pos_err))
            last_err = err_norm

            if err_norm < self.cfg.pos_tol:
                break

            if self.cfg.use_finite_difference_jacobian:
                J_pos = self._finite_difference_position_jacobian(q)
            else:
                # Use this later once we verify row convention.
                self.pin.forwardKinematics(self.model, self.data, q)
                self.pin.updateFramePlacements(self.model, self.data)
                J_candidates = self._analytic_position_jacobian_candidates(q)
                J_pos = J_candidates["rows_0_3"]

            # Damped least squares:
            # dq = J.T (J J.T + lambda^2 I)^-1 error
            A = J_pos @ J_pos.T + (self.cfg.damping**2) * np.eye(3)
            dq_arm = J_pos.T @ np.linalg.solve(A, pos_err)

            # Posture regularization to prevent position-only IK from choosing
            # the wrong joint-space branch.
            #
            # If a source-demo q_nominal_6 is provided, use dynamic_posture_weights.
            # Otherwise, use fallback_posture_weights toward nominal_arm_q.
            q_arm = q[self.arm_q_ids]
            dq_posture = posture_weights * (q_nom - q_arm)
            dq_arm = dq_arm + dq_posture

            dq_norm = float(np.linalg.norm(dq_arm))
            if dq_norm > self.cfg.max_delta_per_iter:
                dq_arm = dq_arm * (self.cfg.max_delta_per_iter / max(dq_norm, 1e-8))

            for local_i, q_idx in enumerate(self.arm_q_ids):
                q[q_idx] += self.cfg.step_size * dq_arm[local_i]

            # Clip joint limits.
            q[self.arm_q_ids] = np.clip(
                q[self.arm_q_ids],
                self.lower[self.arm_q_ids],
                self.upper[self.arm_q_ids],
            )

            # Limit total jump away from current configuration.
            total_delta = q[self.arm_q_ids] - q_start[self.arm_q_ids]
            total_delta_norm = float(np.linalg.norm(total_delta))
            if total_delta_norm > self.cfg.max_total_delta:
                total_delta = total_delta * (self.cfg.max_total_delta / max(total_delta_norm, 1e-8))
                q[self.arm_q_ids] = q_start[self.arm_q_ids] + total_delta

        final_p = self._fk_link_translation(q)
        final_err = float(np.linalg.norm(p_target - final_p))

        q_solved_arm = q[self.arm_q_ids].copy()

        # Final source-branch blend.
        # This is only used when q_nominal_6 is passed.
        # It helps recover the demonstration branch after the position-only IK solve.
        if used_dynamic_prior and self.cfg.dynamic_posture_blend > 0.0:
            blend = float(self.cfg.dynamic_posture_blend)
            blend = min(max(blend, 0.0), 1.0)

            q_solved_arm = (1.0 - blend) * q_solved_arm + blend * q_nominal_arm
            q_solved_arm = np.clip(
                q_solved_arm,
                self.lower[self.arm_q_ids],
                self.upper[self.arm_q_ids],
            )

        q_target = q_current_6.copy()
        q_target[:5] = q_solved_arm

        if gripper_value is not None:
            q_target[5] = float(gripper_value)
        else:
            q_target[5] = q_current_6[5]

        if debug:
            print("[SO101 IK DEBUG]")
            print("  p_start       =", initial_p)
            print("  p_target      =", p_target)
            print("  p_final_pred  =", final_p)
            print("  initial_err   =", initial_err)
            print("  final_err     =", final_err)
            print("  q_current     =", q_current_6)
            print("  q_nominal_6   =", q_nominal_6)
            print("  used_dynamic_prior =", used_dynamic_prior)
            print("  q_nom         =", q_nom)
            print("  posture_weights =", posture_weights)
            print("  q_solved_arm  =", q_solved_arm)
            print("  q_target      =", q_target)

        return q_target