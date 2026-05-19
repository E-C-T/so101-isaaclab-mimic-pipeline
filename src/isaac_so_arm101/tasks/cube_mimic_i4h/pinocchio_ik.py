from __future__ import annotations

from dataclasses import dataclass

from isaac_so_arm101.tasks.cube_mimic.pinocchio_ik import (
    So101IKConfig as _BaseSo101IKConfig,
    So101PinocchioIK as _BaseSo101PinocchioIK,
)


@dataclass
class So101IKConfig(_BaseSo101IKConfig):
    """I4H-specific SO101 Pinocchio IK configuration.

    The geometry is still loaded from the SO101 URDF because the I4H USD uses
    the same joint names and kinematic structure. The important differences are
    the nominal posture and posture priors used during Mimic generation.

    The old URDF nominal posture used wrist_roll around +1.55 rad. For the I4H
    converted dataset, the corrected wrist convention is negative, so using the
    old nominal silently pulls generated IK back to the stale wrist branch.
    """

    # Keep same URDF geometry source.
    urdf_path: str = (
        "/home/insol02/IH_ws/so101_IsaacLab/"
        "src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
    )

    # Use the same Isaac FrameTransformer offset as the task cfg.
    ee_offset_xyz: tuple[float, float, float] = (0.01, 0.0, -0.09)

    # I4H corrected fallback nominal posture.
    # elbow_flex is kept inside the USD-authored joint limit.
    nominal_arm_q: tuple[float, float, float, float, float] = (
        -0.13564736,   # shoulder_pan
        -1.6236473,   # shoulder_lift
        1.685,    # elbow_flex, inside the I4H USD limit
        1.3117621,    # wrist_flex
        -1.759493,   # wrist_roll, corrected I4H convention
    )

    # For generation without q_nominal, strongly avoid drifting back to the old
    # wrist branch, while still allowing position IK to move.
    fallback_posture_weights: tuple[float, float, float, float, float] = (
        0.08,  # shoulder_pan
        0.08,  # shoulder_lift
        0.08,  # elbow_flex
        0.08,  # wrist_flex
        0.08,  # wrist_roll
    )

    # When a source posture is ever provided, let it dominate branch choice.
    dynamic_posture_weights: tuple[float, float, float, float, float] = (
        0.05,  # shoulder_pan
        0.35,  # shoulder_lift
        0.35,  # elbow_flex
        0.35,  # wrist_flex
        0.05,  # wrist_roll
    )

    dynamic_posture_blend: float = 0.10


class So101PinocchioIK(_BaseSo101PinocchioIK):
    """I4H-specific IK wrapper using the corrected I4H IK config."""

    def __init__(self, cfg: So101IKConfig | None = None):
        super().__init__(cfg or So101IKConfig())