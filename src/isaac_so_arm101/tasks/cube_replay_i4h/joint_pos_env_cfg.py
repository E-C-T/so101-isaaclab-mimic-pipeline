from __future__ import annotations

import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass

# New USD robot config you should define separately.
# Suggested path:
#   src/isaac_so_arm101/robots/i4h_so101/so_arm101_i4h.py
from isaac_so_arm101.robots.i4h_so101.so_arm101_i4h import SO_ARM101_I4H_USD_CFG

# Reuse the same cube/table/object base task for now.
from isaac_so_arm101.tasks.cube_replay_i4h.replay_env_cfg import SoArm101CubeReplayI4HEnvCfg


@configclass
class SoArm101CubeReplayI4HEnvCfg_PLAY(SoArm101CubeReplayI4HEnvCfg):
    """SO-101 cube replay env using the Isaac 4 Healthcare SO-ARM USD asset.

    This keeps the same cube task and joint-position action interface, but swaps
    the robot backend from runtime URDF import to the curated I4H USD.
    """

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------------
        # Robot: I4H USD SO-ARM
        # ---------------------------------------------------------------------
        # The I4H example uses robot prim path "{ENV_REGEX_NS}/robot" lowercase,
        # but your existing SO101 task uses "{ENV_REGEX_NS}/Robot" uppercase.
        #
        # Keep uppercase Robot for compatibility with:
        #   - camera configs
        #   - frame transformer paths
        #   - mimic code assumptions
        #   - prior debug scripts
        # I4H USD has /Robot/base/baseframe offset from /Robot/base.
        # We place the articulation root so that base/baseframe is at world origin.
        # Correction for HDF5 Meta data, root pose can be corrected when converting the data
        BASEFRAME_LOCAL_POS = (0.02079, 0.01576, 0.03248)

        self.scene.robot = SO_ARM101_I4H_USD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM101_I4H_USD_CFG.init_state.replace(
                pos=(
                    -BASEFRAME_LOCAL_POS[0],
                    -BASEFRAME_LOCAL_POS[1],
                    -BASEFRAME_LOCAL_POS[2],
                ),
                rot=(0.707, 0.0, 0.0, 0.707),
            ),
        )

        # ---------------------------------------------------------------------
        # Actions: preserve your current absolute joint-position interface
        # ---------------------------------------------------------------------
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=1.0,
            use_default_offset=False,
        )

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
            use_default_offset=False,
        )

        # REQUIRED because PickPlaceEnvCfg leaves this as MISSING.
        # This assumes the USD still has a gripper_link body.
        # If not, inspect the USD link/body names and replace this.
        self.commands.object_pose.body_name = ["gripper"]

        # ---------------------------------------------------------------------
        # End-effector frame for replay/mimic datagen
        # ---------------------------------------------------------------------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
                ),
            ],
        )