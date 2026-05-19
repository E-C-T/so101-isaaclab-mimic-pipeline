import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass

from isaac_so_arm101.robots import SO_ARM101_CFG
from isaac_so_arm101.tasks.cube_replay.replay_env_cfg import SoArm101CubeReplayEnvCfg


@configclass
class SoArm101CubeReplayEnvCfg_PLAY(SoArm101CubeReplayEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM101_CFG.init_state.replace(
                rot=(0.0, 0.0, 0.0, 1.0)
            ),
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=1.0,
            use_default_offset=False,
        )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["gripper"],
        #     open_command_expr={"gripper": 0.5},
        #     close_command_expr={"gripper": 0.0},
        # )

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
            use_default_offset=False,
        )

        # REQUIRED because PickPlaceEnvCfg leaves this as MISSING.
        self.commands.object_pose.body_name = ["gripper_link"]

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
                ),
            ],
        )



# Data Conversion note for urdf: 

#     robot_group.create_dataset(
#         "root_pose",
#         data=np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
#         compression="gzip",
#     )
#     robot_group.create_dataset(
#         "root_velocity",
#         data=np.zeros((1, 6), dtype=np.float32),
#         compression="gzip",
#     )