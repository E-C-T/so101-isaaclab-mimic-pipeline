from __future__ import annotations

from isaaclab.utils import configclass

from isaac_so_arm101.tasks.camera_config.so101_i4h_cameras import (
    make_i4h_up_camera_cfg,
    make_i4h_wrist_camera_cfg,
)
from isaac_so_arm101.tasks.cube_replay_i4h.joint_pos_env_cfg import (
    SoArm101CubeReplayI4HEnvCfg_PLAY,
)


@configclass
class SoArm101CubeReplayI4HCameraEnvCfg_PLAY(SoArm101CubeReplayI4HEnvCfg_PLAY):
    """Camera-enabled I4H cube replay config.

    Use this for rendering/recording camera observations after trajectories
    have already been generated or filtered.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.wrist_camera = make_i4h_wrist_camera_cfg(
            prim_prefix="{ENV_REGEX_NS}/Robot",
            width=640,
            height=480,
            fps=30.0,
        )

        self.scene.up_camera = make_i4h_up_camera_cfg(
            prim_path="{ENV_REGEX_NS}/UpCamera",
            width=640,
            height=480,
            fps=30.0,
        )