from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg


def make_i4h_wrist_camera_cfg(
    prim_prefix: str = "{ENV_REGEX_NS}/Robot",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> TiledCameraCfg:
    """Attach to the existing embedded I4H USD wrist camera."""
    return TiledCameraCfg(
        prim_path=f"{prim_prefix}/gripper/visuals/pcb_board_36x36/Camera",
        spawn=None,
        data_types=["rgb"],
        width=width,
        height=height,
        update_period=1.0 / fps,
    )


def make_i4h_up_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/UpCamera",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> TiledCameraCfg:
    """Spawn a room/up camera similar to the I4H tutorial RoomCamera."""
    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.12, 0.08, 0.70),
            rot=(0.0, 0.7071, -0.7071, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=width,
        height=height,
        update_period=1.0 / fps,
    )