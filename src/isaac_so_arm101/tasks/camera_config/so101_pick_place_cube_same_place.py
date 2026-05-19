from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from scipy.spatial.transform import Rotation as R


def quat_wxyz_from_euler_xyz_deg(roll: float, pitch: float, yaw: float):
    q_xyzw = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_quat()
    x, y, z, w = q_xyzw
    return (float(w), float(x), float(y), float(z))


def make_so101_up_camera_cfg() -> CameraCfg:
    up_cam_rot = quat_wxyz_from_euler_xyz_deg(
        roll=180.0,
        pitch=150.0,
        yaw=35.0,
    )

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/UpCamera",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=0.8,
            horizontal_aperture=3.68,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.60, 0.4, 0.35),
            rot=up_cam_rot,
            convention="world",
        ),
    )


def make_so101_wrist_camera_cfg() -> CameraCfg:
    wrist_cam_rot = quat_wxyz_from_euler_xyz_deg(
        roll=-15.0,
        pitch=180.0,
        yaw=0.0,
    )

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/WristCamera",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            horizontal_aperture=3.68,
            focus_distance=0.8,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.003, -0.055, -0.00),
            rot=wrist_cam_rot,
            convention="ros",
        ),
    )