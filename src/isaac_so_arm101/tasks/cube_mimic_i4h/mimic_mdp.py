from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

import isaac_so_arm101.tasks.cube_replay_i4h.success as replay_success


def object_lifted(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.04,
) -> torch.Tensor:
    obj = env.scene[asset_cfg.name]
    root_state = obj.data.root_state_w
    object_z = root_state[:, 2]
    return object_z >= min_height


def object_above_goal(
    env,
    asset_name: str = "object",
    x_min: float = 0.075,
    x_max: float = 0.225,
    y_min: float = 0.175,
    y_max: float = 0.325,
    z_min: float = 0.04,
    z_max: float = 0.25,
    max_lin_vel: float | None = 1.0,
) -> torch.Tensor:
    """Return True when the object is lifted above the goal XY region.

    This is used as the middle Mimic phase boundary:
        grasp/lift -> carry above goal -> lower/place.
    """

    obj = env.scene[asset_name]
    root_state = obj.data.root_state_w

    pos_w = root_state[:, 0:3]
    lin_vel_w = root_state[:, 7:10]

    in_x = (pos_w[:, 0] >= x_min) & (pos_w[:, 0] <= x_max)
    in_y = (pos_w[:, 1] >= y_min) & (pos_w[:, 1] <= y_max)
    in_z = (pos_w[:, 2] >= z_min) & (pos_w[:, 2] <= z_max)

    if max_lin_vel is None:
        slow_enough = torch.ones_like(in_x, dtype=torch.bool)
    else:
        slow_enough = torch.linalg.norm(lin_vel_w, dim=-1) <= max_lin_vel

    return in_x & in_y & in_z & slow_enough


def object_in_goal(
    env,
    asset_name: str = "object",
    x_min: float = 0.075,
    x_max: float = 0.225,
    y_min: float = 0.175,
    y_max: float = 0.325,
    z_min: float = 0.0,
    z_max: float = 0.15,
    max_lin_vel: float | None = 0.15,
) -> torch.Tensor:
    return replay_success.object_in_aabb_success(
        env,
        asset_name=asset_name,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        max_lin_vel=max_lin_vel,
        debug_print=False,
    )


def object_has_moved(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    start_xy: tuple[float, float] = (0.2275, 0.015),
    min_xy_distance: float = 0.04,
) -> torch.Tensor:
    obj = env.scene[asset_cfg.name]
    root_state = obj.data.root_state_w
    object_xy = root_state[:, 0:2]
    start_xy_tensor = torch.tensor(start_xy, device=object_xy.device, dtype=object_xy.dtype)
    dist = torch.linalg.norm(object_xy - start_xy_tensor, dim=1)
    return dist >= min_xy_distance