from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

import isaac_so_arm101.tasks.cube_replay.success as replay_success


def object_lifted(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.04,
) -> torch.Tensor:
    """Return True when the object has been lifted above the table.

    This is a simple heuristic subtask signal for Isaac Mimic auto annotation.
    For your DexCube setup, the object rests near z ~= 0.01 and reaches around
    z ~= 0.05 to 0.06 during successful transport, so 0.04 is a reasonable
    first threshold.
    """
    obj = env.scene[asset_cfg.name]
    root_state = obj.data.root_state_w
    object_z = root_state[:, 2]
    return object_z >= min_height


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
    """Return True when the object is inside the configured goal AABB."""
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
    """Return True once the object has moved away from its initial XY location.

    This is optional, but useful as an intermediate subtask signal if the
    lift threshold is too strict for a sliding/pushing demo.
    """
    obj = env.scene[asset_cfg.name]
    root_state = obj.data.root_state_w
    object_xy = root_state[:, 0:2]
    start_xy_tensor = torch.tensor(start_xy, device=object_xy.device, dtype=object_xy.dtype)
    dist = torch.linalg.norm(object_xy - start_xy_tensor, dim=1)
    return dist >= min_xy_distance
