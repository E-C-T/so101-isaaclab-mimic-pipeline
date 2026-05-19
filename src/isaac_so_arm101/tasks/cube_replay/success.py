from __future__ import annotations

import torch


def object_in_aabb_success(
    env,
    asset_name: str = "object",
    x_min: float = 0.10,
    x_max: float = 0.20,
    y_min: float = 0.10,
    y_max: float = 0.20,
    z_min: float = 0.00,
    z_max: float = 0.08,
    max_lin_vel: float | None = None,
    debug_print: bool = False,
) -> torch.Tensor:
    """
    Success if object root position is inside an axis-aligned world-frame box.

    Returns:
        Bool tensor of shape [num_envs]
    """
    obj = env.scene[asset_name]
    root_state = obj.data.root_state_w  # shape [num_envs, 13]

    pos = root_state[:, 0:3]
    quat = root_state[:, 3:7]
    lin_vel = root_state[:, 7:10]

    inside = (
        (pos[:, 0] >= x_min)
        & (pos[:, 0] <= x_max)
        & (pos[:, 1] >= y_min)
        & (pos[:, 1] <= y_max)
        & (pos[:, 2] >= z_min)
        & (pos[:, 2] <= z_max)
    )

    if max_lin_vel is not None:
        speed = torch.linalg.norm(lin_vel, dim=1)
        inside = inside & (speed <= max_lin_vel)
    else:
        speed = torch.linalg.norm(lin_vel, dim=1)

    if debug_print:
        print("\n[success.py] object_in_aabb_success debug:")
        print(f"  asset_name = {asset_name}")
        print(f"  root_state_w.shape = {tuple(root_state.shape)}")
        print(f"  pos[0] = {pos[0].detach().cpu().numpy()}")
        print(f"  quat[0] = {quat[0].detach().cpu().numpy()}")
        print(f"  lin_vel[0] = {lin_vel[0].detach().cpu().numpy()}")
        print(f"  speed[0] = {float(speed[0].detach().cpu())}")
        print("  bounds:")
        print(f"    x_min = {x_min}")
        print(f"    x_max = {x_max}")
        print(f"    y_min = {y_min}")
        print(f"    y_max = {y_max}")
        print(f"    z_min = {z_min}")
        print(f"    z_max = {z_max}")
        print(f"    max_lin_vel = {max_lin_vel}")
        print(f"  inside[0] = {bool(inside[0].detach().cpu())}")

    return inside