import omni
from pxr import Gf

def get_prim_world_matrix(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")
    return omni.usd.get_world_transform_matrix(prim)

def get_world_xyz(prim_path: str):
    M = get_prim_world_matrix(prim_path)
    t = M.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])

def transform_local_point_to_world(prim_path: str, local_xyz):
    M = get_prim_world_matrix(prim_path)
    p_local = Gf.Vec3d(*local_xyz)
    p_world = M.Transform(p_local)
    return float(p_world[0]), float(p_world[1]), float(p_world[2])

cube_path = "/World/envs/env_0/Object"
gripper_link_path = "/World/envs/env_0/Robot/gripper_link"

cube_xyz = get_world_xyz(cube_path)

# This matches your Isaac FrameTransformer offset:
# end_effector = gripper_link local offset [0.01, 0.0, -0.09]
eef_xyz = transform_local_point_to_world(
    gripper_link_path,
    (0.01, 0.0, -0.09),
)

gripper_link_xyz = get_world_xyz(gripper_link_path)

print("Cube center:")
print(f"x = {cube_xyz[0]:.4f}")
print(f"y = {cube_xyz[1]:.4f}")
print(f"z = {cube_xyz[2]:.4f}")

print("\nGripper link origin:")
print(f"x = {gripper_link_xyz[0]:.4f}")
print(f"y = {gripper_link_xyz[1]:.4f}")
print(f"z = {gripper_link_xyz[2]:.4f}")

print("\nMimic/IK end_effector = gripper_link + [0.01, 0.0, -0.09]:")
print(f"x = {eef_xyz[0]:.4f}")
print(f"y = {eef_xyz[1]:.4f}")
print(f"z = {eef_xyz[2]:.4f}")

print("\nCube minus Mimic/IK end_effector:")
print(f"dx = {cube_xyz[0] - eef_xyz[0]:.4f}")
print(f"dy = {cube_xyz[1] - eef_xyz[1]:.4f}")
print(f"dz = {cube_xyz[2] - eef_xyz[2]:.4f}")