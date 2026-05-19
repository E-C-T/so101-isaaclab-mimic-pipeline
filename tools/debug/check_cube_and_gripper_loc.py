import omni

def get_world_xyz(prim_path: str):
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    world_xform = omni.usd.get_world_transform_matrix(prim)
    t = world_xform.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])

# Cube position
cube_path = "/World/envs/env_0/Object"
cube_x, cube_y, cube_z = get_world_xyz(cube_path)

print("Cube center:")
print(f"x = {cube_x:.4f}")
print(f"y = {cube_y:.4f}")
print(f"z = {cube_z:.4f}")

# Gripper/tool frame position
gripper_path = "/World/envs/env_0/Robot/gripper_link/gripper_frame_link"
grip_x, grip_y, grip_z = get_world_xyz(gripper_path)

print("\nGripper frame position:")
print(f"x = {grip_x:.4f}")
print(f"y = {grip_y:.4f}")
print(f"z = {grip_z:.4f}")

print("\nCube minus gripper:")
print(f"dx = {cube_x - grip_x:.4f}")
print(f"dy = {cube_y - grip_y:.4f}")
print(f"dz = {cube_z - grip_z:.4f}")

print("\nSuggested first goal region:")
print(f"x_min = {cube_x - 0.04:.4f}")
print(f"x_max = {cube_x + 0.04:.4f}")
print(f"y_min = {cube_y - 0.04:.4f}")
print(f"y_max = {cube_y + 0.04:.4f}")
print(f"z_min = {max(0.0, cube_z - 0.04):.4f}")
print(f"z_max = {cube_z + 0.04:.4f}")