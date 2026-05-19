import gymnasium as gym


# ---------------------------------------------------------------------------
# Basic replay / joint-position control on the I4H USD model.
# ---------------------------------------------------------------------------
gym.register(
    id="Isaac-SO-ARM101-Cube-I4H-Joint-Pos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_so_arm101.tasks.cube_replay_i4h.joint_pos_env_cfg:"
            "SoArm101CubeReplayI4HEnvCfg_PLAY"
        ),
    },
    disable_env_checker=True,
)


# ---------------------------------------------------------------------------
# Isaac Mimic using Pinocchio retargeting.
#
# Use this for annotating existing 6D joint-action demonstrations:
#   [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# ---------------------------------------------------------------------------
gym.register(
    id="Isaac-SO-ARM101-Cube-I4H-Pinocchio-Mimic-v0",
    entry_point=(
        "isaac_so_arm101.tasks.cube_mimic_i4h.pinocchio_mimic_env:"
        "SoArm101CubePinocchioMimicI4HEnv"
    ),
    kwargs={
        "env_cfg_entry_point": (
            "isaac_so_arm101.tasks.cube_mimic_i4h.pinocchio_mimic_env_cfg:"
            "SoArm101CubePinocchioMimicI4HEnvCfg"
        ),
    },
    disable_env_checker=True,
)


# ---------------------------------------------------------------------------
# Isaac Mimic using Isaac Lab USD-based Differential IK.
#
# Use this for generating new 4D Diff IK demonstrations:
#   [target_eef_x, target_eef_y, target_eef_z, gripper]
# ---------------------------------------------------------------------------
gym.register(
    id="Isaac-SO-ARM101-Cube-I4H-Diff-IK-Mimic-v0",
    entry_point=(
        "isaac_so_arm101.tasks.cube_mimic_i4h.diff_ik_mimic_env:"
        "SoArm101CubeDiffIKMimicI4HEnv"
    ),
    kwargs={
        "env_cfg_entry_point": (
            "isaac_so_arm101.tasks.cube_mimic_i4h.diff_ik_mimic_env_cfg:"
            "SoArm101CubeDiffIKMimicI4HEnvCfg"
        ),
    },
    disable_env_checker=True,
)
