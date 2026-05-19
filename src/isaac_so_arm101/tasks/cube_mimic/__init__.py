import gymnasium as gym


gym.register(
    id="Isaac-SO-ARM101-Cube-Joint-Pos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_so_arm101.tasks.cube_mimic.joint_pos_mimic_env_cfg:"
            "SoArm101CubeJointPosMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Cube-Joint-Pos-Mimic-v0",
    entry_point="isaac_so_arm101.tasks.cube_mimic.mimic_env:SoArm101CubeJointPosMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_so_arm101.tasks.cube_mimic.joint_pos_mimic_env_cfg:"
            "SoArm101CubeJointPosMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)