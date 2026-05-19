import gymnasium as gym

gym.register(
    id="Isaac-SO-ARM101-Cube-I4H-Replay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_so_arm101.tasks.cube_replay_i4h.joint_pos_env_cfg:"
            "SoArm101CubeReplayI4HEnvCfg_PLAY"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Cube-I4H-Replay-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_so_arm101.tasks.cube_replay_i4h.joint_pos_env_cfg_camera:"
            "SoArm101CubeReplayI4HCameraEnvCfg_PLAY"
        ),
    },
    disable_env_checker=True,
)