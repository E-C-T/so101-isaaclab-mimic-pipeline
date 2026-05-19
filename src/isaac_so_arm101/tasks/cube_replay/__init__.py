import gymnasium as gym

gym.register(
    id="Isaac-SO-ARM101-Cube-Replay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaac_so_arm101.tasks.cube_replay.joint_pos_env_cfg:SoArm101CubeReplayEnvCfg_PLAY",
    },
    disable_env_checker=True,
)