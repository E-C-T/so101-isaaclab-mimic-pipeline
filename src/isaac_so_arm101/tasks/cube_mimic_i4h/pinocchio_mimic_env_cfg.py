from __future__ import annotations

import torch

from isaaclab.envs import DataGenConfig, MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaac_so_arm101.tasks.cube_mimic_i4h.mimic_mdp as mimic_mdp
from isaac_so_arm101.tasks.cube_replay_i4h.joint_pos_env_cfg import (
    SoArm101CubeReplayI4HEnvCfg_PLAY,
)


OBJECT_START_XY = (0.1873, 0.015)

# This is only the env-construction/default reset pose for Mimic generation.
# It must be inside the I4H USD joint limits. The exact HDF5 replay states can
# still be restored by reset_to during annotation/replay. Do not set this to a
# demo pose that Isaac Lab rejects during articulation validation.
I4H_MIMIC_INIT_JOINT_POS = {
    "shoulder_pan": -0.13564736,
    "shoulder_lift": -1.62364730,
    "elbow_flex": 1.68500000,
    "wrist_flex": 1.31176210,
    "wrist_roll": -1.75949300,
    "gripper": 0.03890861,
}


@configclass
class SubtaskTermsCfg(ObsGroup):
    """State-based subtask signals used by Isaac Mimic auto annotation/generation."""

    object_lifted = ObsTerm(
        func=mimic_mdp.object_lifted,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "min_height": 0.1,
        },
    )

    object_above_goal = ObsTerm(
        func=mimic_mdp.object_above_goal,
        params={
            "asset_name": "object",
        },
    )

    object_has_moved = ObsTerm(
        func=mimic_mdp.object_has_moved,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "start_xy": OBJECT_START_XY,
            "min_xy_distance": 0.04,
        },
    )

    object_in_goal = ObsTerm(
        func=mimic_mdp.object_in_goal,
        params={
            "asset_name": "object",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class SoArm101CubePinocchioMimicI4HEnvCfg(SoArm101CubeReplayI4HEnvCfg_PLAY, MimicEnvCfg):
    """SO-101 I4H USD Mimic environment using Pinocchio retargeting.

    Inherits the calibrated I4H replay environment:
      - I4H USD robot
      - calibrated robot root pose
      - cube/table/goal region
      - joint-position action interface

    The I4H-specific runtime behavior is implemented in cube_mimic_i4h/mimic_env.py.
    """

    def __post_init__(self):
        super().__post_init__()

        # self.goal_region = {
        #     "x_min": 0.025,
        #     "x_max": 0.175,
        #     "y_min": 0.125,
        #     "y_max": 0.275,
        #     "z_min": 0.0,
        #     "z_max": 0.10,
        #     "max_lin_vel": None,
        # }

        self.datagen_config = DataGenConfig(
            name="so101_cube_i4h_pinocchio_mimic",
            generation_num_trials=10,
            generation_keep_failed=False,
            generation_guarantee=True,
            max_num_failures=50,
            seed=1,
        )

        # Keep calibrated root pose from the replay parent cfg. Only set a
        # valid in-limit joint default for environment construction/generation.
        self.scene.robot.init_state.joint_pos = I4H_MIMIC_INIT_JOINT_POS

        self.subtask_configs = {
            "end_effector": [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="object_lifted",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 1},
                    subtask_term_offset_range=(0, 2),
                    action_noise=0.0,
                    num_interpolation_steps=1,
                    num_fixed_steps=2,
                    apply_noise_during_interpolation=False,
                    description="Grasp, close the gripper, and lift the cube.",
                    next_subtask_description="Carry the lifted cube above the goal region.",
                ),
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="object_above_goal",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 1},
                    subtask_term_offset_range=(0, 2),
                    action_noise=0.0,
                    num_interpolation_steps=1,
                    num_fixed_steps=2,
                    apply_noise_during_interpolation=False,
                    description="Carry the cube above the goal region while keeping the gripper closed.",
                    next_subtask_description="Lower the cube into the goal region.",
                ),
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="object_in_goal",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 1},
                    subtask_term_offset_range=(0, 0),
                    action_noise=0.0,
                    num_interpolation_steps=0,
                    num_fixed_steps=2,
                    apply_noise_during_interpolation=False,
                    description="Lower the cube into the goal region and release.",
                    next_subtask_description="Task complete.",
                ),
            ]
        }


        if hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = False
            self.observations.policy.concatenate_terms = False

        self.observations.subtask_terms = SubtaskTermsCfg()

        self.observations.subtask_terms.object_above_goal.params = {
            "asset_name": "object",

            "x_min": self.goal_region["x_min"],
            "x_max": self.goal_region["x_max"],

            # slightly deeper into bin before trigger
            "y_min": self.goal_region["y_min"] + 0.025,
            "y_max": self.goal_region["y_max"],

            "z_min": 0.04,
            "z_max": 0.25,

            "max_lin_vel": 1.0,
        }

        self.observations.subtask_terms.object_in_goal.params = {
            "asset_name": "object",
            **self.goal_region, **self.goal_region,
        }

        self.idle_action = torch.zeros(6)

        if hasattr(self.scene, "ee_frame"):
            self.scene.ee_frame.debug_vis = False

        if hasattr(self.terminations, "object_dropping"):
            self.terminations.object_dropping = None

# Backwards-compatible alias for older local scripts.
SoArm101CubeJointPosMimicI4HEnvCfg = SoArm101CubePinocchioMimicI4HEnvCfg