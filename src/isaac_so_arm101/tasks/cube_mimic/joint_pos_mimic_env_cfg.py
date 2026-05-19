from __future__ import annotations

import torch

from isaaclab.envs import MimicEnvCfg, SubTaskConfig, DataGenConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaac_so_arm101.tasks.cube_mimic.mimic_mdp as mimic_mdp
from isaac_so_arm101.tasks.cube_replay.joint_pos_env_cfg import SoArm101CubeReplayEnvCfg_PLAY

from isaac_so_arm101.tasks.camera_config.so101_pick_place_cube_same_place import (
    make_so101_up_camera_cfg,
    make_so101_wrist_camera_cfg,
)

@configclass
class SubtaskTermsCfg(ObsGroup):
    """Subtask signals used by Isaac Mimic auto annotation.

    For the first SO-101 cube pick-place Mimic pass, use simple state-based
    boolean signals:
      1. object_lifted
      2. object_has_moved
      3. object_in_goal

    The goal is not perfect semantics yet. The goal is to expose standard
    subtask_terms so annotate_demos.py has a structured signal source.
    """

    object_lifted = ObsTerm(
        func=mimic_mdp.object_lifted,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "min_height": 0.04,
        },
    )

    object_has_moved = ObsTerm(
        func=mimic_mdp.object_has_moved,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "start_xy": (0.2275, 0.015),
            "min_xy_distance": 0.04,
        },
    )

    object_in_goal = ObsTerm(
        func=mimic_mdp.object_in_goal,
        params={
            "asset_name": "object",
            "x_min": 0.075,
            "x_max": 0.225,
            "y_min": 0.175,
            "y_max": 0.325,
            "z_min": 0.0,
            "z_max": 0.15,
            "max_lin_vel": 0.15,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class SoArm101CubeJointPosMimicEnvCfg(SoArm101CubeReplayEnvCfg_PLAY, MimicEnvCfg):
    """State-based SO-101 cube Mimic environment.

    This intentionally keeps the same absolute joint-position action interface
    as the currently replayable dataset. Do not switch to IK-relative actions
    until you either recollect demos or convert actions into the new action
    representation.
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config = DataGenConfig(
            name="so101_cube_joint_pos_mimic",
            generation_num_trials=10,
            generation_keep_failed=False,
            generation_guarantee=True,
            max_num_failures=50,
            seed=1,
        )

        # IMPORTANT:
        # The successful HDF5 demos reset the SO-101 root orientation to [1, 0, 0, 0]
        # in wxyz quaternion order. Mimic generation uses env.reset(), not demo reset_to(),
        # so the cfg default root rotation must match the successful demo convention.
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            "shoulder_pan": -0.0316,
            "shoulder_lift": -1.6728,
            "elbow_flex": 1.6850,
            "wrist_flex": 1.3147,
            "wrist_roll": 1.5534,
            "gripper": 0.0389,
        }

        # Ensure mimic generation uses the same camera views as replay/calibration.
        self.scene.up_camera = make_so101_up_camera_cfg()
        self.scene.wrist_camera = make_so101_wrist_camera_cfg()

        self.subtask_configs = {
            "end_effector": [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="object_lifted",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 1},
                    subtask_term_offset_range=(0, 2),
                    action_noise=0.0,
                    num_interpolation_steps=15,
                    num_fixed_steps=20,
                    apply_noise_during_interpolation=False,
                    description="Grasp and lift the cube.",
                    next_subtask_description="Move the cube to the goal region.",
                ),
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="object_in_goal",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 1},
                    subtask_term_offset_range=(0, 0),
                    action_noise=0.0,
                    num_interpolation_steps=10,
                    num_fixed_steps=25,
                    apply_noise_during_interpolation=False,
                    description="Place the cube in the goal region.",
                    next_subtask_description="Task complete.",
                ),
            ]
        }

        # self.subtask_configs = {
        #     "end_effector": [
        #         SubTaskConfig(
        #             object_ref="object",
        #             subtask_term_signal="object_lifted",
        #             selection_strategy="random",
        #             subtask_term_offset_range=(0, 5),
        #             action_noise=0.01,
        #             num_interpolation_steps=5,
        #             num_fixed_steps=0,
        #             apply_noise_during_interpolation=False,
        #             description="Lift or transport the object.",
        #             next_subtask_description="Place the object inside the goal region.",
        #         ),
        #         SubTaskConfig(
        #             object_ref="object",
        #             subtask_term_signal="object_in_goal",
        #             selection_strategy="random",
        #             subtask_term_offset_range=(0, 0),
        #             action_noise=0.001,
        #             num_interpolation_steps=5,
        #             num_fixed_steps=5,
        #             apply_noise_during_interpolation=False,
        #             description="Place the object inside the goal region.",
        #             next_subtask_description="Task complete.",
        #         ),
        #     ]
        # }

        # self.subtask_configs = {
        #     "end_effector": [
        #         SubTaskConfig(
        #             object_ref="object",
        #             subtask_term_signal="object_in_goal",
        #             selection_strategy="random",
        #             subtask_term_offset_range=(0, 0),
        #             action_noise=0.0,
        #             num_interpolation_steps=0,
        #             num_fixed_steps=0,
        #             apply_noise_during_interpolation=False,
        #             description="Move the object into the goal region.",
        #             next_subtask_description="Task complete.",
        #         ),
        #     ]
        # }

        # Keep the policy observation group dictionary-like for imitation learning
        # tooling, matching the style used by Isaac Lab Mimic examples.
        if hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = False
            self.observations.policy.concatenate_terms = False

        # Add the standard Mimic subtask signal group.
        self.observations.subtask_terms = SubtaskTermsCfg()

        # Ensure subtask goal params exactly match this env's goal_region.
        self.observations.subtask_terms.object_in_goal.params = {
            "asset_name": "object",
            **self.goal_region,
        }

        # This is a safe idle action for the absolute joint-position interface.
        # Your action space is currently 5 arm joints + 1 gripper joint.
        self.idle_action = torch.zeros(6)

        # Keep the EE frame available for future Mimic/IK/visuomotor work, but do
        # not draw it during headless dataset generation.
        if hasattr(self.scene, "ee_frame"):
            self.scene.ee_frame.debug_vis = False



        # Keep success as a real termination term for Isaac Lab/Mimic tooling.
        # Do not set self.terminations = {} here.
        # The parent replay cfg already defines:
        #   self.terminations.success = DoneTerm(...)
        #
        # If inherited timeout/drop terms cause premature stopping during Mimic
        # generation, disable them explicitly here without deleting success.
        if hasattr(self.terminations, "object_dropping"):
            self.terminations.object_dropping = None
