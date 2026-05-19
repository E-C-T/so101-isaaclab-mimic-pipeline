from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import DataGenConfig, MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaac_so_arm101.tasks.cube_mimic_i4h.mimic_mdp as mimic_mdp

from isaac_so_arm101.tasks.cube_replay_i4h.joint_pos_env_cfg import (
    SoArm101CubeReplayI4HEnvCfg_PLAY,
)


OBJECT_START_XY = (0.1873, 0.015)

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
    """Subtask terms for phase-based Mimic generation."""

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
            "x_min": 0.075,
            "x_max": 0.225,
            "y_min": 0.225,
            "y_max": 0.325,
            "z_min": 0.075,
            "z_max": 0.25,
            "max_lin_vel": None,
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
            "z_max": 0.075,
            "max_lin_vel": None,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class SoArm101CubeDiffIKMimicI4HEnvCfg(
    SoArm101CubeReplayI4HEnvCfg_PLAY,
    MimicEnvCfg,
):
    """SO101 I4H Mimic env using USD-based Diff IK for Mimic generation."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config = DataGenConfig(
            name="so101_cube_i4h_diff_ik_mimic",
            generation_num_trials=10,
            generation_keep_failed=False,
            generation_guarantee=True,
            max_num_failures=50,
            seed=1,
        )

        # Stable known initial pose.
        self.scene.robot.init_state.joint_pos = I4H_MIMIC_INIT_JOINT_POS

        # ---------------------------------------------------------------------
        # USD-based Differential IK arm action.
        # ---------------------------------------------------------------------
        #
        # Use position-only first. SO101 has 5 arm joints, so full 6D pose IK is
        # overconstrained and can introduce wrist/branch weirdness.
        #
         # I4H USD body names include: base, shoulder, upper_arm, lower_arm, wrist, gripper, moving_jaw_so101_v1.
        # Use the stable gripper body for the IK body, not the moving jaw.
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            body_name="gripper",
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.01, 0.0, -0.09),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            scale=1.0,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={
                    "lambda_val": 0.01,
                },
            ),
        )

        # Gripper remains direct joint position command.
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
            use_default_offset=False,
        )

        # ---------------------------------------------------------------------
        # Mimic phases:
        #   1. grasp/lift
        #   2. carry above goal
        #   3. lower/place/release
        # ---------------------------------------------------------------------
        self.subtask_configs = {
            "end_effector": [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="object_lifted",
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 1},
                    subtask_term_offset_range=(0, 2),
                    action_noise=0.0,
                    num_interpolation_steps=4,
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
                    num_interpolation_steps=4,
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
                    num_interpolation_steps=2,
                    num_fixed_steps=1,
                    apply_noise_during_interpolation=False,
                    description="Lower the cube into the goal region and release.",
                    next_subtask_description="Task complete.",
                ),
            ]
        }

        # Observations.
        if hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = False
            self.observations.policy.concatenate_terms = False

        self.observations.subtask_terms = SubtaskTermsCfg()

        # Keep in sync with replay cfg goal region if available.
        if hasattr(self, "goal_region"):
            self.observations.subtask_terms.object_above_goal.params = {
                "asset_name": "object",
                **self.goal_region,
                "z_min": 0.04,
                "z_max": 0.25,
                "max_lin_vel": 1.0,
            }
            self.observations.subtask_terms.object_in_goal.params = {
                "asset_name": "object",
                **self.goal_region,
            }

        # DiffIK pose action is 7D (xyz + wxyz quaternion), gripper is 1D.
        self.idle_action = torch.zeros(8)

        # Optional debug flags used by the env class below.
        self.debug_mimic_signals = True
        self.debug_diffik_frame_conversion = True
        
        # Gripper gating / carry clamp.
        # The gripper is always the last action dimension, so this works for pose-mode
        # DiffIK actions [x, y, z, qw, qx, qy, qz, gripper].
        self.hold_gripper_during_carry = False
        self.open_gripper_value = 0.0
        self.closed_gripper_value = 0.03890861
        self.object_lifted_min_height = 0.04
        self.above_goal_z_min = 0.05
        self.above_goal_z_max = 0.12
        self.above_goal_max_lin_vel = 0.35

        if hasattr(self.scene, "ee_frame"):
            self.scene.ee_frame.debug_vis = False

        if hasattr(self.terminations, "object_dropping"):
            self.terminations.object_dropping = None
