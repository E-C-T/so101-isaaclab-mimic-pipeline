from __future__ import annotations

import numpy as np


def identity_action(action: np.ndarray) -> np.ndarray:
    return action


def joint_position_action(action: np.ndarray) -> np.ndarray:
    return action.astype(np.float32)


def split_arm_and_gripper(action: np.ndarray, gripper_index: int = -1):
    arm = action[:gripper_index] if gripper_index != -1 else action[:-1]
    gripper = float(action[gripper_index])
    return arm.astype(np.float32), gripper