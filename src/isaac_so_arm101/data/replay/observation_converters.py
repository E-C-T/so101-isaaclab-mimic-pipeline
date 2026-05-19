from __future__ import annotations

from typing import Any, Dict


def to_policy_observation(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "joint_positions": raw_obs.get("joint_positions"),
        "joint_velocities": raw_obs.get("joint_velocities"),
        "gripper_position": raw_obs.get("gripper_position"),
        "images": raw_obs.get("images", {}),
        "task_text": raw_obs.get("task_text"),
        "ee_pose": raw_obs.get("ee_pose"),
    }