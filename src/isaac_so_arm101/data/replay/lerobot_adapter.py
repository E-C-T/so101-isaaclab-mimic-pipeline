from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np


@dataclass
class ReplayStep:
    index: int
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    gripper: Optional[float] = None
    images: Optional[Dict[str, Any]] = None
    task_text: Optional[str] = None
    timestamp: Optional[float] = None


class LeRobotEpisodeAdapter:
    """Adapter for loading LeRobot episodes into a canonical replay format."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)

    def load_episode(self, episode_id: int) -> Dict[str, Any]:
        raise NotImplementedError("Implement LeRobot episode loading here.")

    def iter_episode_steps(self, episode: Dict[str, Any]) -> Iterator[ReplayStep]:
        raise NotImplementedError("Implement LeRobot step iteration here.")