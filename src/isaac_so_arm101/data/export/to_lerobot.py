from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def export_episode_to_lerobot(output_dir: str | Path, episode: Dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Implement LeRobot export here.")