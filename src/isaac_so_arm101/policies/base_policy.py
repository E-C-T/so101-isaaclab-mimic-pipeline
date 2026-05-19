from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePolicy(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Any:
        pass