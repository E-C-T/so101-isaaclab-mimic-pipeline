from __future__ import annotations

from typing import Any, Dict

from .base_policy import BasePolicy


class Pi0PolicyWrapper(BasePolicy):
    def __init__(self, model: Any):
        self.model = model

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    def act(self, observation: Dict[str, Any]) -> Any:
        raise NotImplementedError("Implement pi0 policy inference adapter.")