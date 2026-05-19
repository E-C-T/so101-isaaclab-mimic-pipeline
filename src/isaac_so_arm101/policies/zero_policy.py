from __future__ import annotations

import numpy as np

from .base_policy import BasePolicy


class ZeroPolicy(BasePolicy):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def reset(self) -> None:
        pass

    def act(self, observation):
        return np.zeros(self.action_dim, dtype=np.float32)