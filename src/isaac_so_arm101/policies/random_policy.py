from __future__ import annotations

import numpy as np

from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_dim: int, low: float = -1.0, high: float = 1.0):
        self.action_dim = action_dim
        self.low = low
        self.high = high

    def reset(self) -> None:
        pass

    def act(self, observation):
        return np.random.uniform(self.low, self.high, size=(self.action_dim,)).astype(np.float32)