import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        rms = np.sqrt(np.mean(x**2) + eps)
        x_hat = x / rms
        return np.round(gamma * x_hat, 4)
