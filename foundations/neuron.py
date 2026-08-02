import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str
    ) -> float:
        z = x @ w + b
        if activation == "sigmoid":
            return np.round(1 / (1 + np.exp(-z)), 5)
        else:
            return np.round(np.maximum(z, 0), 5)
