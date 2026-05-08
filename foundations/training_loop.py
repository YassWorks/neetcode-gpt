import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self,
              X: NDArray[np.float64],
              y: NDArray[np.float64],
              epochs: int,
              lr: float) -> Tuple[NDArray[np.float64], float]:
        w = np.zeros((X.shape[1], 1))
        b = 0
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            y_hat = X @ w + b
            dL_dw = (2/X.shape[0]) * X.T @ (y_hat - y)
            dL_db = 2 * np.mean(y_hat - y)
            w = w - lr * dL_dw
            b = b - lr * dL_db
        return (np.round(w.flatten(), 5), np.round(b, 5))
