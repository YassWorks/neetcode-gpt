import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1+np.exp(-x))

    def backward(self, 
                 x: NDArray[np.float64], 
                 w: NDArray[np.float64], 
                 b: float, 
                 y_true: float) -> Tuple[NDArray[np.float64], float]:
        y_hat = self.sigmoid(x @ w + b)
        grad_w = (y_hat - y_true) * y_hat * (1 - y_hat) * x
        grad_b = (y_hat - y_true) * y_hat * (1 - y_hat)
        return (np.round(grad_w, 5), np.round(grad_b, 5))
