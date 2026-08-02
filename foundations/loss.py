import numpy as np
from numpy.typing import NDArray


class Solution:
    def binary_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        n = len(y_true)
        return np.round(
            -1
            / n
            * np.sum(
                y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred)
            ),
            4,
        )

    def categorical_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        n = len(y_true)
        ans = -1 / n * np.sum(np.sum(y_true * np.log(y_pred + 1e-7), axis=1))
        return np.round(ans, 4)
