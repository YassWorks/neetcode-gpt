import numpy as np
from numpy.typing import NDArray


class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z_s = z - np.max(z)
        sum = np.sum(np.exp(z_s))
        arr = np.exp(z_s) / sum
        return np.round(arr, 4)
