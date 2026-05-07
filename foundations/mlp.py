import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def ReLU(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(x, 0)
    
    def forward(self, 
                x: NDArray[np.float64], 
                weights: List[NDArray[np.float64]], 
                biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        a = x
        for i in range(len(weights)):
            a = self.ReLU(a @ weights[i] + biases[i])
        return np.round(a, 5)
