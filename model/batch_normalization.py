import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, 
                   x: List[List[float]],
                   gamma: List[float],
                   beta: List[float],
                   running_mean: List[float],
                   running_var: List[float],
                   momentum: float,
                   eps: float,
                   training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            m=momentum
            running_mean = (1-m) * running_mean + m * mean
            running_var = (1-m) * running_var + m * var
        else:
            mean = running_mean
            var = running_var
        
        x_hat = (x - mean) / np.sqrt(var + eps)
        y = gamma * x_hat + beta
        return (np.round(y,4), np.round(running_mean,4), np.round(running_var,4))
