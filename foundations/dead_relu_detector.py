import torch
import torch.nn as nn
from typing import List


class Solution:
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        results = []
        with torch.no_grad():
            h = x
            for layer in model.children():
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    dead_fraction = torch.round(
                        (h == 0).all(dim=0).float().mean(), decimals=4
                    ).item()
                    results.append(dead_fraction)
        return results

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if any(v > 0.5 for v in dead_fractions):
            return "use_leaky_relu"

        if dead_fractions[0] > 0.3:
            return "reinitialize"

        if (
            all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        return "healthy"
