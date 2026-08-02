import torch
import torch.nn as nn
from typing import List, Dict


class Solution:
    def compute_activation_stats(
        self, model: nn.Module, x: torch.Tensor
    ) -> List[Dict[str, float]]:
        result = []
        with torch.no_grad():
            h = x
            for layer in model.children():
                h = layer(h)
                if isinstance(layer, torch.nn.Linear):
                    layer_info = {
                        "mean": torch.round(h.mean(), decimals=4).item(),
                        "std": torch.round(h.std(), decimals=4).item(),
                        "dead_fraction": torch.round(
                            (h <= 0).all(dim=0).float().mean(), decimals=4
                        ).item(),
                    }
                    result.append(layer_info)
        return result

    def compute_gradient_stats(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> List[Dict[str, float]]:
        result = []
        model.zero_grad()
        y_hat = model(x)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        loss.backward()
        for layer in model.children():
            if isinstance(layer, torch.nn.Linear):
                grad = layer.weight.grad
                mean = torch.round(grad.mean(), decimals=4).item()
                std = torch.round(grad.std(), decimals=4).item()
                norm = torch.round(torch.norm(grad), decimals=4).item()
                layer_info = {"mean": mean, "std": std, "norm": norm}
                result.append(layer_info)
        return result

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]],
    ) -> str:
        if any(
            k == "dead_fraction" and v > 0.5
            for d in activation_stats
            for k, v in d.items()
        ):
            return "dead_neurons"

        if any(k == "norm" and v > 1000 for d in gradient_stats for k, v in d.items()):
            return "exploding_gradients"

        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        if any(k == "std" and v < 0.1 for d in activation_stats for k, v in d.items()):
            return "vanishing_gradients"

        if any(k == "std" and v > 10.0 for d in activation_stats for k, v in d.items()):
            return "exploding_gradients"

        return "healthy"
