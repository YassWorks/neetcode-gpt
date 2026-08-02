import torch
import torch.nn
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        N, M = to_reshape.shape[0], to_reshape.shape[1]
        return torch.round(to_reshape.reshape(M * N // 2, 2), decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        return torch.round(torch.mean(to_avg, axis=0), decimals=4)

    def concatenate(
        self, cat_one: TensorType[float], cat_two: TensorType[float]
    ) -> TensorType[float]:
        return torch.round(torch.cat((cat_one, cat_two), dim=1), decimals=4)

    def get_loss(
        self, prediction: TensorType[float], target: TensorType[float]
    ) -> TensorType[float]:
        return torch.round(torch.nn.functional.mse_loss(prediction, target), decimals=4)
