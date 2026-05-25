import torch
from torchtyping import TensorType
from typing import Tuple


class Solution:

    def create_batches(
        self,
        data: TensorType[int],
        context_length: int,
        batch_size: int
    ) -> Tuple[TensorType[int], TensorType[int]]:

        torch.manual_seed(0)

        max_start = len(data) - context_length - 1
        starts = torch.randint(0, max_start+1, (batch_size,))
        offsets = torch.arange(context_length)
        
        X_idx = starts[:, None] + offsets[None, :]
        Y_idx = X_idx + 1

        X = data[X_idx]
        Y = data[Y_idx]

        return X, Y
