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
        start_indices = torch.randint(
            0,
            max_start + 1,
            (batch_size,)
        )

        X = []
        Y = []

        for i in start_indices:
            i = i.item()
            X.append(data[i : i + context_length])
            Y.append(data[i + 1 : i + context_length + 1])

        return torch.stack(X), torch.stack(Y)