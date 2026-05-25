import torch
from typing import List, Tuple


class Solution:

    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        n = len(tokens)

        starts = torch.randint(0, n - context_length, (batch_size,))
        X, Y = [], []

        for i in starts.tolist():
            x_seq = tokens[i:i + context_length]
            y_seq = tokens[i + 1:i + context_length + 1]

            X.append(x_seq)
            Y.append(y_seq)

        return X, Y
