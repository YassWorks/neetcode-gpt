import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(
        self, positive: List[str], negative: List[str]
    ) -> TensorType[float]:
        words = [word for text in positive + negative for word in text.split(" ")]
        words = sorted(list(set(words)))
        vocab = {word: i + 1 for i, word in enumerate(words)}

        encoded = [
            torch.tensor([vocab[w] for w in sent.split()], dtype=torch.long)
            for sent in positive + negative
        ]

        encoded = torch.nn.utils.rnn.pad_sequence(
            encoded, batch_first=True, padding_value=0
        )

        return encoded
