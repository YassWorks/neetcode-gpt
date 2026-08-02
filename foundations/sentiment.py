import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocabulary_size, 16)
        self.net = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embeddings = self.embed(x)
        sentence_embedding = embeddings.mean(dim=1)
        output = self.net(sentence_embedding)
        return torch.round(output, decimals=4)
