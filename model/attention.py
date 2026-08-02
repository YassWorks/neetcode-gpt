import math
import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)
        K_transposed = K.transpose(1, 2)
        weights = torch.matmul(Q, K_transposed) / math.sqrt(self.attention_dim)
        mask = torch.tril(torch.ones((weights.shape[1], weights.shape[1])))
        weights = weights.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(weights, dim=2)
        return torch.round(torch.matmul(weights, V), decimals=4)
