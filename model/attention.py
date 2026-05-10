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
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K =  self.key(embedded)
        Q =  self.query(embedded)
        V =  self.value(embedded)
        K_transposed = K.transpose(1, 2)
        weights = torch.matmul(Q, K_transposed) / math.sqrt(self.attention_dim)
        mask = torch.tril(torch.ones((weights.shape[1], weights.shape[1])))
        weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(weights, dim=2)
        return torch.round(torch.matmul(weights, V), decimals=4)
