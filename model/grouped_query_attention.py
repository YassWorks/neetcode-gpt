import torch
import torch.nn as nn
import math
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):

    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        Q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)
        V = V.repeat_interleave(repeat_factor, dim=1)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        attn = weights @ V
        attn = attn.transpose(1, 2)
        attn = attn.contiguous().reshape(B, T, self.num_heads * self.head_dim)

        output = self.output_proj(attn)
        return torch.round(output, decimals=4)