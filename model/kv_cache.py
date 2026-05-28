import torch
import torch.nn as nn
from typing import Tuple, Optional


class KVCache:

    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if (self.cache_k == None or self.cache_v == None):
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.concat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.concat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):

    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        if (kv_cache == None):
            kv_cache = KVCache()
        cache_k, cache_v = kv_cache.update(K, V)
        d_k = Q.size(-1)
        attn = torch.softmax((Q @ cache_k.transpose(-2, -1)) / math.sqrt(d_k), dim=-1) @ cache_v
        return torch.round(attn, decimals=4), kv_cache
