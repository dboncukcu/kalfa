import torch
from torch import nn
from torch.nn import functional


class CausalAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not a multiple of heads {heads}")
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        mixed = functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                                        is_causal=True)
        return self.out(mixed.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = CausalAttention(dim, heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, layers=6, dim=384, heads=6, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.tokens = nn.Embedding(vocab_size, dim)
        self.positions = nn.Embedding(self.seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.drop(self.tokens(input_ids) + self.positions(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
