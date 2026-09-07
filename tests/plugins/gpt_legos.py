"""Test stand in for a GPT plugin: a tiny causal transformer over embedded tokens."""

import kalfa
import torch
from torch import nn


class TinyGPT(nn.Module):
    def __init__(self, d_model, layers, heads, seq_len):
        super().__init__()
        self.seq_len = int(seq_len)
        self.positions = nn.Embedding(self.seq_len, int(d_model))
        layer = nn.TransformerEncoderLayer(int(d_model), int(heads), dim_feedforward=2 * int(d_model),
                                           batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, int(layers))

    def forward(self, embedded):
        length = embedded.shape[1]
        positions = torch.arange(length, device=embedded.device)
        hidden = embedded + self.positions(positions)[None, :, :]
        mask = torch.triu(torch.ones(length, length, device=embedded.device, dtype=torch.bool), diagonal=1)
        return self.blocks(hidden, mask=mask)


@kalfa.lego("/layer/gpt/gpt", alias="gpt",
            description="Test GPT: position embeddings and a causal transformer encoder over embedded tokens; the "
                        "token embedding and the logits layer are separate nodes")
def gpt(d_model, layers, heads, seq_len):
    return TinyGPT(d_model, layers, heads, seq_len)
