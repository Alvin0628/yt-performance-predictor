"""Tabular feature tokenizer: 1 continuous feature = 1 token (FT-Transformer style)."""
import torch
import torch.nn as nn


class TabularTokenizer(nn.Module):
    """x_i * w_i + b_i → d-dimensional token per continuous feature.
    Genre → embedding table (index 0 = <unk>).

    Init scaling: weight std 0.5, bias std 0.02, genre embedding std 0.5.
    Reason: numerical tokens should have a magnitude comparable to image/text
    tokens after projection (≈0.5–0.6), rather than 0.02 (old default),
    which is 30x smaller.
    """
    def __init__(self, n_continuous, n_genres_with_unk, d, dropout=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_continuous, d) * 0.5)
        self.bias = nn.Parameter(torch.randn(n_continuous, d) * 0.02)
        self.genre_emb = nn.Embedding(n_genres_with_unk, d)
        nn.init.normal_(self.genre_emb.weight, std=0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, continuous, genre_idx):
        # continuous: (B, n_cont) fp32
        # genre_idx:  (B,) int64, 0 = <unk>
        num_tokens = continuous.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        gen_token = self.genre_emb(genre_idx).unsqueeze(1)
        x = torch.cat([num_tokens, gen_token], dim=1)  # (B, n_cont + 1, d)
        return self.dropout(x)