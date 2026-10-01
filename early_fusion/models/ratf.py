"""RATF — M4: token-level early fusion.

Features:
- use_image / use_text can be disabled for ablation ladder.
- Input LayerNorm (raw token magnitude is large).
- Modality embedding init std 0.02 (instead of default N(0,1)).
"""
import torch
import torch.nn as nn

from early_fusion.models.feature_tokenizer import TabularTokenizer
from early_fusion.models.fusion_transformer import JointTransformer


class RATF_M4(nn.Module):
    def __init__(self,
                 image_dim=768, text_dim=512,
                 n_continuous=11, n_genres=10,
                 d=128, nhead=4, num_layers=2, dim_ff=512,
                 n_image_tokens=50, n_text_tokens=32,
                 dropout=0.2, embedding_noise_std=0.02,
                 use_image=True, use_text=True):
        super().__init__()
        self.embedding_noise_std = embedding_noise_std
        self.use_image = use_image
        self.use_text = use_text

        # Input LayerNorm
        self.image_ln = nn.LayerNorm(image_dim)
        self.text_ln = nn.LayerNorm(text_dim)

        # Projection
        self.image_proj = nn.Linear(image_dim, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.tabular_tokenizer = TabularTokenizer(n_continuous, n_genres, d, dropout)

        # Modality + positional embeddings
        self.modality_emb = nn.Embedding(3, d)
        nn.init.normal_(self.modality_emb.weight, std=0.02)
        self.pos_image = nn.Parameter(torch.randn(n_image_tokens, d) * 0.02)
        self.pos_text = nn.Parameter(torch.randn(n_text_tokens, d) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Encoder + final LN + head
        self.transformer = JointTransformer(d, nhead, num_layers, dim_ff, dropout)
        self.final_ln = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _noise(self, x):
        if self.training and self.embedding_noise_std > 0:
            return x + torch.randn_like(x) * self.embedding_noise_std
        return x

    def forward(self, image_tokens, text_tokens, text_mask, continuous, genre_idx):
        B, dev = continuous.size(0), continuous.device
        zeros = lambda n: torch.zeros(B, n, dtype=torch.bool, device=dev)

        tab = self.tabular_tokenizer(continuous, genre_idx) + self.modality_emb.weight[2]
        parts = [self.cls_token.expand(B, -1, -1)]
        masks = [zeros(1)]

        if self.use_text:
            txt = self.text_proj(self._noise(self.text_ln(text_tokens)))
            parts.append(txt + self.modality_emb.weight[0] + self.pos_text)
            masks.append(~text_mask.bool())

        if self.use_image:
            img = self.image_proj(self._noise(self.image_ln(image_tokens)))
            parts.append(img + self.modality_emb.weight[1] + self.pos_image)
            masks.append(zeros(img.size(1)))

        parts.append(tab)
        masks.append(zeros(tab.size(1)))

        x = self.transformer(torch.cat(parts, 1), padding_mask=torch.cat(masks, 1))
        return self.head(self.final_ln(x[:, 0])).squeeze(-1)