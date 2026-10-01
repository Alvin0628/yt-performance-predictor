"""RATF — M4a: pooled variant of M4.

Image → 1 token (CLS pooled, position 0 of 50 tokens).
Text  → 1 token (EOS pooled, last position according to the mask).
Tabular → 12 tokens (same as M4).

Total 15 tokens (not 95).

This is a separate file from ratf.py so that:
- ratf.py remains the M4 baseline (95 tokens).
- ratf_pooled.py = M4a (15 tokens).
- Git history stays clean, with clear traceability.
"""
import torch
import torch.nn as nn

from early_fusion.models.feature_tokenizer import TabularTokenizer
from early_fusion.models.fusion_transformer import JointTransformer


class RATF_M4a(nn.Module):
    def __init__(self,
                 image_dim=768, text_dim=512,
                 n_continuous=11, n_genres=10,
                 d=128, nhead=4, num_layers=2, dim_ff=512,
                 dropout=0.2, embedding_noise_std=0.02,
                 use_image=True, use_text=True):
        super().__init__()
        self.embedding_noise_std = embedding_noise_std
        self.use_image = use_image
        self.use_text = use_text

        # Input LayerNorm
        self.image_ln = nn.LayerNorm(image_dim)
        self.text_ln = nn.LayerNorm(text_dim)

        # Projection to d (image/text are already pooled → 1 token)
        self.image_proj = nn.Linear(image_dim, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.tabular_tokenizer = TabularTokenizer(n_continuous, n_genres, d, dropout)

        # Modality + CLS
        self.modality_emb = nn.Embedding(3, d)
        nn.init.normal_(self.modality_emb.weight, std=0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # No positional embedding: the sequence only has 15 tokens,
        # image/text each have only 1 token, while tabular remains 12 tokens.
        # The Transformer can distinguish positions via CLS + modality embeddings.

        # Encoder + head
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

        # Tabular: 12 tokens
        tab = self.tabular_tokenizer(continuous, genre_idx) + self.modality_emb.weight[2]

        # CLS
        parts = [self.cls_token.expand(B, -1, -1)]
        masks = [zeros(1)]

        # Text pooled: EOS hidden state
        if self.use_text:
            eos_idx = (text_mask.sum(dim=1) - 1).long().clamp(min=0)
            txt_pooled = text_tokens[torch.arange(B, device=dev), eos_idx]  # (B, 512)
            txt = self.text_proj(self._noise(self.text_ln(txt_pooled))).unsqueeze(1)  # (B, 1, d)
            parts.append(txt + self.modality_emb.weight[0])
            masks.append(zeros(1))

        # Image pooled: CLS token (position 0 of 50)
        if self.use_image:
            img_pooled = image_tokens[:, 0]  # (B, 768)
            img = self.image_proj(self._noise(self.image_ln(img_pooled))).unsqueeze(1)  # (B, 1, d)
            parts.append(img + self.modality_emb.weight[1])
            masks.append(zeros(1))

        # Tabular
        parts.append(tab)
        masks.append(zeros(tab.size(1)))

        x = self.transformer(torch.cat(parts, 1), padding_mask=torch.cat(masks, 1))
        return self.head(self.final_ln(x[:, 0])).squeeze(-1)