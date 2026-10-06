"""RATF M6 — Granular variant: M5 + reliability gate dengan token penuh.

Perbedaan dari ratf_m6.py (pooled):
- Image: 50 token (CLS + 49 patch), bukan 1 pooled.
- Text: 32 token, bukan 1 EOS pooled.
- Tambah positional embedding untuk image (50) dan text (32).
- Cross-attention dengan key_padding_mask untuk text.

Struktur:
1. Proyeksi (image 50 token, text 32 token, tabular 12 token).
2. Tri-modal cross-attention paralel.
3. Modality + positional embedding.
4. Reliability gate per token.
5. Sequence [CLS, text(32), image(50), tabular(12)] = 95 token.
6. Joint self-attention.
7. Head.
"""
import torch
import torch.nn as nn

from early_fusion.models.feature_tokenizer import TabularTokenizer
from early_fusion.models.fusion_transformer import JointTransformer


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention dengan residual scaling learnable."""
    def __init__(self, d, nhead, dropout=0.2, init_alpha=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, q, kv, kv_mask=None):
        q_n = self.ln_q(q)
        kv_n = self.ln_kv(kv)
        attn_out, _ = self.attn(
            q_n, kv_n, kv_n, key_padding_mask=kv_mask, need_weights=False,
        )
        return self.ln_out(q + self.alpha * self.dropout(attn_out))


class ReliabilityGate(nn.Module):
    """Per-token reliability: r_i = sigmoid(MLP(h_i)); h'_i = (1 + alpha * r_i) * h_i."""
    def __init__(self, d, dropout=0.2, init_alpha=1.0):
        super().__init__()
        hidden = max(d // 4, 16)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, h):
        r = torch.sigmoid(self.mlp(h))
        return (1.0 + self.alpha * r) * h, r


class RATF_M6_Granular(nn.Module):
    def __init__(self,
                 image_dim=768, text_dim=512,
                 n_continuous=11, n_genres=10,
                 d=128, nhead=4, num_layers=2, dim_ff=512,
                 n_image_tokens=50, n_text_tokens=32,
                 dropout=0.2, embedding_noise_std=0.02,
                 use_image=True, use_text=True,
                 cross_attn_layers=1):
        super().__init__()
        self.embedding_noise_std = embedding_noise_std
        self.use_image = use_image
        self.use_text = use_text
        self.cross_attn_layers = cross_attn_layers
        self.n_image_tokens = n_image_tokens
        self.n_text_tokens = n_text_tokens

        # Input LayerNorm
        self.image_ln = nn.LayerNorm(image_dim)
        self.text_ln = nn.LayerNorm(text_dim)

        # Projection
        self.image_proj = nn.Linear(image_dim, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.tabular_tokenizer = TabularTokenizer(n_continuous, n_genres, d, dropout)

        # Modality + positional + CLS
        self.modality_emb = nn.Embedding(3, d)
        nn.init.normal_(self.modality_emb.weight, std=0.02)
        self.pos_image = nn.Parameter(torch.randn(n_image_tokens, d) * 0.02)
        self.pos_text = nn.Parameter(torch.randn(n_text_tokens, d) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Cross-attention blocks
        self.cross_blocks = nn.ModuleDict()
        for i in range(cross_attn_layers):
            if use_text:
                self.cross_blocks[f"t2tab_{i}"] = CrossAttentionBlock(d, nhead, dropout)
                self.cross_blocks[f"tab2t_{i}"] = CrossAttentionBlock(d, nhead, dropout)
            if use_image:
                self.cross_blocks[f"i2tab_{i}"] = CrossAttentionBlock(d, nhead, dropout)
                self.cross_blocks[f"tab2i_{i}"] = CrossAttentionBlock(d, nhead, dropout)
            if use_text and use_image:
                self.cross_blocks[f"t2i_{i}"] = CrossAttentionBlock(d, nhead, dropout)
                self.cross_blocks[f"i2t_{i}"] = CrossAttentionBlock(d, nhead, dropout)

        # Reliability gates
        if use_text:
            self.rel_text = ReliabilityGate(d, dropout)
        if use_image:
            self.rel_image = ReliabilityGate(d, dropout)
        self.rel_tab = ReliabilityGate(d, dropout)

        # Encoder + head
        self.transformer = JointTransformer(d, nhead, num_layers, dim_ff, dropout)
        self.final_ln = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._last_reliability = {}

    def _noise(self, x):
        if self.training and self.embedding_noise_std > 0:
            return x + torch.randn_like(x) * self.embedding_noise_std
        return x

    def forward(self, image_tokens, text_tokens, text_mask, continuous, genre_idx,
                return_reliability=False):
        B, dev = continuous.size(0), continuous.device
        zeros = lambda n: torch.zeros(B, n, dtype=torch.bool, device=dev)

        # Tabular: 12 token
        tab = self.tabular_tokenizer(continuous, genre_idx)  # (B, 12, d)

        # Text: 32 token
        txt = None
        txt_mask_bool = None
        if self.use_text:
            txt = self.text_proj(self._noise(self.text_ln(text_tokens)))  # (B, 32, d)
            txt = txt + self.pos_text
            txt_mask_bool = ~text_mask.bool()  # True = ignore

        # Image: 50 token
        img = None
        if self.use_image:
            img = self.image_proj(self._noise(self.image_ln(image_tokens)))  # (B, 50, d)
            img = img + self.pos_image

        # Cross-attention tri-modal paralel
        for i in range(self.cross_attn_layers):
            txt_orig, img_orig, tab_orig = txt, img, tab

            if self.use_text and self.use_image:
                txt_from_img = self.cross_blocks[f"t2i_{i}"](txt_orig, img_orig)
                txt_from_tab = self.cross_blocks[f"t2tab_{i}"](txt_orig, tab_orig)
                txt_new = (txt_from_img + txt_from_tab) / 2.0

                img_from_txt = self.cross_blocks[f"i2t_{i}"](img_orig, txt_orig, kv_mask=txt_mask_bool)
                img_from_tab = self.cross_blocks[f"i2tab_{i}"](img_orig, tab_orig)
                img_new = (img_from_txt + img_from_tab) / 2.0

                tab_from_txt = self.cross_blocks[f"tab2t_{i}"](tab_orig, txt_orig, kv_mask=txt_mask_bool)
                tab_from_img = self.cross_blocks[f"tab2i_{i}"](tab_orig, img_orig)
                tab_new = (tab_from_txt + tab_from_img) / 2.0

                txt, img, tab = txt_new, img_new, tab_new

            elif self.use_text:
                txt_from_tab = self.cross_blocks[f"t2tab_{i}"](txt_orig, tab_orig)
                tab_from_txt = self.cross_blocks[f"tab2t_{i}"](tab_orig, txt_orig, kv_mask=txt_mask_bool)
                txt, tab = txt_from_tab, tab_from_txt

            elif self.use_image:
                img_from_tab = self.cross_blocks[f"i2tab_{i}"](img_orig, tab_orig)
                tab_from_img = self.cross_blocks[f"tab2i_{i}"](tab_orig, img_orig)
                img, tab = img_from_tab, tab_from_img

        # Modality embedding setelah cross-attention
        tab = tab + self.modality_emb.weight[2]
        if self.use_text:
            txt = txt + self.modality_emb.weight[0]
        if self.use_image:
            img = img + self.modality_emb.weight[1]

        # Reliability gate
        tab, r_tab = self.rel_tab(tab)
        if self.use_text:
            txt, r_txt = self.rel_text(txt)
        if self.use_image:
            img, r_img = self.rel_image(img)

        if return_reliability:
            self._last_reliability = {
                "tab_mean": float(r_tab.mean().item()),
                "tab_std": float(r_tab.std().item()),
            }
            if self.use_text:
                self._last_reliability["text_mean"] = float(r_txt.mean().item())
                self._last_reliability["text_std"] = float(r_txt.std().item())
            if self.use_image:
                self._last_reliability["image_mean"] = float(r_img.mean().item())
                self._last_reliability["image_std"] = float(r_img.std().item())

        # Sequence [CLS, text(32), image(50), tabular(12)]
        parts = [self.cls_token.expand(B, -1, -1)]
        masks = [zeros(1)]
        if self.use_text:
            parts.append(txt)
            masks.append(txt_mask_bool)
        if self.use_image:
            parts.append(img)
            masks.append(zeros(img.size(1)))
        parts.append(tab)
        masks.append(zeros(tab.size(1)))

        x = torch.cat(parts, 1)
        pad_mask = torch.cat(masks, 1)

        x = self.transformer(x, padding_mask=pad_mask)
        return self.head(self.final_ln(x[:, 0])).squeeze(-1)

    def get_last_reliability(self):
        return dict(self._last_reliability)