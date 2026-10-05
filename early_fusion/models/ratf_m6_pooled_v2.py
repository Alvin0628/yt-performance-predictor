"""RATF M6 — Pooled v2: M5 pooled + ModalityReliabilityGate + modality dropout.

Versi pooled dari ratf_m6_granular_v2.py. Satu-satunya variabel yang berbeda dari granular v2
adalah JUMLAH TOKEN:  image 1 (CLS)  + text 1 (EOS)  + tabular 12  + CLS 1  = 15 token  (granular: 95).

Disamakan PERSIS dengan granular v2 (di-import, bukan disalin, supaya tidak bisa drift):
  - CrossAttentionBlock
  - ModalityReliabilityGate (softmax / sigmoid2, last layer zero-init)

Perubahan dari ratf_m6.py (v1 pooled) — tanda [CHANGED] / [NEW] / [REMOVED]:
[REMOVED]  ReliabilityGate per-token (sigmoid independen, faktor 1+alpha*r di [1,2]).
[CHANGED]  Gate -> ModalityReliabilityGate per-sample per-modality (pooled input = token itu sendiri
           untuk text/image, mean 12 token untuk tabular).
[NEW]      Modality dropout (text/image -> learned missing token, di-mask di joint transformer,
           dikeluarkan dari softmax gate). Tabular tidak pernah di-drop.
[NEW]      _last_gate (B, M) untuk statistik seluruh val set.
[CHANGED]  Tabular-only: tanpa gate (tidak ada yang dibandingkan).
[CHANGED]  Bug kecil v1: `zeros(tab.size(0) if False else tab.size(1))` dibersihkan.
Tidak berubah: pooling (CLS image, EOS text), urutan cross-attn paralel, modality emb setelah cross-attn.
"""
import torch
import torch.nn as nn

from early_fusion.models.feature_tokenizer import TabularTokenizer
from early_fusion.models.fusion_transformer import JointTransformer
from early_fusion.models.ratf_m6_granular_v2 import (   # [CHANGED] reuse, bukan salin
    CrossAttentionBlock, ModalityReliabilityGate,
)


class RATF_M6_Pooled_V2(nn.Module):
    def __init__(self,
                 image_dim=768, text_dim=512,
                 n_continuous=11, n_genres=10,
                 d=128, nhead=4, num_layers=2, dim_ff=512,
                 dropout=0.2, embedding_noise_std=0.02,
                 use_image=True, use_text=True,
                 cross_attn_layers=1,
                 gate_mode="softmax",          # [NEW]
                 gate_dropout=0.1,             # [NEW]
                 modality_dropout=0.0):        # [NEW]
        super().__init__()
        self.embedding_noise_std = embedding_noise_std
        self.use_image = use_image
        self.use_text = use_text
        self.cross_attn_layers = cross_attn_layers
        self.modality_dropout = modality_dropout

        self.image_ln = nn.LayerNorm(image_dim)
        self.text_ln = nn.LayerNorm(text_dim)

        self.image_proj = nn.Linear(image_dim, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.tabular_tokenizer = TabularTokenizer(n_continuous, n_genres, d, dropout)

        self.modality_emb = nn.Embedding(3, d)   # 0=text, 1=image, 2=tab
        nn.init.normal_(self.modality_emb.weight, std=0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # [NEW] learned missing token untuk modality dropout
        if use_text:
            self.missing_text = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if use_image:
            self.missing_image = nn.Parameter(torch.randn(1, 1, d) * 0.02)

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

        # [CHANGED] satu gate untuk semua modality (urutan: text, image, tab)
        self.mod_names = (["text"] if use_text else []) + (["image"] if use_image else []) + ["tab"]
        self.gate = (ModalityReliabilityGate(d, len(self.mod_names), gate_mode, gate_dropout)
                     if len(self.mod_names) > 1 else None)

        self.transformer = JointTransformer(d, nhead, num_layers, dim_ff, dropout)
        self.final_ln = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._last_gate = None

    def _noise(self, x):
        if self.training and self.embedding_noise_std > 0:
            return x + torch.randn_like(x) * self.embedding_noise_std
        return x

    def forward(self, image_tokens, text_tokens, text_mask, continuous, genre_idx,
                return_reliability=False):
        B, dev = continuous.size(0), continuous.device
        zeros = lambda n: torch.zeros(B, n, dtype=torch.bool, device=dev)
        ones_b = lambda: torch.ones(B, dtype=torch.bool, device=dev)

        # [NEW] sampling modality dropout (hanya saat training)
        drop_t = drop_i = None
        if self.training and self.modality_dropout > 0:
            if self.use_text:
                drop_t = torch.rand(B, device=dev) < self.modality_dropout
            if self.use_image:
                drop_i = torch.rand(B, device=dev) < self.modality_dropout

        tab = self.tabular_tokenizer(continuous, genre_idx)  # (B, 12, d)

        # ---------- Pooling (tidak berubah dari v1) ----------
        txt = None
        if self.use_text:
            eos_idx = (text_mask.sum(dim=1) - 1).long().clamp(min=0)
            txt_pooled = text_tokens[torch.arange(B, device=dev), eos_idx]
            txt = self.text_proj(self._noise(self.text_ln(txt_pooled))).unsqueeze(1)  # (B,1,d)
            if drop_t is not None:  # [NEW]
                txt = torch.where(drop_t[:, None, None], self.missing_text.expand_as(txt), txt)

        img = None
        if self.use_image:
            img_pooled = image_tokens[:, 0]
            img = self.image_proj(self._noise(self.image_ln(img_pooled))).unsqueeze(1)  # (B,1,d)
            if drop_i is not None:  # [NEW]
                img = torch.where(drop_i[:, None, None], self.missing_image.expand_as(img), img)

        # ---------- Cross-attention tri-modal paralel (tidak berubah) ----------
        for i in range(self.cross_attn_layers):
            txt_orig, img_orig, tab_orig = txt, img, tab

            if self.use_text and self.use_image:
                txt_from_img = self.cross_blocks[f"t2i_{i}"](txt_orig, img_orig)
                txt_from_tab = self.cross_blocks[f"t2tab_{i}"](txt_orig, tab_orig)
                txt_new = (txt_from_img + txt_from_tab) / 2.0

                img_from_txt = self.cross_blocks[f"i2t_{i}"](img_orig, txt_orig)
                img_from_tab = self.cross_blocks[f"i2tab_{i}"](img_orig, tab_orig)
                img_new = (img_from_txt + img_from_tab) / 2.0

                tab_from_txt = self.cross_blocks[f"tab2t_{i}"](tab_orig, txt_orig)
                tab_from_img = self.cross_blocks[f"tab2i_{i}"](tab_orig, img_orig)
                tab_new = (tab_from_txt + tab_from_img) / 2.0

                txt, img, tab = txt_new, img_new, tab_new

            elif self.use_text:
                txt_from_tab = self.cross_blocks[f"t2tab_{i}"](txt_orig, tab_orig)
                tab_from_txt = self.cross_blocks[f"tab2t_{i}"](tab_orig, txt_orig)
                txt, tab = txt_from_tab, tab_from_txt

            elif self.use_image:
                img_from_tab = self.cross_blocks[f"i2tab_{i}"](img_orig, tab_orig)
                tab_from_img = self.cross_blocks[f"tab2i_{i}"](tab_orig, img_orig)
                img, tab = img_from_tab, tab_from_img

        # ---------- Modality embedding setelah cross-attention (tidak berubah) ----------
        tab = tab + self.modality_emb.weight[2]
        if self.use_text:
            txt = txt + self.modality_emb.weight[0]
        if self.use_image:
            img = img + self.modality_emb.weight[1]

        # ---------- [CHANGED] Reliability gate level modality ----------
        w = None
        if self.gate is not None:
            pooled, avail = [], []
            if self.use_text:
                pooled.append(txt.squeeze(1))                      # 1 token -> itu sendiri
                avail.append(~drop_t if drop_t is not None else ones_b())
            if self.use_image:
                pooled.append(img.squeeze(1))
                avail.append(~drop_i if drop_i is not None else ones_b())
            pooled.append(tab.mean(dim=1))                         # 12 token tabular -> 1
            avail.append(ones_b())

            w, _ = self.gate(pooled, torch.stack(avail, dim=1))    # (B, M)

            k = {n: j for j, n in enumerate(self.mod_names)}
            tab = tab * w[:, k["tab"]].view(B, 1, 1)
            if self.use_text:
                txt = txt * w[:, k["text"]].view(B, 1, 1)
            if self.use_image:
                img = img * w[:, k["image"]].view(B, 1, 1)

        if return_reliability and w is not None:
            self._last_gate = w.detach()

        # ---------- Sequence [CLS, text, image, tabular] = 15 token ----------
        parts = [self.cls_token.expand(B, -1, -1)]
        masks = [zeros(1)]
        if self.use_text:
            parts.append(txt)
            masks.append(drop_t[:, None] if drop_t is not None else zeros(1))   # [NEW]
        if self.use_image:
            parts.append(img)
            masks.append(drop_i[:, None] if drop_i is not None else zeros(1))   # [NEW]
        parts.append(tab)
        masks.append(zeros(tab.size(1)))   # [CHANGED] bersih; CLS + tab tidak pernah di-mask

        x = torch.cat(parts, 1)
        pad_mask = torch.cat(masks, 1)

        x = self.transformer(x, padding_mask=pad_mask)
        return self.head(self.final_ln(x[:, 0])).squeeze(-1)

    def get_last_gate_weights(self):
        return self.mod_names, self._last_gate

    def get_last_reliability(self):
        if self._last_gate is None:
            return {}
        out = {}
        for j, n in enumerate(self.mod_names):
            col = self._last_gate[:, j]
            out[f"{n}_mean"] = float(col.mean().item())
            out[f"{n}_std"] = float(col.std().item()) if col.numel() > 1 else 0.0
        return out
