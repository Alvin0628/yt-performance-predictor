"""RATF M6 — Granular v2: M5 + reliability gate yang benar-benar bisa belajar.

Perubahan dari ratf_m6_granular.py (tandai [CHANGED] / [NEW] / [REMOVED]):

[CHANGED] ReliabilityGate per-token (sigmoid independen, faktor 1+alpha*r di [1,2])
          -> ModalityReliabilityGate per-sample per-modality.
          * Input  : mean-pool tiap modality (text: masked mean, tabular: 12 token -> 1).
          * Output : bobot w_m, mode "softmax" => w = M * softmax(logit)  (rata-rata = 1)
                     mode "sigmoid2"           => w = 2 * sigmoid(logit)   (range 0..2)
          * Identity saat init (last layer zero-init -> w = 1 semua modality), gradien tetap
            mengalir ke last layer. Gate bisa MENURUNKAN modality (w < 1), tidak hanya menaikkan.
[REMOVED] alpha pada gate (tidak diperlukan lagi).
[NEW]     Modality dropout (text / image diganti learned "missing token" per sample, ditandai
          sebagai padding di joint transformer, dan dikeluarkan dari softmax gate).
[NEW]     self._last_gate: tensor bobot (B, M) supaya statistik bisa dihitung di seluruh val set.
[CHANGED] get_last_reliability() memakai statistik std ANTAR SAMPLE (bukan std token).
"""
import torch
import torch.nn as nn

from early_fusion.models.feature_tokenizer import TabularTokenizer
from early_fusion.models.fusion_transformer import JointTransformer


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention dengan residual scaling learnable. (tidak berubah)"""
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


# ----------------------------------------------------------------------------------
# [CHANGED] Gate baru: level modality, bukan level token
# ----------------------------------------------------------------------------------
class ModalityReliabilityGate(nn.Module):
    """Bobot reliabilitas per sample per modality.

    pooled : list of (B, d), urutan sama dengan kolom output.
    avail  : (B, M) bool, False = modality di-drop (bobot 0, keluar dari softmax).
    return : w (B, M), logits (B, M)
    """
    def __init__(self, d, n_modalities, mode="softmax", dropout=0.1):
        super().__init__()
        assert mode in ("softmax", "sigmoid2")
        self.mode = mode
        self.n_modalities = n_modalities
        hidden = max(d // 4, 16)
        self.scorers = nn.ModuleList()
        for _ in range(n_modalities):
            last = nn.Linear(hidden, 1)
            nn.init.zeros_(last.weight)   # identity saat init, gradien ke 'last' tetap ada
            nn.init.zeros_(last.bias)
            self.scorers.append(nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                last,
            ))

    def forward(self, pooled, avail=None):
        logits = torch.cat([s(p) for s, p in zip(self.scorers, pooled)], dim=1)  # (B, M)
        if self.mode == "softmax":
            if avail is not None:
                logits = logits.masked_fill(~avail, float("-inf"))
                n = avail.sum(dim=1, keepdim=True).clamp(min=1).to(logits.dtype)
            else:
                n = float(self.n_modalities)
            w = n * torch.softmax(logits, dim=1)   # mean(w over available) = 1
        else:
            w = 2.0 * torch.sigmoid(logits)
            if avail is not None:
                w = w * avail.to(w.dtype)
        return w, logits


class RATF_M6_Granular_V2(nn.Module):
    def __init__(self,
                 image_dim=768, text_dim=512,
                 n_continuous=11, n_genres=10,
                 d=128, nhead=4, num_layers=2, dim_ff=512,
                 n_image_tokens=50, n_text_tokens=32,
                 dropout=0.2, embedding_noise_std=0.02,
                 use_image=True, use_text=True,
                 cross_attn_layers=1,
                 gate_mode="softmax",          # [NEW]
                 gate_dropout=0.1,             # [NEW]
                 modality_dropout=0.0):        # [NEW] p drop per modality (text/image), tabular tidak pernah di-drop
        super().__init__()
        self.embedding_noise_std = embedding_noise_std
        self.use_image = use_image
        self.use_text = use_text
        self.cross_attn_layers = cross_attn_layers
        self.n_image_tokens = n_image_tokens
        self.n_text_tokens = n_text_tokens
        self.modality_dropout = modality_dropout

        self.image_ln = nn.LayerNorm(image_dim)
        self.text_ln = nn.LayerNorm(text_dim)

        self.image_proj = nn.Linear(image_dim, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.tabular_tokenizer = TabularTokenizer(n_continuous, n_genres, d, dropout)

        self.modality_emb = nn.Embedding(3, d)   # 0=text, 1=image, 2=tab
        nn.init.normal_(self.modality_emb.weight, std=0.02)
        self.pos_image = nn.Parameter(torch.randn(n_image_tokens, d) * 0.02)
        self.pos_text = nn.Parameter(torch.randn(n_text_tokens, d) * 0.02)
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
        # tabular-only: tidak ada yang dibandingkan -> tanpa gate
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

    @staticmethod
    def _masked_mean(x, valid=None):
        """x: (B, L, d); valid: (B, L) bool True = token valid."""
        if valid is None:
            return x.mean(dim=1)
        v = valid.unsqueeze(-1).to(x.dtype)
        return (x * v).sum(1) / v.sum(1).clamp(min=1.0)

    def forward(self, image_tokens, text_tokens, text_mask, continuous, genre_idx,
                return_reliability=False):
        B, dev = continuous.size(0), continuous.device
        zeros = lambda n: torch.zeros(B, n, dtype=torch.bool, device=dev)
        text_valid = text_mask.bool()  # True = token valid

        # [NEW] sampling modality dropout (hanya saat training)
        drop_t = drop_i = None
        if self.training and self.modality_dropout > 0:
            if self.use_text:
                drop_t = torch.rand(B, device=dev) < self.modality_dropout
            if self.use_image:
                drop_i = torch.rand(B, device=dev) < self.modality_dropout

        tab = self.tabular_tokenizer(continuous, genre_idx)  # (B, 12, d)

        txt = None
        txt_mask_bool = None
        if self.use_text:
            txt = self.text_proj(self._noise(self.text_ln(text_tokens))) + self.pos_text
            txt_mask_bool = ~text_valid  # True = ignore
            if drop_t is not None:  # [NEW] ganti isi dengan missing token (tanpa leak, tanpa NaN)
                txt = torch.where(drop_t[:, None, None], self.missing_text.expand_as(txt), txt)

        img = None
        if self.use_image:
            img = self.image_proj(self._noise(self.image_ln(image_tokens))) + self.pos_image
            if drop_i is not None:  # [NEW]
                img = torch.where(drop_i[:, None, None], self.missing_image.expand_as(img), img)

        # Cross-attention tri-modal paralel (tidak berubah)
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

        # Modality embedding (tidak berubah)
        tab = tab + self.modality_emb.weight[2]
        if self.use_text:
            txt = txt + self.modality_emb.weight[0]
        if self.use_image:
            img = img + self.modality_emb.weight[1]

        # ------------------------------------------------------------------
        # [CHANGED] Reliability gate level modality
        # ------------------------------------------------------------------
        w = None
        if self.gate is not None:
            pooled, avail = [], []
            if self.use_text:
                pooled.append(self._masked_mean(txt, text_valid))
                avail.append(~drop_t if drop_t is not None else torch.ones(B, dtype=torch.bool, device=dev))
            if self.use_image:
                pooled.append(self._masked_mean(img))
                avail.append(~drop_i if drop_i is not None else torch.ones(B, dtype=torch.bool, device=dev))
            pooled.append(self._masked_mean(tab))
            avail.append(torch.ones(B, dtype=torch.bool, device=dev))

            w, _ = self.gate(pooled, torch.stack(avail, dim=1))  # (B, M)

            k = {n: j for j, n in enumerate(self.mod_names)}
            tab = tab * w[:, k["tab"]].view(B, 1, 1)
            if self.use_text:
                txt = txt * w[:, k["text"]].view(B, 1, 1)
            if self.use_image:
                img = img * w[:, k["image"]].view(B, 1, 1)

        if return_reliability and w is not None:
            self._last_gate = w.detach()

        # Sequence [CLS, text(32), image(50), tabular(12)]
        parts = [self.cls_token.expand(B, -1, -1)]
        masks = [zeros(1)]
        if self.use_text:
            parts.append(txt)
            m = txt_mask_bool
            if drop_t is not None:  # [NEW] modality yang di-drop tidak terlihat joint transformer
                m = m | drop_t[:, None]
            masks.append(m)
        if self.use_image:
            parts.append(img)
            m = zeros(img.size(1))
            if drop_i is not None:  # [NEW]
                m = m | drop_i[:, None]
            masks.append(m)
        parts.append(tab)
        masks.append(zeros(tab.size(1)))   # CLS + tab tidak pernah di-mask -> tidak ada baris all-masked

        x = torch.cat(parts, 1)
        pad_mask = torch.cat(masks, 1)

        x = self.transformer(x, padding_mask=pad_mask)
        return self.head(self.final_ln(x[:, 0])).squeeze(-1)

    # [CHANGED] statistik di level sample
    def get_last_gate_weights(self):
        """(names, w[B, M]) dari forward terakhir dengan return_reliability=True."""
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
