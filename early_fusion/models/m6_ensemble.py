"""Inferensi model akhir M6 granular v2 (ensemble N seed) dari SATU file .pt.

Lokasi yang disarankan: early_fusion/models/m6_ensemble.py

Pemakaian minimal:
    from early_fusion.models.m6_ensemble import M6Ensemble
    ens = M6Ensemble.load("early_fusion/models/final/m6_granular_ensemble_v1.pt")
    cont, gidx = ens.prepare_tabular(df)                    # df = baris-baris berformat snapshot
    score = ens.predict(image_tokens, text_tokens, text_mask, cont, gidx)          # (n,)
    score, members = ens.predict(..., return_members=True)  # members: (n_member, n)

Skor = prediksi target (log rasio views terhadap trailing average); makin tinggi makin diperkirakan
berkinerja di atas rata-rata kanal. Yang bermakna adalah URUTAN antar video (metrik evaluasi = Spearman),
bukan nilai absolut. Simpangan antar anggota (members.std(0)) bisa dipakai sebagai indikator ketidakpastian.

Input (sama persis dengan saat training / token cache):
    image_tokens (n, 50, 768)   CLS + 49 patch
    text_tokens  (n, 32, 512)
    text_mask    (n, 32)        True = token valid
    continuous   (n, n_cont)    dari prepare_tabular (scaler train + subs sudah dibuang)
    genre_idx    (n,)           0 = <unk>

Catatan keamanan: file .pt memuat objek sklearn (scaler) sehingga dimuat dengan weights_only=False.
Muat HANYA file milik sendiri / yang checksum-nya dikenal (lihat file .json pendamping).
"""
import numpy as np
import torch

from early_fusion.models.ratf_m6_granular_v2 import RATF_M6_Granular_V2

N_IMAGE_TOKENS = 50
N_TEXT_TOKENS = 32
NHEAD = 4
SUBS_IDX = 0
FORMAT_VERSION = 1


def _build(cfg, variant, n_cont, n_genres):
    use_image = variant in ("full", "no_text")
    use_text = variant in ("full", "no_image")
    return RATF_M6_Granular_V2(
        image_dim=768, text_dim=512,
        n_continuous=n_cont, n_genres=n_genres,
        d=cfg["d"], nhead=NHEAD, num_layers=cfg["num_layers"],
        dim_ff=int(cfg["d"] * cfg["ff_mult"]),
        n_image_tokens=N_IMAGE_TOKENS, n_text_tokens=N_TEXT_TOKENS,
        dropout=cfg["dropout"], embedding_noise_std=cfg["emb_noise"],
        use_image=use_image, use_text=use_text,
        cross_attn_layers=cfg["cross_attn_layers"],
        gate_mode=cfg["gate_mode"], gate_dropout=cfg["gate_dropout"],
        modality_dropout=cfg["mod_dropout"],
    )


class M6Ensemble:
    def __init__(self, models, meta, scaler, genres_train, device):
        self.models = models
        self.meta = meta
        self.scaler = scaler
        self.genres_train = genres_train
        self.device = device

    @classmethod
    def load(cls, path, device=None):
        device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"format_version tidak dikenal: {blob.get('format_version')}")
        models = []
        for m in blob["members"]:
            net = _build(blob["cfg"], blob["variant"], blob["n_cont"], blob["n_genres"])
            net.load_state_dict(m["state_dict"])
            net.to(device).eval()
            models.append(net)
        meta = {k: blob[k] for k in blob if k not in ("members", "scaler")}
        return cls(models, meta, blob["scaler"], blob["genres_train"], device)

    @property
    def n_members(self):
        return len(self.models)

    def prepare_tabular(self, df):
        """df (format snapshot) -> (continuous float32, genre_idx int64), identik dengan pipeline training."""
        from models.dataset import build_tabular_matrix
        mat, _ = build_tabular_matrix(df, self.genres_train, scaler=self.scaler)
        mat = np.delete(mat, SUBS_IDX, axis=1).astype(np.float32)        # drop subscriber_count
        g2i = {g: i + 1 for i, g in enumerate(self.genres_train)}          # 0 = <unk>
        gidx = np.array([g2i.get(g, 0) for g in df["genre"].fillna("")], dtype=np.int64)
        return mat, gidx

    @torch.no_grad()
    def predict_batch(self, image_tokens, text_tokens, text_mask, continuous, genre_idx):
        """Satu batch tensor (sudah di device). Return numpy (n_member, batch)."""
        outs = []
        for net in self.models:
            p = net(image_tokens, text_tokens, text_mask, continuous, genre_idx)
            outs.append(p.detach().float().cpu().numpy())
        return np.stack(outs, axis=0)

    @torch.no_grad()
    def predict(self, image_tokens, text_tokens, text_mask, continuous, genre_idx,
                batch_size=256, return_members=False):
        n = len(continuous)
        dev = self.device
        member_preds = np.zeros((self.n_members, n), dtype=np.float32)
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            img = torch.as_tensor(image_tokens[sl]).to(dev).float()
            txt = torch.as_tensor(text_tokens[sl]).to(dev).float()
            msk = torch.as_tensor(text_mask[sl]).to(dev)
            tab = torch.as_tensor(continuous[sl]).to(dev).float()
            gi = torch.as_tensor(genre_idx[sl]).to(dev).long()
            member_preds[:, i:i + batch_size] = self.predict_batch(img, txt, msk, tab, gi)
        mean = member_preds.mean(axis=0)
        return (mean, member_preds) if return_members else mean
