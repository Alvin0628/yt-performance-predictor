"""M6 granular v2 — shared core untuk tuning, konfirmasi, final, dan refit.

Lokasi yang disarankan: early_fusion/experiments/m6_core.py

Prinsip:
  * Data (snapshot, split, token cache, scaler) dimuat SEKALI per proses -> load_data().
  * Satu fungsi training -> train_one(cfg, seed, data, ...) dipakai SEMUA tahap, jadi tidak ada
    perbedaan diam-diam antara run tuning dan run final.
  * Model TIDAK diubah (ratf_m6_granular_v2.RATF_M6_Granular_V2), sehingga hasil v2 yang lama
    tetap valid sebagai titik referensi (lihat Phase 0: reproduksi).
  * Test set TIDAK pernah disentuh kecuali eval_test=True (hanya dipanggil run_m6_config.py,
    yang punya ledger penjaga).

[SPEED] Seluruh token cache dipindah ke GPU SEKALI (load_data) dan batch diambil lewat indexing tensor.
        Versi lama memakai DataLoader(num_workers=0) + Subset + collate per-sample di Python -> bottleneck
        (tiap batch ~14 MB disusun per sample di CPU, GPU menganggur). Env var:
          M6_DATA_DEVICE=cpu   simpan data di CPU (bila VRAM kecil); tetap lebih cepat dari DataLoader
          M6_DATA_HALF=1       simpan token sebagai float16 (separuh memori); default: dtype asli cache
[SPEED] TF32 matmul (efektif di GPU Ampere+, tanpa efek di GPU lama), eval batch 512, tanpa .item() per step.
[SPEED] early_abort: hentikan run yang jelas buruk (best val Spearman < floor setelah min_epoch).
Catatan: urutan shuffle & TF32 mengubah angka sedikit dibanding versi lama (bukan bit-identik).

Fitur baru dibanding train_m6_temporal_granular_v2.py (semua diatur lewat cfg, default = perilaku lama):
  [NEW] schedule "cosine" (warmup -> cosine decay ke lr_min_frac) selain "const".
  [NEW] EMA bobot (ema_decay > 0); evaluasi memakai bobot EMA.
  [NEW] auxiliary pairwise ranking loss (rank_lambda > 0), selaras dengan metrik Spearman.
  [NEW] batch_size / weight_decay / dll. sebagai cfg; d, num_layers, cross_attn_layers, ff_mult.
  [NEW] fit="trainval": refit pada train+val dengan epoch tetap (untuk model produksi).

Dua protokol evaluasi (ditentukan oleh schedule):
  schedule="const"  -> "early_stop": pilih epoch dengan val loss (Huber) terbaik, patience (perilaku lama).
  schedule="cosine" -> "final"     : latih penuh sepanjang horizon `epochs`, pakai bobot AKHIR (EMA bila aktif).
                                      Tidak ada pemilihan epoch memakai val -> tidak ada epoch-peeking, dan
                                      resep yang sama bisa dipakai apa adanya untuk refit train+val.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_absolute_error

from models.dataset import build_tabular_matrix
from early_fusion.datasets.token_cache import load_cache
from early_fusion.models.ratf_m6_granular_v2 import RATF_M6_Granular_V2
from early_fusion.experiments._common import set_seed, get_git_info, load_snapshot
from early_fusion.splits.load_split import load_canonical_split, apply_split_to_df
from early_fusion.data_spec import DEFAULT_SPEC


# Kept for old callers; the values now come from DEFAULT_SPEC.
SNAPSHOT_HASH = DEFAULT_SPEC.snapshot_hash
CACHE_DIR = Path(DEFAULT_SPEC.cache_dir).as_posix()
SUBS_IDX = 0
N_IMAGE_TOKENS = 50
N_TEXT_TOKENS = 32
NHEAD = 4

# Default = konfigurasi C dari eksperimen sebelumnya (granular, sigmoid2, md0.15, glr10),
# dengan semua fitur baru dimatikan -> harus mereproduksi hasil C (Phase 0).
DEFAULT_CFG = dict(
    # arsitektur
    d=128, num_layers=2, cross_attn_layers=1, ff_mult=4,
    # regularisasi
    dropout=0.2, emb_noise=0.02, mod_dropout=0.15,
    # gate
    gate_mode="sigmoid2", gate_dropout=0.1, gate_lr_mult=10.0,
    # optimisasi
    lr=1e-4, weight_decay=0.01, batch_size=64, grad_clip=1.0,
    schedule="const", epochs=200, patience=15,
    warmup_epochs=0.7, lr_min_frac=0.01,
    # fitur baru (off by default)
    ema_decay=0.0, rank_lambda=0.0,
)

_huber = nn.HuberLoss()
EVAL_BS = 512
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.ascontiguousarray(x))


def _build_store(img_tokens, txt_tokens, txt_mask, cont_all, genre_idx, targets, device, verbose=True):
    """Semua tensor data dalam satu dict; default di GPU supaya batch = indexing tensor (tanpa DataLoader)."""
    want = os.environ.get("M6_DATA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    half = os.environ.get("M6_DATA_HALF", "0") == "1"
    img_t, txt_t = _to_tensor(img_tokens), _to_tensor(txt_tokens)
    if half:
        img_t, txt_t = img_t.half(), txt_t.half()
    cpu_store = dict(img=img_t, txt=txt_t, mask=_to_tensor(txt_mask),
                     tab=torch.from_numpy(cont_all), gi=torch.from_numpy(genre_idx), y=torch.from_numpy(targets))
    try:
        store = {k: v.to(want) for k, v in cpu_store.items()}
        where = want
    except RuntimeError as e:                      # OOM di GPU -> jatuh ke CPU
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        store, where = cpu_store, "cpu"
        print("[data] VRAM tidak cukup untuk token cache; memakai penyimpanan CPU (set M6_DATA_HALF=1 untuk hemat).")
    if verbose:
        mb = sum(v.numel() * v.element_size() for v in store.values()) / 1e6
        print(f"[data] token store di {where}: {mb:,.0f} MB "
              f"(img {tuple(store['img'].shape)} {store['img'].dtype}, txt {tuple(store['txt'].shape)} {store['txt'].dtype})")
    return store


def iterate_batches(store, idx, bs, device, shuffle=False, gen=None):
    """Generator batch dict (kunci sama seperti TokenDataset lama). idx: array indeks baris."""
    idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    if shuffle:
        idx_t = idx_t[torch.randperm(len(idx_t), generator=gen)]
    sdev = store["y"].device
    for i in range(0, len(idx_t), bs):
        b = idx_t[i:i + bs].to(sdev)
        yield dict(
            image_tokens=store["img"][b].to(device, non_blocking=True).float(),
            text_tokens=store["txt"][b].to(device, non_blocking=True).float(),
            text_mask=store["mask"][b].to(device, non_blocking=True),
            tabular=store["tab"][b].to(device, non_blocking=True),
            genre_idx=store["gi"][b].to(device, non_blocking=True),
            target=store["y"][b].to(device, non_blocking=True).float(),
        )


def load_data(verbose=True, spec=None):
    """Muat snapshot + canonical temporal_no_subs split + token cache. Dipanggil sekali.

    spec=None -> DEFAULT_SPEC (snapshot, split, cache yang tetap seperti sebelumnya).
    """
    spec = spec or DEFAULT_SPEC
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _, _ = load_snapshot(verbose=verbose, spec=spec, load_embeddings=False)
    snap_hash = df.attrs["snapshot_hash"]

    split = load_canonical_split(verbose=verbose, spec=spec)
    assert split["snapshot_hash"] == snap_hash, \
        f"split dibuat untuk snapshot {split['snapshot_hash']}, snapshot sekarang {snap_hash}"
    train_idx, val_idx, test_idx = apply_split_to_df(df, split)
    if verbose:
        print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    split_hashes = {
        "train_ids_hash": split["train_ids_hash"],
        "val_ids_hash": split["val_ids_hash"],
        "test_ids_hash": split["test_ids_hash"],
    }

    genres_train = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    n_genres_with_unk = len(genres_train) + 1

    _, scaler = build_tabular_matrix(df.iloc[train_idx], genres_train, fit_scaler=True)
    all_tab, _ = build_tabular_matrix(df, genres_train, scaler=scaler)
    all_tab = np.delete(all_tab, SUBS_IDX, axis=1)          # drop subs
    cont_all = all_tab.astype(np.float32)
    n_cont = cont_all.shape[1]

    genre_to_idx = {g: i + 1 for i, g in enumerate(genres_train)}
    genre_idx = np.array([genre_to_idx.get(g, 0) for g in df["genre"].fillna("")], dtype=np.int64)

    if spec.cache_kind == "store":
        # TokenStore keyed by video_id: rows are looked up in snapshot order, so no order or
        # snapshot-hash assertions are needed (take() raises if a video is missing).
        from early_fusion.datasets.token_store import TokenStore
        tok = TokenStore(spec.cache_dir).take(df["video_id"].tolist())
        img_tokens, txt_tokens, txt_mask = tok["img"], tok["txt"], tok["mask"]
    else:
        img_tokens, txt_tokens, txt_mask, _thumb_ok, index_df, cache_meta = load_cache(spec.cache_dir)
        assert (index_df["video_id"].values == df["video_id"].values).all()
        assert cache_meta["snapshot_hash"] == snap_hash

    targets = df["target"].values.astype(np.float32)

    store = _build_store(img_tokens, txt_tokens, txt_mask, cont_all, genre_idx, targets, device, verbose)

    git_sha, git_dirty = get_git_info()
    if verbose:
        print(f"n_cont={n_cont}, n_genres(+unk)={n_genres_with_unk}, device={device}")
    return dict(
        device=device, store=store,
        train_idx=np.asarray(train_idx), val_idx=np.asarray(val_idx), test_idx=np.asarray(test_idx),
        n_cont=n_cont, n_genres=n_genres_with_unk,
        scaler=scaler, genres_train=genres_train,
        split_hashes=split_hashes, git_sha=git_sha, git_dirty=git_dirty,
        snapshot_hash=snap_hash, spec=spec,
    )


# ----------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------
def variant_flags(variant):
    assert variant in ("full", "no_image", "no_text", "tab_only"), variant
    return variant in ("full", "no_text"), variant in ("full", "no_image")   # use_image, use_text


def build_model(cfg, data, variant="full"):
    use_image, use_text = variant_flags(variant)
    assert cfg["d"] % NHEAD == 0, f"d={cfg['d']} harus habis dibagi nhead={NHEAD}"
    return RATF_M6_Granular_V2(
        image_dim=768, text_dim=512,
        n_continuous=data["n_cont"], n_genres=data["n_genres"],
        d=cfg["d"], nhead=NHEAD, num_layers=cfg["num_layers"],
        dim_ff=int(cfg["d"] * cfg["ff_mult"]),
        n_image_tokens=N_IMAGE_TOKENS, n_text_tokens=N_TEXT_TOKENS,
        dropout=cfg["dropout"], embedding_noise_std=cfg["emb_noise"],
        use_image=use_image, use_text=use_text,
        cross_attn_layers=cfg["cross_attn_layers"],
        gate_mode=cfg["gate_mode"], gate_dropout=cfg["gate_dropout"],
        modality_dropout=cfg["mod_dropout"],
    ).to(data["device"])


def pairwise_rank_loss(p, y):
    """RankNet logistik atas semua pasangan dalam batch dengan y_i != y_j."""
    dy = y[:, None] - y[None, :]
    s = torch.sign(dy)
    mask = s != 0
    if not bool(mask.any()):
        return p.sum() * 0.0
    dp = p[:, None] - p[None, :]
    return F.softplus(-s * dp)[mask].mean()


@torch.no_grad()
def ema_update(ema_model, model, decay, step):
    d = min(decay, (1.0 + step) / (10.0 + step))   # warmup EMA supaya tidak bias ke init acak
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(d).add_(p.detach(), alpha=1.0 - d)
    for be, b in zip(ema_model.buffers(), model.buffers()):
        be.copy_(b)


def make_lr_lambda(cfg, warmup_steps, total_steps):
    schedule, min_frac = cfg["schedule"], cfg["lr_min_frac"]

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "const":
            return 1.0
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return lr_lambda


@torch.no_grad()
def evaluate(model, store, idx, device, collect_gate=False):
    """Return (huber_loss, preds, targets, gate_weights_list). model di-set eval()."""
    model.eval()
    total, n = 0.0, 0
    preds, ys, gate_ws = [], [], []
    for batch in iterate_batches(store, idx, EVAL_BS, device, shuffle=False):
        img = batch["image_tokens"].to(device)
        txt = batch["text_tokens"].to(device)
        mask = batch["text_mask"].to(device)
        tab = batch["tabular"].to(device)
        gi = batch["genre_idx"].to(device)
        y = batch["target"].float().to(device)
        p = model(img, txt, mask, tab, gi, return_reliability=collect_gate)
        if collect_gate:
            _, w = model.get_last_gate_weights()
            if w is not None:
                gate_ws.append(w.cpu())
        total += _huber(p, y).item() * len(y)
        n += len(y)
        preds.append(p.detach().cpu().numpy())
        ys.append(y.cpu().numpy())
    return total / n, np.concatenate(preds), np.concatenate(ys), gate_ws


def summarize_gate(model, gate_ws):
    if not gate_ws:
        return {}
    W = torch.cat(gate_ws, dim=0)
    stats = {}
    for j, name in enumerate(model.mod_names):
        col = W[:, j]
        stats[f"{name}_mean"] = float(col.mean())
        stats[f"{name}_std"] = float(col.std())
        stats[f"{name}_min"] = float(col.min())
        stats[f"{name}_max"] = float(col.max())
    return stats


def metrics(preds, targets):
    sp = spearmanr(preds, targets)[0]
    pos = (targets > 0).astype(int)
    auc = roc_auc_score(pos, preds) if len(np.unique(pos)) == 2 else float("nan")
    return dict(spearman=float(sp), auc=float(auc), mae=float(mean_absolute_error(targets, preds)))


# ----------------------------------------------------------------------------------
# Satu run training
# ----------------------------------------------------------------------------------
def train_one(cfg, seed, data, variant="full", fit="train", fixed_epochs=None,
              eval_test=False, return_state=False, verbose=False, early_abort=None):
    """Latih satu model.

    fit="train"    : latih di train, evaluasi di val (protokol sesuai cfg["schedule"]).
    fit="trainval" : latih di train+val dengan epoch tetap (tanpa val); untuk refit produksi.
                     schedule="const" WAJIB memberi fixed_epochs; "cosine" default = cfg["epochs"].
    eval_test      : hanya True dari run_m6_config.py (ada ledger penjaga). Tuning TIDAK memakainya.
    early_abort    : dict(min_epoch=8, sp_floor=0.24) -> hentikan run bila best val Spearman sejauh ini
                     masih < sp_floor setelah min_epoch (hanya dipakai tuning; fit="train").
    """
    t0 = time.perf_counter()
    cfg = {**DEFAULT_CFG, **cfg}
    assert fit in ("train", "trainval")
    device = data["device"]
    use_image, use_text = variant_flags(variant)

    set_seed(seed)
    model = build_model(cfg, data, variant)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tr_idx = data["train_idx"] if fit == "train" else np.concatenate([data["train_idx"], data["val_idx"]])
    bs = int(cfg["batch_size"])
    store = data["store"]
    gen = torch.Generator().manual_seed(int(seed))      # urutan shuffle deterministik per seed

    cosine = cfg["schedule"] == "cosine"
    protocol = "final" if (cosine or fit == "trainval") else "early_stop"
    if fit == "trainval":
        if fixed_epochs is None:
            if not cosine:
                raise ValueError("fit='trainval' dengan schedule='const' butuh fixed_epochs")
            fixed_epochs = cfg["epochs"]
        n_epochs = int(fixed_epochs)
    else:
        n_epochs = int(cfg["epochs"])

    steps_per_epoch = math.ceil(len(tr_idx) / bs)
    warmup_steps = max(1, int(round(cfg["warmup_epochs"] * steps_per_epoch)))
    total_steps = n_epochs * steps_per_epoch

    gate_params = list(model.gate.parameters()) if model.gate is not None else []
    gate_ids = {id(p) for p in gate_params}
    base_params = [p for p in model.parameters() if id(p) not in gate_ids]
    groups = [{"params": base_params, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}]
    if gate_params:
        groups.append({"params": gate_params, "lr": cfg["lr"] * cfg["gate_lr_mult"], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(cfg, warmup_steps, total_steps))

    ema_model = None
    if cfg["ema_decay"] > 0:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    eval_model = ema_model if ema_model is not None else model

    lam = float(cfg["rank_lambda"])
    step_ctr = {"n": 0}

    def train_epoch():
        model.train()
        total, n = torch.zeros((), device=device), 0
        for batch in iterate_batches(store, tr_idx, bs, device, shuffle=True, gen=gen):
            img = batch["image_tokens"].to(device)
            txt = batch["text_tokens"].to(device)
            mask = batch["text_mask"].to(device)
            tab = batch["tabular"].to(device)
            gi = batch["genre_idx"].to(device)
            y = batch["target"].float().to(device)

            opt.zero_grad(set_to_none=True)
            p = model(img, txt, mask, tab, gi)
            loss = _huber(p, y)
            if lam > 0:
                loss = loss + lam * pairwise_rank_loss(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            step_ctr["n"] += 1
            if ema_model is not None:
                ema_update(ema_model, model, cfg["ema_decay"], step_ctr["n"])
            total += loss.detach() * len(y)          # tanpa .item() per step -> tanpa sinkronisasi GPU
            n += len(y)
        return float(total) / n

    history = []
    best_val, best_epoch, stale, best_state, best_pack = float("inf"), 0, 0, None, None
    last_pack = None
    epochs_run = 0
    aborted = False
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_epoch()
        epochs_run = epoch
        if fit != "train":
            continue
        va_loss, va_preds, va_t, va_gate = evaluate(eval_model, store, data["val_idx"], device, collect_gate=True)
        va_sp = float(spearmanr(va_preds, va_t)[0])
        history.append(dict(epoch=epoch, tr_loss=tr_loss, va_loss=va_loss, va_sp=va_sp))
        last_pack = (va_loss, va_preds, va_t, va_gate)
        if verbose:
            print(f"ep {epoch:3d} | tr={tr_loss:.4f} va={va_loss:.4f} sp={va_sp:.4f} | {time.perf_counter() - t0:.0f}s")
        if early_abort and epoch >= early_abort["min_epoch"]:
            best_sp = max((h["va_sp"] for h in history if np.isfinite(h["va_sp"])), default=-1.0)
            if best_sp < early_abort["sp_floor"]:
                aborted = True
                break
        if protocol == "early_stop":
            if va_loss < best_val:
                best_val, best_epoch, stale = va_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
                best_pack = last_pack
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break

    out = dict(
        cfg=cfg, seed=seed, variant=variant, fit=fit, protocol=protocol,
        n_params=int(n_params), epochs_run=int(epochs_run), aborted=bool(aborted),
        n_train=int(len(tr_idx)), use_image=use_image, use_text=use_text,
    )

    if fit == "train":
        if protocol == "early_stop":
            if best_state is None:   # val loss tidak pernah membaik (mis. NaN): pakai keadaan terakhir
                best_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
                best_pack, best_epoch = last_pack, epochs_run
            eval_model.load_state_dict(best_state)
            va_loss, va_preds, va_t, va_gate = best_pack
            out["best_epoch"] = int(best_epoch)
        else:
            va_loss, va_preds, va_t, va_gate = last_pack
            out["best_epoch"] = int(n_epochs)       # protokol "final": bobot akhir
        m = metrics(va_preds, va_t)
        out.update(val_spearman=m["spearman"], val_auc=m["auc"], val_mae=m["mae"], val_loss=float(va_loss),
                   val_preds=va_preds, val_targets=va_t, gate_stats=summarize_gate(model, va_gate),
                   history=history)
    else:
        out["best_epoch"] = int(n_epochs)

    if eval_test:
        _, te_preds, te_t, _ = evaluate(eval_model, store, data["test_idx"], device)
        tm = metrics(te_preds, te_t)
        out.update(test_spearman=tm["spearman"], test_auc=tm["auc"], test_mae=tm["mae"],
                   test_preds=te_preds, test_targets=te_t)

    if return_state:
        out["state_dict"] = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}

    out["seconds"] = float(time.perf_counter() - t0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
