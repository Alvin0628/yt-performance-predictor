"""M6 (granular) training on canonical temporal_no_subs split.

Model: RATF_M6_Granular (95 token: 50 image + 32 text + 12 tabular + 1 CLS).
Split: canonical temporal_no_subs (drop subs, dari early_fusion/splits/).
Output: m6_temporal_granular_results.jsonl
Checkpoint: m6_temporal_granular_seed{SEED}_{variant}.pt

Perbedaan dari train_m6_unseen.py:
- Split: temporal canonical (bukan unseen).
- Drop subs dari tabular (leakage removal).
- Model: RATF_M6_Granular (bukan RATF_M6 pooled).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_absolute_error

from models.dataset import build_tabular_matrix
from early_fusion.datasets.token_cache import load_cache
from early_fusion.datasets.token_dataset import TokenDataset
from early_fusion.models.ratf_m6_granular import RATF_M6_Granular
from early_fusion.experiments._common import (
    set_seed, hash_ids, get_git_info, load_snapshot,
)
from early_fusion.splits.load_split import load_canonical_split, apply_split_to_df


SEED = int(os.environ.get("SEED", 42))
SNAPSHOT_HASH = "c14dba895034fc4c"
CACHE_DIR = "data_snapshots/token_cache"
RESULTS = Path("early_fusion/results/m6_temporal_granular_results.jsonl")
CHECKPOINT_DIR = Path("early_fusion/models/checkpoints")

D = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 512
DROPOUT = 0.2
EMB_NOISE = 0.02
LR = 1e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 10
WARMUP_STEPS = 300
GRAD_CLIP = 1.0
CROSS_ATTN_LAYERS = 1
N_IMAGE_TOKENS = 50
N_TEXT_TOKENS = 32
SUBS_IDX = 0  # subscriber_count_at_upload = kolom log pertama di build_tabular_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    args = parser.parse_args()

    use_image = not args.no_image
    use_text = not args.no_text
    if not use_image and not use_text:
        variant = "m6_temporal_granular_tabular_only"
    elif not use_image:
        variant = "m6_temporal_granular_no_image"
    elif not use_text:
        variant = "m6_temporal_granular_no_text"
    else:
        variant = "m6_temporal_granular_full"

    set_seed(SEED)
    git_sha, git_dirty = get_git_info()
    print(f"git: {git_sha[:8]} dirty={git_dirty}")
    print(f"variant: {variant} (use_image={use_image}, use_text={use_text})")
    print(f"split: canonical temporal_no_subs, drop_subs: True")
    print(f"model: RATF_M6_Granular (50 image + 32 text + 12 tabular + 1 CLS)")
    print(f"cross_attn_layers: {CROSS_ATTN_LAYERS}, reliability: ON")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---------- Load snapshot ----------
    df, _, _ = load_snapshot()

    # ---------- Load canonical split ----------
    split = load_canonical_split(verbose=True)
    train_idx, val_idx, test_idx = apply_split_to_df(df, split)
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    split_hashes = {
        "train_ids_hash": split["train_ids_hash"],
        "val_ids_hash": split["val_ids_hash"],
        "test_ids_hash": split["test_ids_hash"],
    }

    # ---------- Build tabular (drop subs) ----------
    # Fit scaler di train. Build untuk semua df (full 11285) supaya
    # TokenDataset punya cont_all untuk semua baris. Drop subs = kolom 0.
    genres_train = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    n_genres_with_unk = len(genres_train) + 1

    _, scaler = build_tabular_matrix(
        df.iloc[train_idx], genres_train, fit_scaler=True)
    all_tab, _ = build_tabular_matrix(df, genres_train, scaler=scaler)
    all_tab = np.delete(all_tab, SUBS_IDX, axis=1)  # drop subscriber_count
    cont_all_full = all_tab.astype(np.float32)
    n_cont = cont_all_full.shape[1]
    print(f"n_cont={n_cont}, n_genres(+unk)={n_genres_with_unk} (drop subs from {n_cont + 1})")

    # ---------- Genre index ----------
    genre_to_idx = {g: i + 1 for i, g in enumerate(genres_train)}  # 0 = unk
    genre_idx = np.array(
        [genre_to_idx.get(g, 0) for g in df["genre"].fillna("")], dtype=np.int64
    )

    # ---------- Load token cache ----------
    img_tokens, txt_tokens, txt_mask, thumb_ok, index_df, cache_meta = load_cache(CACHE_DIR)
    assert (index_df["video_id"].values == df["video_id"].values).all()
    assert cache_meta["snapshot_hash"] == SNAPSHOT_HASH

    # ---------- Targets & ids ----------
    targets = df["target"].values.astype(np.float32)
    video_ids = df["video_id"].values

    # ---------- Dataset ----------
    full_ds = TokenDataset(img_tokens, txt_tokens, txt_mask,
                           cont_all_full, genre_idx, targets, video_ids)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=BATCH_SIZE, num_workers=0)

    # ---------- Model ----------
    model = RATF_M6_Granular(
        image_dim=768, text_dim=512,
        n_continuous=n_cont, n_genres=n_genres_with_unk,
        d=D, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
        n_image_tokens=N_IMAGE_TOKENS, n_text_tokens=N_TEXT_TOKENS,
        dropout=DROPOUT, embedding_noise_std=EMB_NOISE,
        use_image=use_image, use_text=use_text,
        cross_attn_layers=CROSS_ATTN_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.HuberLoss()

    def run_epoch(loader, train=False, collect_rel=False):
        model.train() if train else model.eval()
        total, n = 0.0, 0
        preds, ys = [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                img = batch["image_tokens"].to(device)
                txt = batch["text_tokens"].to(device)
                mask = batch["text_mask"].to(device)
                tab = batch["tabular"].to(device)
                gi = batch["genre_idx"].to(device)
                y = batch["target"].float().to(device)

                if train:
                    opt.zero_grad()
                p = model(img, txt, mask, tab, gi, return_reliability=collect_rel)
                loss = loss_fn(p, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    sched.step()
                total += loss.item() * len(y)
                n += len(y)
                preds.extend(p.detach().cpu().numpy().tolist())
                ys.extend(y.cpu().numpy().tolist())
        return total / n, np.array(preds), np.array(ys)

    best_val = float("inf"); best_state = None; best_epoch = 0; stale = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, _, _ = run_epoch(train_loader, train=True)
        va_loss, va_preds, va_targets = run_epoch(val_loader, train=False)
        va_sp = spearmanr(va_preds, va_targets)[0]

        if va_loss < best_val:
            best_val = va_loss; best_epoch = epoch; stale = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"ep {epoch:3d} | tr={tr_loss:.4f} va={va_loss:.4f} sp={va_sp:.4f} *")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop ep {epoch}")
                break

    model.load_state_dict(best_state)
    _, val_preds, val_targets = run_epoch(val_loader, train=False)
    val_sp = spearmanr(val_preds, val_targets)[0]
    has_both = len(np.unique((val_targets > 0).astype(int))) == 2
    val_auc = roc_auc_score((val_targets > 0).astype(int), val_preds) if has_both else float("nan")

    # Reliability collect
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image_tokens"].to(device)
            txt = batch["text_tokens"].to(device)
            mask = batch["text_mask"].to(device)
            tab = batch["tabular"].to(device)
            gi = batch["genre_idx"].to(device)
            _ = model(img, txt, mask, tab, gi, return_reliability=True)
            break
    rel_stats = model.get_last_reliability()

    print()
    print(f"=== M6 temporal granular val (seed={SEED}, variant={variant}) ===")
    print(f"val Spearman: {val_sp:.4f}")
    print(f"val AUC:      {val_auc:.4f}")
    print(f"best epoch:   {best_epoch}, params: {n_params:,}")
    print(f"reliability:")
    for k, v in rel_stats.items():
        print(f"  {k}: {v:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"m6_temporal_granular_seed{SEED}_{variant}.pt"
    torch.save({
        "model_state_dict": best_state,
        "variant": variant, "use_image": use_image, "use_text": use_text,
        "n_continuous": n_cont, "n_genres": n_genres_with_unk,
        "d": D, "nhead": NHEAD, "num_layers": NUM_LAYERS, "dim_ff": DIM_FF,
        "cross_attn_layers": CROSS_ATTN_LAYERS,
        "scaler": scaler, "genres_train": genres_train,
        "seed": SEED, "best_epoch": best_epoch, "best_val_loss": best_val,
        "val_spearman": float(val_sp),
        "reliability_stats": rel_stats,
        "snapshot_hash": SNAPSHOT_HASH, "split_hashes": split_hashes,
        "git_sha": git_sha, "git_dirty": git_dirty,
        "split_mode": "temporal_no_subs", "drop_subs": True,
    }, ckpt_path)

    test_metrics = {}
    if args.eval_test:
        _, test_preds, test_targets = run_epoch(test_loader, train=False)
        test_sp = spearmanr(test_preds, test_targets)[0]
        has_both_t = len(np.unique((test_targets > 0).astype(int))) == 2
        test_auc = roc_auc_score((test_targets > 0).astype(int), test_preds) if has_both_t else float("nan")
        test_mae = mean_absolute_error(test_targets, test_preds)
        print()
        print(f"=== M6 temporal granular TEST (seed={SEED}, variant={variant}) ===")
        print(f"test Spearman: {test_sp:.4f}")
        print(f"test AUC:      {test_auc:.4f}")
        print(f"test MAE:      {test_mae:.4f}")
        test_metrics = {
            "test_spearman": float(test_sp),
            "test_auc": float(test_auc),
            "test_mae": float(test_mae),
        }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps({
            "model": "M6_granular_temporal_no_subs",
            "variant": variant,
            "seed": SEED,
            "split_mode": "temporal_no_subs", "drop_subs": True,
            "cross_attn_layers": CROSS_ATTN_LAYERS,
            "val_spearman": float(val_sp),
            "val_auc": float(val_auc),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "n_params": int(n_params),
            "n_cont": int(n_cont), "n_genres": int(n_genres_with_unk),
            "use_image": use_image, "use_text": use_text,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "reliability_stats": rel_stats,
            "snapshot_hash": SNAPSHOT_HASH,
            "split_hashes": split_hashes,
            "git_sha": git_sha, "git_dirty": git_dirty,
            "test_eval": bool(args.eval_test),
            **test_metrics,
        }) + "\n")
    print(f"\nsaved: {RESULTS}")
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()