"""M4a training: RATF pooled (1 image token + 1 text token + 12 tabular tokens = 15 tokens).

Same as train_m4.py except:
- Import RATF_M4a (instead of RATF_M4).
- Output to m4a_results.jsonl.
- Checkpoint to m4a_seed{SEED}_{variant}.pt.
- Variant prefix "m4a_".
- No --pooled flag (it is pooled by design).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import argparse
import random
import hashlib
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_absolute_error

from features.target import compute_target
from models.dataset import (build_tabular_matrix,
                            TABULAR_LOG_COLS, TABULAR_NUMERIC_COLS, TABULAR_BOOL_COLS)
from early_fusion.datasets.token_cache import load_cache
from early_fusion.datasets.token_dataset import TokenDataset
from early_fusion.models.ratf_pooled import RATF_M4a


SEED = int(os.environ.get("SEED", 42))
SNAPSHOT = Path("data_snapshots/snapshot.parquet")
CACHE_DIR = "data_snapshots/token_cache"
RESULTS = Path("early_fusion/results/m4a_results.jsonl")
CHECKPOINT_DIR = Path("early_fusion/models/checkpoints")
SNAPSHOT_HASH = "c14dba895034fc4c"

# Hyperparameter M4a (sama dengan M4 untuk perbandingan apples-to-apples)
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


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def hash_df(df):
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def hash_ids(ids):
    return hashlib.sha256(",".join(sorted(map(str, ids))).encode()).hexdigest()[:16]


def get_git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return sha, bool(dirty)
    except Exception:
        return "unknown", False


def prepare_tabular(df, train_idx, genres_train):
    _, scaler = build_tabular_matrix(df.iloc[train_idx], genres_train, fit_scaler=True)
    all_tab, _ = build_tabular_matrix(df, genres_train, scaler=scaler)

    n_log = len(TABULAR_LOG_COLS)
    n_num = len(TABULAR_NUMERIC_COLS)
    n_bool = len(TABULAR_BOOL_COLS)
    n_genres = len(genres_train)
    expected = n_log + n_num + n_bool + n_genres
    assert all_tab.shape[1] == expected

    cont = all_tab[:, :n_log + n_num + n_bool].astype(np.float32)
    genre_oh = all_tab[:, n_log + n_num + n_bool:]

    has_genre = genre_oh.sum(axis=1) > 0
    genre_idx = np.zeros(len(df), dtype=np.int64)
    genre_idx[has_genre] = np.argmax(genre_oh[has_genre], axis=1) + 1
    return cont, genre_idx, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    args = parser.parse_args()

    use_image = not args.no_image
    use_text = not args.no_text
    if not use_image and not use_text:
        variant = "m4a_tabular_only"
    elif not use_image:
        variant = "m4a_no_image"
    elif not use_text:
        variant = "m4a_no_text"
    else:
        variant = "m4a_full"

    set_seed(SEED)
    git_sha, git_dirty = get_git_info()
    print(f"git: {git_sha[:8]} dirty={git_dirty}")
    print(f"variant: {variant} (use_image={use_image}, use_text={use_text})")

    df = pd.read_parquet(SNAPSHOT)
    computed_hash = hash_df(df)
    assert computed_hash == SNAPSHOT_HASH, f"snapshot hash mismatch"
    print(f"snapshot: {len(df)} rows, hash={computed_hash}")

    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    train_idx = np.where(df["split"].values == "train")[0]
    val_idx = np.where(df["split"].values == "val")[0]
    test_idx = np.where(df["split"].values == "test")[0]
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    split_hashes = {
        "train_ids_hash": hash_ids(df.iloc[train_idx]["video_id"].tolist()),
        "val_ids_hash": hash_ids(df.iloc[val_idx]["video_id"].tolist()),
        "test_ids_hash": hash_ids(df.iloc[test_idx]["video_id"].tolist()),
    }
    print(f"split hashes: {split_hashes}")

    genres_train = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())
    cont_all, genre_idx, tab_scaler = prepare_tabular(df, train_idx, genres_train)
    n_cont = cont_all.shape[1]
    n_genres_with_unk = len(genres_train) + 1
    print(f"n_cont={n_cont}, n_genres(+unk)={n_genres_with_unk}")

    img_tokens, txt_tokens, txt_mask, thumb_ok, index_df, cache_meta = load_cache(CACHE_DIR)
    assert (index_df["video_id"].values == df["video_id"].values).all()
    assert cache_meta["snapshot_hash"] == SNAPSHOT_HASH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    targets = df["target"].values.astype(np.float32)
    video_ids = df["video_id"].values

    full_ds = TokenDataset(img_tokens, txt_tokens, txt_mask,
                           cont_all, genre_idx, targets, video_ids)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=BATCH_SIZE, num_workers=0)

    model = RATF_M4a(
        image_dim=768, text_dim=512,
        n_continuous=n_cont, n_genres=n_genres_with_unk,
        d=D, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
        dropout=DROPOUT, embedding_noise_std=EMB_NOISE,
        use_image=use_image, use_text=use_text,
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

    def run_epoch(loader, train=False):
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
                p = model(img, txt, mask, tab, gi)
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

    print()
    print(f"=== M4a val (seed={SEED}, variant={variant}) ===")
    print(f"val Spearman: {val_sp:.4f}")
    print(f"val AUC:      {val_auc:.4f}")
    print(f"best epoch:   {best_epoch}, params: {n_params:,}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"m4a_seed{SEED}_{variant}.pt"
    torch.save({
        "model_state_dict": best_state,
        "variant": variant, "use_image": use_image, "use_text": use_text,
        "n_continuous": n_cont, "n_genres": n_genres_with_unk,
        "d": D, "nhead": NHEAD, "num_layers": NUM_LAYERS, "dim_ff": DIM_FF,
        "scaler": tab_scaler, "genres_train": genres_train,
        "seed": SEED, "best_epoch": best_epoch, "best_val_loss": best_val,
        "val_spearman": float(val_sp),
        "snapshot_hash": SNAPSHOT_HASH, "split_hashes": split_hashes,
        "git_sha": git_sha, "git_dirty": git_dirty,
    }, ckpt_path)

    test_metrics = {}
    if args.eval_test:
        _, test_preds, test_targets = run_epoch(test_loader, train=False)
        test_sp = spearmanr(test_preds, test_targets)[0]
        has_both_t = len(np.unique((test_targets > 0).astype(int))) == 2
        test_auc = roc_auc_score((test_targets > 0).astype(int), test_preds) if has_both_t else float("nan")
        test_mae = mean_absolute_error(test_targets, test_preds)
        print()
        print(f"=== M4a TEST (seed={SEED}, variant={variant}) ===")
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
            "model": "M4a_token_early_fusion_pooled",
            "variant": variant,
            "seed": SEED,
            "val_spearman": float(val_sp),
            "val_auc": float(val_auc),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "n_params": int(n_params),
            "n_continuous": int(n_cont),
            "n_genres": int(n_genres_with_unk),
            "d": D, "nhead": NHEAD, "num_layers": NUM_LAYERS, "dim_ff": DIM_FF,
            "lr": LR, "batch_size": BATCH_SIZE, "dropout": DROPOUT,
            "warmup_steps": WARMUP_STEPS, "grad_clip": GRAD_CLIP,
            "use_image": use_image, "use_text": use_text,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
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