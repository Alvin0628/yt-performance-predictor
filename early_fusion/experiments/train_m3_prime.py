"""M3' — late fusion TUNED on canonical temporal split (drop subs).

Split protocol:
- Canonical split dari early_fusion/splits/temporal_no_subs.json (temporal).
- Drop `subscriber_count_at_upload` dari fitur tabular (leakage removal).
- Split di-load dari file, bukan dihitung ulang. Verifikasi hash.
- Konsisten dengan M4a/M5/M6 — semua pakai split file yang sama.

Perbedaan dari train_m3_prime.py lama:
- Import dari _common (set_seed, load_snapshot, prepare_tabular_drop_subs).
- Import dari load_split (load_canonical_split, apply_split_to_df).
- Drop subs via prepare_tabular_drop_subs.
- Catat split_mode + drop_subs di JSONL.

Model, training protocol, hyperparameter: IDENTIK.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                              roc_auc_score)

from features.target import invert_target
from models.dataset import VideoDataset
from models.late_fusion_model import LateFusionModel

from early_fusion.experiments._common import (
    set_seed, hash_ids, get_git_info,
    load_snapshot, prepare_tabular_drop_subs,
)
from early_fusion.splits.load_split import load_canonical_split, apply_split_to_df


# ==== Config ====
SEED = int(os.environ.get("SEED", 42))
SNAPSHOT_HASH = "c14dba895034fc4c"

CHECKPOINT_PATH_TMPL = "early_fusion/models/checkpoints/m3_prime_seed{seed}.pt"
RESULTS_PATH = Path("early_fusion/results/m3_prime_results.jsonl")

IMAGE_ENCODER = "clip_b32"
TEXT_ENCODER = "clip"

BATCH_SIZE = 64
EPOCHS = 400
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
EMBEDDING_NOISE_STD = 0.02
EARLY_STOP_PATIENCE = 15
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(
            batch["image_embedding"].to(device),
            batch["text_embedding"].to(device),
            batch["tabular"].to(device),
        )
        loss = loss_fn(pred, batch["target"].float().to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(batch["target"])
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    preds_all, targets_all = [], []
    for batch in loader:
        pred = model(
            batch["image_embedding"].to(device),
            batch["text_embedding"].to(device),
            batch["tabular"].to(device),
        )
        loss = loss_fn(pred, batch["target"].float().to(device))
        total += loss.item() * len(batch["target"])
        preds_all.extend(pred.cpu().numpy().tolist())
        targets_all.extend(batch["target"].numpy().tolist())
    return total / len(loader.dataset), np.array(preds_all), np.array(targets_all)


def main():
    set_seed(SEED)
    git_sha, git_dirty = get_git_info()
    print(f"git: {git_sha[:8]} dirty={git_dirty}")
    print(f"seed={SEED}, lr={LEARNING_RATE}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}, warmup={WARMUP_FRAC}, grad_clip={GRAD_CLIP}")
    print(f"split_mode: temporal_no_subs, drop_subs: True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---------- Load snapshot ----------
    df, image_embeddings, text_embeddings = load_snapshot()

    # ---------- Load canonical split ----------
    split = load_canonical_split(verbose=True)
    train_idx, val_idx, test_idx = apply_split_to_df(df, split)
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    split_hashes = {
        "train_ids_hash": split["train_ids_hash"],
        "val_ids_hash": split["val_ids_hash"],
        "test_ids_hash": split["test_ids_hash"],
    }

    # ---------- Build tabular features (drop subs) ----------
    train_tab, val_tab, test_tab, scaler, genre_categories, tabular_dim = \
        prepare_tabular_drop_subs(df, train_idx, val_idx, test_idx)
    print(f"n_genres: {len(genre_categories)}")

    def make_dataset(idx, tabular):
        sub = df.iloc[idx]
        return VideoDataset(
            image_embeddings[idx], text_embeddings[idx], tabular,
            sub["target"].values, sub["video_id"].values,
        )

    train_ds = make_dataset(train_idx, train_tab)
    val_ds = make_dataset(val_idx, val_tab)
    test_ds = make_dataset(test_idx, test_tab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    # ---------- Model ----------
    model = LateFusionModel(
        image_dim=image_embeddings.shape[1],
        text_dim=text_embeddings.shape[1],
        tabular_dim=tabular_dim,
        dropout=DROPOUT,
        embedding_noise_std=EMBEDDING_NOISE_STD,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    # ---------- Optimizer ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(int(total_steps * WARMUP_FRAC), 1)
    print(f"total_steps={total_steps}, warmup_steps={warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = torch.nn.HuberLoss()

    checkpoint_path = CHECKPOINT_PATH_TMPL.format(seed=SEED)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # ---------- Training loop ----------
    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, loss_fn, device)
        val_spearman, _ = spearmanr(val_preds, val_targets)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "genre_categories": genre_categories,
                "scaler": scaler,
                "image_dim": image_embeddings.shape[1],
                "text_dim": text_embeddings.shape[1],
                "tabular_dim": tabular_dim,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_spearman_raw": val_spearman,
                "image_encoder": IMAGE_ENCODER,
                "text_encoder": TEXT_ENCODER,
                "seed": SEED,
                "snapshot_hash": SNAPSHOT_HASH,
                "split_mode": "temporal_no_subs",
                "drop_subs": True,
                "split_hashes": split_hashes,
                "git_sha": git_sha,
                "git_dirty": git_dirty,
            }, checkpoint_path)
            print(f"ep {epoch:3d} | tr={train_loss:.4f} va={val_loss:.4f} "
                  f"sp={val_spearman:.4f} *")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"early stop ep {epoch}")
                break

    # ---------- Load best checkpoint ----------
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ---------- Val metrics ----------
    _, val_preds, val_targets = evaluate(model, val_loader, loss_fn, device)
    val_sp = spearmanr(val_preds, val_targets)[0]
    has_both = len(np.unique((val_targets > 0).astype(int))) == 2
    val_auc = roc_auc_score((val_targets > 0).astype(int), val_preds) if has_both else float("nan")

    print()
    print(f"=== M3' val (seed={SEED}) ===")
    print(f"val Spearman: {val_sp:.4f}")
    print(f"val AUC:      {val_auc:.4f}")
    print(f"best epoch:   {checkpoint['epoch']}, params: {n_params:,}")

    # ---------- Test metrics ----------
    test_loss, test_preds, test_targets = evaluate(model, test_loader, loss_fn, device)
    test_sp = spearmanr(test_preds, test_targets)[0]
    has_both_t = len(np.unique((test_targets > 0).astype(int))) == 2
    test_auc = roc_auc_score((test_targets > 0).astype(int), test_preds) if has_both_t else float("nan")
    test_mae_target = mean_absolute_error(test_targets, test_preds)
    test_sub = df.iloc[test_idx]
    predicted_views = invert_target(test_preds, test_sub["trailing_avg_views"].values)
    test_mae_views = mean_absolute_error(test_sub["views"].values, predicted_views)
    test_mape = mean_absolute_percentage_error(test_sub["views"].values, predicted_views)
    test_rmse_views = np.sqrt(np.mean((test_sub["views"].values - predicted_views) ** 2))

    print()
    print(f"=== M3' TEST (seed={SEED}) ===")
    print(f"test Spearman: {test_sp:.4f}")
    print(f"test AUC:      {test_auc:.4f}")
    print(f"test MAE (target): {test_mae_target:.4f}")

    # ---------- Save results ----------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps({
            "model": "M3_prime_late_fusion_tuned",
            "seed": SEED,
            "split_mode": "temporal_no_subs",
            "drop_subs": True,
            "val_spearman": float(val_sp),
            "val_auc": float(val_auc),
            "val_loss": float(checkpoint["val_loss"]),
            "test_spearman": float(test_sp),
            "test_auc": float(test_auc),
            "test_mae_target": float(test_mae_target),
            "test_mae_views": float(test_mae_views),
            "test_mape_views": float(test_mape),
            "test_rmse_views": float(test_rmse_views),
            "best_epoch": int(checkpoint["epoch"]),
            "n_params": int(n_params),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "lr": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "warmup_frac": WARMUP_FRAC, "grad_clip": GRAD_CLIP,
            "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
            "snapshot_hash": SNAPSHOT_HASH,
            "split_hashes": split_hashes,
            "git_sha": git_sha, "git_dirty": git_dirty,
        }) + "\n")
    print(f"\nsaved: {RESULTS_PATH}")
    print(f"saved: {checkpoint_path}")


if __name__ == "__main__":
    main()