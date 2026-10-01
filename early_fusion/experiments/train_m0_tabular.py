"""M0 baseline: tabular-only, adapting models/ablation_modalities.py
friend's (VARIANTS['tabular_only']) so that:
  - read from snapshot.parquet (instead of PostgreSQL)
  - write output to early_fusion/
  - use the SAME split as M3 (snapshot `split` column)

Model, scaler, and build_variant_tabular_matrix are NOT changed — exactly the friend's code.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import numpy as np
import pandas as pd
import torch

from features.target import compute_target
from models.train import (
    set_seed, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EARLY_STOP_PATIENCE, WEIGHT_DECAY, DROPOUT,
    EMBEDDING_NOISE_STD, SEED,
)
import models.ablation_modalities as am
from models.ablation_modalities import run_variant, VARIANTS


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
RESULTS = Path("early_fusion/results/m0_results.jsonl")
CHECKPOINT_DIR = "early_fusion/models/checkpoints/m0"
VARIANT = "tabular_only"
SNAPSHOT_HASH = "c14dba895034fc4c"


def parse_vector(s):
    if isinstance(s, str):
        return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(s, dtype=np.float32)


def main():
    set_seed(SEED)
    # Redirect checkpoint output to our folder, do not touch models/checkpoints/ablations/
    am.CHECKPOINT_DIR = CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df = pd.read_parquet(SNAPSHOT)
    print(f"loaded {len(df)} rows from {SNAPSHOT}")

    image_emb = np.stack(df["image_embedding"].apply(parse_vector).values)
    text_emb = np.stack(df["text_embedding"].apply(parse_vector).values)
    df = df.drop(columns=["image_embedding", "text_embedding"]).reset_index(drop=True)
    print(f"image_emb: {image_emb.shape}, text_emb: {text_emb.shape}")

    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    # Use the split column from the snapshot (do not recalculate) — consistent with M3
    train_idx = np.where(df["split"].values == "train")[0]
    val_idx = np.where(df["split"].values == "val")[0]
    test_idx = np.where(df["split"].values == "test")[0]
    print(f"split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    result = run_variant(
        VARIANT, VARIANTS[VARIANT], df,
        train_idx, val_idx, test_idx,
        image_emb, text_emb, device,
    )

    print()
    print(f"=== M0 tabular-only (seed={SEED}) ===")
    print(f"Spearman:   {result['Spearman']:.4f}")
    print(f"AUC:        {result['AUC']:.4f}")
    print(f"RMSE(views):{result['RMSE (views)']:>15,.0f}")
    print(f"MAE(views): {result['MAE (views)']:>15,.0f}")
    print(f"MAPE:       {result['MAPE (views)']:.2%}")
    print(f"best_epoch: {result['_best_epoch']}, tabular_dim: {result['_tabular_dim']}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps({
            "model": "M0_tabular_only", "seed": SEED,
            "variant": VARIANT,
            "test_spearman": float(result["Spearman"]),
            "test_auc": float(result["AUC"]),
            "test_rmse_views": float(result["RMSE (views)"]),
            "test_mae_views": float(result["MAE (views)"]),
            "test_mape_views": float(result["MAPE (views)"]),
            "best_epoch": int(result["_best_epoch"]),
            "tabular_dim": int(result["_tabular_dim"]),
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "snapshot_hash": SNAPSHOT_HASH,
        }) + "\n")
    print(f"\nsaved: {RESULTS}")


if __name__ == "__main__":
    main()