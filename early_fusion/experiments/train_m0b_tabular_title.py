"""M0b baseline: tabular + title features, tanpa image/text.

Tujuan: mengisolasi efek 8 fitur judul dari efek image+text pada gap M3-M0.

- Import `run_variant` dan `build_variant_tabular_matrix` dari teman apa adanya.
- Config M0b: include_title=True, include_video=False, use_image=False, use_text=False.
- Baca dari snapshot.parquet, split dari kolom 'split'.
- Output ke early_fusion/results/m0b_results.jsonl.
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
from models.train import set_seed, SEED

import models.ablation_modalities as am
from models.ablation_modalities import run_variant


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
RESULTS = Path("early_fusion/results/m0b_results.jsonl")
CHECKPOINT_DIR = "early_fusion/models/checkpoints/m0b"
SNAPSHOT_HASH = "c14dba895034fc4c"

# Config M0b: tabular + title features, tanpa image/text.
# Ini sama dengan config yang akan dipakai kalau temanmu menambahkan variant baru
# di ablation_modalities.py — kita definisikan di luar supaya tidak mengubah file teman.
M0B_NAME = "m0b_tabular_title"
M0B_CONFIG = {
    "include_title": True,   # ← +8 fitur judul (6 numerik + 2 boolean)
    "include_video": False,
    "use_image": False,
    "use_text": False,
}


def parse_vector(s):
    if isinstance(s, str):
        return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(s, dtype=np.float32)


def main():
    set_seed(SEED)
    # Redirect checkpoint ke folder kita — jangan sentuh models/checkpoints/ablations/
    am.CHECKPOINT_DIR = CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df = pd.read_parquet(SNAPSHOT)
    print(f"loaded {len(df)} rows from {SNAPSHOT}")

    # Tetap baca embeddings (meski tidak dipakai) supaya signature run_variant cocok.
    # use_image=False dan use_text=False → embeddings tidak akan diteruskan ke dataset.
    image_emb = np.stack(df["image_embedding"].apply(parse_vector).values)
    text_emb = np.stack(df["text_embedding"].apply(parse_vector).values)
    df = df.drop(columns=["image_embedding", "text_embedding"]).reset_index(drop=True)

    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    # Split dari kolom snapshot — konsisten dengan M0/M3
    train_idx = np.where(df["split"].values == "train")[0]
    val_idx = np.where(df["split"].values == "val")[0]
    test_idx = np.where(df["split"].values == "test")[0]
    print(f"split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Jalankan M0b melalui run_variant milik teman (bukan full_fusion → AblationModel)
    result = run_variant(
        M0B_NAME, M0B_CONFIG, df,
        train_idx, val_idx, test_idx,
        image_emb, text_emb, device,
    )

    print()
    print(f"=== M0b tabular + title (seed={SEED}) ===")
    print(f"Spearman:   {result['Spearman']:.4f}")
    print(f"AUC:        {result['AUC']:.4f}")
    print(f"RMSE(views):{result['RMSE (views)']:>15,.0f}")
    print(f"MAE(views): {result['MAE (views)']:>15,.0f}")
    print(f"MAPE:       {result['MAPE (views)']:.2%}")
    print(f"best_epoch: {result['_best_epoch']}, tabular_dim: {result['_tabular_dim']}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps({
            "model": "M0b_tabular_plus_title", "seed": SEED,
            "variant": M0B_NAME,
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