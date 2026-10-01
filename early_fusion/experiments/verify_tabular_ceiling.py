"""Verifikasi: apakah M4 tabular-only 0.4003 wajar, atau ada kebocoran?

Cek murah:
1. Ridge regression di tabular (linear baseline).
2. HistGradientBoostingRegressor (non-linear tabular baseline).
3. Mean-reversion predictor (-log1p(trailing_avg_views)) — murni, tanpa fitting.

Interpretasi:
- Ridge/GBDT val ≈ 0.38–0.41 → M4 tabular wajar, baseline under-trained.
- GBDT ≈ 0.28 dan M4 jauh di atasnya → curigai kebocoran.
- Mean-reversion tinggi (0.35+) → sebagian besar sinyal dari trailing saja.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from features.target import compute_target
from models.dataset import build_tabular_matrix


SNAPSHOT = Path("data_snapshots/snapshot.parquet")


def main():
    df = pd.read_parquet(SNAPSHOT)
    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    train_idx = np.where(df["split"].values == "train")[0]
    val_idx = np.where(df["split"].values == "val")[0]
    print(f"split: train={len(train_idx)} val={len(val_idx)}")

    genres_train = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())

    tab_tr, sc = build_tabular_matrix(df.iloc[train_idx], genres_train, fit_scaler=True)
    tab_va, _ = build_tabular_matrix(df.iloc[val_idx], genres_train, scaler=sc)
    y_tr = df["target"].values[train_idx]
    y_va = df["target"].values[val_idx]

    print(f"tabular dim: {tab_tr.shape[1]}")
    print()

    # 1. Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(tab_tr, y_tr)
    sp_ridge = spearmanr(ridge.predict(tab_va), y_va)[0]
    print(f"Ridge (alpha=1.0):              val Spearman = {sp_ridge:.4f}")

    # 2. GBDT
    gbdt = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          random_state=42)
    gbdt.fit(tab_tr, y_tr)
    sp_gbdt = spearmanr(gbdt.predict(tab_va), y_va)[0]
    print(f"GBDT (max_iter=300, lr=0.05):   val Spearman = {sp_gbdt:.4f}")

    # 3. Mean-reversion (prediktor tanpa fitting)
    # Karena target = log(1+views) - log(1+trailing), mean reversion = prediksi negatif trailing.
    sp_mr = spearmanr(-np.log1p(df["trailing_avg_views"].values[val_idx]), y_va)[0]
    print(f"Mean-reversion (-log1p(trail)): val Spearman = {sp_mr:.4f}")
    print()

    # Ringkasan
    print("=" * 60)
    print("Perbandingan dengan M4 tabular-only (val = 0.4003):")
    print(f"  Ridge:         {sp_ridge:.4f}")
    print(f"  GBDT:          {sp_gbdt:.4f}")
    print(f"  Mean-reversion:{sp_mr:.4f}")
    print()
    print("Interpretasi:")
    if sp_gbdt >= 0.35:
        print("  → GBDT tinggi: M4 tabular wajar, baseline M0/M0b/M3 UNDER-TRAINED")
    elif sp_gbdt <= 0.30:
        print("  → GBDT rendah tapi M4 tinggi: CURIGAI KEBOCORAN di M4")
    else:
        print("  → GBDT di tengah: perlu cek lebih lanjut")


if __name__ == "__main__":
    main()