"""Baseline tabular di val DAN test: Ridge, GBDT, mean-reversion.

Diperlukan untuk tabel comparable di paper.
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
RESULTS = Path("early_fusion/results/baselines_tabular.jsonl")


def main():
    df = pd.read_parquet(SNAPSHOT)
    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    tr = np.where(df["split"].values == "train")[0]
    va = np.where(df["split"].values == "val")[0]
    te = np.where(df["split"].values == "test")[0]
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}")

    genres = sorted(df.iloc[tr]["genre"].dropna().unique().tolist())
    tab_tr, sc = build_tabular_matrix(df.iloc[tr], genres, fit_scaler=True)
    tab_va, _ = build_tabular_matrix(df.iloc[va], genres, scaler=sc)
    tab_te, _ = build_tabular_matrix(df.iloc[te], genres, scaler=sc)

    y_tr = df["target"].values[tr]
    y_va = df["target"].values[va]
    y_te = df["target"].values[te]

    print(f"tabular dim: {tab_tr.shape[1]}")
    print()

    results = []

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(tab_tr, y_tr)
    sp_va = spearmanr(ridge.predict(tab_va), y_va)[0]
    sp_te = spearmanr(ridge.predict(tab_te), y_te)[0]
    print(f"Ridge (alpha=1.0):     val={sp_va:.4f}  test={sp_te:.4f}")
    results.append({"model": "Ridge", "alpha": 1.0,
                    "val_spearman": float(sp_va), "test_spearman": float(sp_te),
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te)})

    # GBDT
    gbdt = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          random_state=42)
    gbdt.fit(tab_tr, y_tr)
    sp_va = spearmanr(gbdt.predict(tab_va), y_va)[0]
    sp_te = spearmanr(gbdt.predict(tab_te), y_te)[0]
    print(f"GBDT (iter=300, lr=0.05): val={sp_va:.4f}  test={sp_te:.4f}")
    results.append({"model": "GBDT", "max_iter": 300, "learning_rate": 0.05,
                    "val_spearman": float(sp_va), "test_spearman": float(sp_te),
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te)})

    # GBDT dengan max_iter lebih besar (opsional, untuk lihat ceiling)
    gbdt2 = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.03,
                                           random_state=42)
    gbdt2.fit(tab_tr, y_tr)
    sp_va = spearmanr(gbdt2.predict(tab_va), y_va)[0]
    sp_te = spearmanr(gbdt2.predict(tab_te), y_te)[0]
    print(f"GBDT (iter=1000, lr=0.03): val={sp_va:.4f}  test={sp_te:.4f}")
    results.append({"model": "GBDT_large", "max_iter": 1000, "learning_rate": 0.03,
                    "val_spearman": float(sp_va), "test_spearman": float(sp_te),
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te)})

    # Mean-reversion (baseline trivial)
    mr_va = -np.log1p(df["trailing_avg_views"].values[va])
    mr_te = -np.log1p(df["trailing_avg_views"].values[te])
    sp_va = spearmanr(mr_va, y_va)[0]
    sp_te = spearmanr(mr_te, y_te)[0]
    print(f"Mean-reversion:          val={sp_va:.4f}  test={sp_te:.4f}")
    results.append({"model": "MeanReversion",
                    "val_spearman": float(sp_va), "test_spearman": float(sp_te)})

    print()
    print("=" * 60)
    print("Ringkasan (val / test):")
    for r in results:
        print(f"  {r['model']:<20} {r['val_spearman']:.4f} / {r['test_spearman']:.4f}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved: {RESULTS}")


if __name__ == "__main__":
    main()