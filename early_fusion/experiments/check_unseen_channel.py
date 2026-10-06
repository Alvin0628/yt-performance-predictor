"""Compare GBDT on the temporal split vs unseen-channel split.

Goal: check whether the GBDT test 0.4120 on the temporal split is inflated
by channel leakage (subs = channel ID).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from features.target import compute_target
from models.dataset import build_tabular_matrix


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
RANDOM_STATE = 42


def eval_split(df, train_idx, test_idx, label, drop_subs=False):
    """Train Ridge + GBDT on train_idx, evaluate on test_idx."""
    df = df.reset_index(drop=True)
    df["target"] = compute_target(df["views"], df["trailing_avg_views"])

    genres_train = sorted(df.loc[train_idx]["genre"].dropna().unique().tolist())
    tab_tr, sc = build_tabular_matrix(df.loc[train_idx], genres_train, fit_scaler=True)
    tab_te, _ = build_tabular_matrix(df.loc[test_idx], genres_train, scaler=sc)

    if drop_subs:
        # Remove the subscriber_count_at_upload column from the tabular features
        # Since build_tabular_matrix uses log-transformed columns, we need to remove it manually
        # subscriber is the first log-transformed column (index 0)
        # See TABULAR_LOG_COLS = ["subscriber_count_at_upload", "trailing_avg_views"]
        # So column 0 = subs
        tab_tr = np.delete(tab_tr, 0, axis=1)
        tab_te = np.delete(tab_te, 0, axis=1)

    y_tr = df["target"].values[train_idx]
    y_te = df["target"].values[test_idx]

    results = {}
    for name, m in [
        ("Ridge", Ridge(alpha=1.0)),
        ("GBDT", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                random_state=RANDOM_STATE)),
    ]:
        m.fit(tab_tr, y_tr)
        preds = m.predict(tab_te)
        sp = spearmanr(preds, y_te)[0]
        results[name] = sp

    print(f"  {label}:")
    print(f"    n_train={len(train_idx)}, n_test={len(test_idx)}, "
          f"n_test_channels_unique={df.loc[test_idx]['channel_id'].nunique()}")
    print(f"    tabular_dim={tab_tr.shape[1]}" + (" (subs DROPPED)" if drop_subs else ""))
    for name, sp in results.items():
        print(f"    {name}: {sp:.4f}")
    return results


def main():
    df = pd.read_parquet(SNAPSHOT).reset_index(drop=True)
    df["target"] = compute_target(df["views"], df["trailing_avg_views"])
    print(f"snapshot: {len(df)} rows, {df['channel_id'].nunique()} channels")
    print()

    # ============ Scenario 1: Temporal Split (existing) ============
    print("=" * 70)
    print("SCENARIO 1: TEMPORAL SPLIT (test channel is present in train)")
    print("=" * 70)
    tr_idx = np.where(df["split"].values == "train")[0]
    te_idx = np.where(df["split"].values == "test")[0]
    overlap_channels = set(df.loc[tr_idx]["channel_id"]) & set(df.loc[te_idx]["channel_id"])
    print(f"  Overlap channels (train ∩ test): {len(overlap_channels)} / {df.loc[te_idx]['channel_id'].nunique()} test channels")
    print()
    _ = eval_split(df, tr_idx, te_idx, "Temporal (with subs)")
    print()
    _ = eval_split(df, tr_idx, te_idx, "Temporal (subs dropped)", drop_subs=True)
    print()

    # ============ Scenario 2: Unseen-Channel Split ============
    print("=" * 70)
    print("SCENARIO 2: UNSEEN-CHANNEL SPLIT (test channel is NOT present in train)")
    print("=" * 70)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_va_idx = np.where(df["split"].values != "test")[0]  # train+val only (test remains untouched)
    df_tv = df.loc[train_va_idx].reset_index(drop=True)
    groups = df_tv["channel_id"].values

    tr_sub, te_sub = next(gss.split(df_tv, groups=groups))
    overlap = set(df_tv.loc[tr_sub]["channel_id"]) & set(df_tv.loc[te_sub]["channel_id"])
    print(f"  Overlap channels (train ∩ test): {len(overlap)} (should be 0)")
    print(f"  Train channels: {df_tv.loc[tr_sub]['channel_id'].nunique()}")
    print(f"  Test channels:  {df_tv.loc[te_sub]['channel_id'].nunique()}")
    print()

    # Call eval_split with df_tv
    _ = eval_split(df_tv, tr_sub, te_sub, "Unseen-channel (with subs)")
    print()
    _ = eval_split(df_tv, tr_sub, te_sub, "Unseen-channel (subs dropped)")


if __name__ == "__main__":
    main()