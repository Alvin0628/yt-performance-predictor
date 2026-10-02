"""Check whether subscriber_count_at_upload is nearly constant per channel.

If yes, subs = proxy channel_id → the model can 'memorize' the channel → leakage
in the temporal split.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


SNAPSHOT = Path("data_snapshots/snapshot.parquet")


def main():
    df = pd.read_parquet(SNAPSHOT)
    print(f"snapshot: {len(df)} rows, {df['channel_id'].nunique()} channels")
    print()

    # 1. Check unique subs per channel
    print("=== Check 1: Unique subscriber_count_at_upload per channel ===")
    n_unique = df.groupby("channel_id")["subscriber_count_at_upload"].nunique()
    print(f"  Channels with exactly 1 unique subs: {(n_unique == 1).sum()} / {len(n_unique)}")
    print(f"  Channels with exactly 2 unique subs: {(n_unique == 2).sum()}")
    print(f"  Channels with 3+ unique subs:        {(n_unique >= 3).sum()}")
    print(f"  Mean unique values per channel:      {n_unique.mean():.2f}")
    print(f"  Median unique values per channel:    {n_unique.median():.0f}")
    print()

    # 2. Check average videos per channel
    print("=== Check 2: Videos per channel ===")
    n_videos = df.groupby("channel_id").size()
    print(f"  Mean videos per channel: {n_videos.mean():.1f}")
    print(f"  Median:                  {n_videos.median():.0f}")
    print(f"  Min:                     {n_videos.min()}")
    print(f"  Max:                     {n_videos.max()}")
    print()

    # 3. Check subscriber range per channel (max - min) relatively
    print("=== Check 3: Subscriber range (max - min) per channel ===")
    subs_range = df.groupby("channel_id")["subscriber_count_at_upload"].agg(["min", "max", "mean"])
    subs_range["range"] = subs_range["max"] - subs_range["min"]
    subs_range["range_pct"] = subs_range["range"] / subs_range["mean"].clip(lower=1) * 100
    print(f"  Mean range (abs):     {subs_range['range'].mean():,.0f}")
    print(f"  Median range (abs):   {subs_range['range'].median():,.0f}")
    print(f"  Mean range (%):       {subs_range['range_pct'].mean():.2f}%")
    print(f"  Median range (%):     {subs_range['range_pct'].median():.2f}%")
    print()

    # 4. Correlation: is subs more like a channel ID than a value itself?
    print("=== Check 4: Subs as identifier ===")
    # If subs is nearly constant per channel, then subs ≈ channel_id
    # We measure: what % of subs variance is explained by channel_id
    # (R² from simple ANOVA)
    from sklearn.metrics import r2_score
    subs_global_mean = df["subscriber_count_at_upload"].mean()
    df["subs_pred_by_channel"] = df.groupby("channel_id")["subscriber_count_at_upload"].transform("mean")
    ss_res = ((df["subscriber_count_at_upload"] - df["subs_pred_by_channel"]) ** 2).sum()
    ss_tot = ((df["subscriber_count_at_upload"] - subs_global_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  R² (channel_id explains subs variance): {r2:.6f}")
    if r2 > 0.99:
        print("  → subs is almost deterministic from channel_id. MAJOR LEAKAGE.")
    elif r2 > 0.95:
        print("  → subs is strongly related to channel_id. MODERATE LEAKAGE.")
    elif r2 > 0.80:
        print("  → subs is related to channel_id. Needs attention.")
    else:
        print("  → subs is not a proxy for channel_id. OK.")
    print()

    # 5. Distribution: how many unique subs values overall
    print("=== Check 5: Global subs distribution ===")
    print(f"  Unique subs values globally: {df['subscriber_count_at_upload'].nunique()}")
    print(f"  Min: {df['subscriber_count_at_upload'].min():,.0f}")
    print(f"  Median: {df['subscriber_count_at_upload'].median():,.0f}")
    print(f"  Max: {df['subscriber_count_at_upload'].max():,.0f}")
    print()

    # 6. Verification: subs vs views correlation
    print("=== Check 6: Subs vs views correlation ===")
    from scipy.stats import spearmanr, pearsonr
    sp, _ = spearmanr(df["subscriber_count_at_upload"], df["views"])
    pe, _ = pearsonr(df["subscriber_count_at_upload"], df["views"])
    print(f"  Spearman(subs, views): {sp:.4f}")
    print(f"  Pearson(subs, views):  {pe:.4f}")
    print()

    # Conclusion
    print("=" * 60)
    print("CONCLUSION:")
    if r2 > 0.95:
        print(f"  subs R²={r2:.4f} — subs is almost deterministic from channel_id.")
        print("  In the temporal split, the model can 'memorize' the channel effect.")
        print("  NEED TO TEST UNSEEN-CHANNEL SPLIT.")
    else:
        print(f"  subs R²={r2:.4f} — subs is not a perfect substitute for channel_id.")
        print("  Temporal leakage risk is lower.")
    print("=" * 60)


if __name__ == "__main__":
    main()