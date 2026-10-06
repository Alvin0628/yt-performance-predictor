"""Freeze the dataset snapshot + split manifest into parquet + JSON.

Run once after precompute_embeddings is complete.
This snapshot is the canonical version sent to the teammate.

Filter: image_embedding IS NOT NULL (10 rows with failed thumbnails are excluded).
"""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from models.precompute_embeddings import engine


OUT_DIR = Path("data_snapshots")
OUT_DIR.mkdir(exist_ok=True)


def hash_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def main():
    # 1. All finalized rows (for funnel/audit)
    df_all = pd.read_sql(
        "SELECT video_id, channel_ref, image_embedding IS NOT NULL AS has_img, "
        "text_embedding IS NOT NULL AS has_txt "
        "FROM videos WHERE label_finalized = true",
        engine,
    )

    # 2. Snapshot: only rows with image_embedding
    df = pd.read_sql(
        "SELECT * FROM videos "
        "WHERE label_finalized = true "
        "AND image_embedding IS NOT NULL "
        "AND trailing_avg_views IS NOT NULL "   # ← TAMBAH INI
        "ORDER BY published_at, video_id",
        engine,
    )
    print(f"finalized (all):   {len(df_all)}")
    print(f"finalized + img:   {len(df)}")
    print(f"excluded (no img): {len(df_all) - len(df)}")

    # 3. Temporal 80/10/10 split
    n = len(df)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    df["split"] = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)

    n_channels = df["channel_ref"].nunique()
    print(f"  n_train: {n_tr}  n_val: {n_va}  n_test: {n - n_tr - n_va}")
    print(f"  channels: {n_channels}")

    # 4. Save parquet
    out_parquet = OUT_DIR / "snapshot.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"  saved: {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)")

    # 5. Collect excluded video_ids
    excluded = df_all[~df_all["has_img"]]
    excluded_ids = excluded["video_id"].tolist()
    (OUT_DIR / "excluded_videos.json").write_text(
        json.dumps(excluded_ids, indent=2)
    )

    manifest = {
        "n_finalized_all": int(len(df_all)),
        "n_total": n,
        "n_excluded_no_image": int(len(df_all) - n),
        "n_train": n_tr,
        "n_val": n_va,
        "n_test": n - n_tr - n_va,
        "n_channels": n_channels,
        "snapshot_hash": hash_df(df),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "columns": list(df.columns),
        "query": (
            "SELECT * FROM videos WHERE label_finalized = true "
            "AND image_embedding IS NOT NULL "
            "AND trailing_avg_views IS NOT NULL "   # ← TAMBAH INI
            "ORDER BY published_at, video_id"
        ),
        "excluded_video_ids_file": "excluded_videos.json",
    }
    out_json = OUT_DIR / "split_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2))
    print()
    print(json.dumps(manifest, indent=2))
    print(f"\n  saved: {out_json}")
    print(f"  saved: {OUT_DIR / 'excluded_videos.json'}")


if __name__ == "__main__":
    main()