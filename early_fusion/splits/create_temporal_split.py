"""Create the canonical temporal split with subscriber counts dropped.

Run ONCE. Output: early_fusion/splits/temporal_no_subs.json

All M3'/M4a/M5/M6 training scripts load this file so that the split remains consistent.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import hashlib
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
SNAPSHOT_HASH = "c14dba895034fc4c"
OUT_DIR = Path("early_fusion/splits")
OUT_FILE = OUT_DIR / "temporal_no_subs.json"


def hash_ids(ids):
    return hashlib.sha256(",".join(sorted(map(str, ids))).encode()).hexdigest()[:16]


def get_git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return sha, bool(dirty)
    except Exception:
        return "unknown", False


def main():
    df = pd.read_parquet(SNAPSHOT)
    print(f"loaded snapshot: {len(df)} rows")

    train_ids = df.loc[df["split"] == "train", "video_id"].tolist()
    val_ids = df.loc[df["split"] == "val", "video_id"].tolist()
    test_ids = df.loc[df["split"] == "test", "video_id"].tolist()

    print(f"train: {len(train_ids)} videos")
    print(f"val:   {len(val_ids)} videos")
    print(f"test:  {len(test_ids)} videos")

    git_sha, git_dirty = get_git_info()

    split_data = {
        "snapshot_hash": SNAPSHOT_HASH,
        "split_mode": "temporal_no_subs",
        "drop_columns": ["subscriber_count_at_upload"],
        "n_total": len(df),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train_ids_hash": hash_ids(train_ids),
        "val_ids_hash": hash_ids(val_ids),
        "test_ids_hash": hash_ids(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(split_data, indent=2))
    print(f"\nsaved: {OUT_FILE}")
    print(f"split hashes:")
    print(f"  train: {split_data['train_ids_hash']}")
    print(f"  val:   {split_data['val_ids_hash']}")
    print(f"  test:  {split_data['test_ids_hash']}")


if __name__ == "__main__":
    main()