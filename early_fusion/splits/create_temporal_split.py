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

import argparse

import numpy as np
import pandas as pd

from early_fusion.data_spec import DEFAULT_SPEC, hash_df


SNAPSHOT = Path(DEFAULT_SPEC.snapshot_path)
SNAPSHOT_HASH = DEFAULT_SPEC.snapshot_hash
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


def build_split(df, snapshot_hash):
    """Split manifest from the snapshot's own 'split' column (assigned at export time)."""
    train_ids = df.loc[df["split"] == "train", "video_id"].tolist()
    val_ids = df.loc[df["split"] == "val", "video_id"].tolist()
    test_ids = df.loc[df["split"] == "test", "video_id"].tolist()

    git_sha, git_dirty = get_git_info()

    return {
        "snapshot_hash": snapshot_hash,
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("--snapshot-hash", default=SNAPSHOT_HASH,
                    help="hash yang dicatat di split; 'auto' = hitung dari file")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.snapshot)
    print(f"loaded snapshot: {len(df)} rows")
    snapshot_hash = hash_df(df) if args.snapshot_hash == "auto" else args.snapshot_hash

    split_data = build_split(df, snapshot_hash)
    print(f"train: {split_data['n_train']} videos")
    print(f"val:   {split_data['n_val']} videos")
    print(f"test:  {split_data['n_test']} videos")

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(split_data, indent=2))
    print(f"\nsaved: {out_file}")
    print("split hashes:")
    print(f"  train: {split_data['train_ids_hash']}")
    print(f"  val:   {split_data['val_ids_hash']}")
    print(f"  test:  {split_data['test_ids_hash']}")


if __name__ == "__main__":
    main()
