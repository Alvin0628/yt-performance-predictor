"""Load canonical split + verifikasi hash.

Dipakai di semua training M3'/M4a/M5/M6 supaya split konsisten.
"""
import json
from pathlib import Path

from early_fusion.data_spec import DEFAULT_SPEC


# Kept for old callers; the values now come from DEFAULT_SPEC.
SPLIT_FILE = Path(DEFAULT_SPEC.split_file)
SNAPSHOT_HASH = DEFAULT_SPEC.snapshot_hash


def load_canonical_split(verbose=True, spec=None):
    """Return dict: {train_ids, val_ids, test_ids, train_ids_hash, ...}.
    Raise kalau snapshot hash mismatch.

    spec=None -> DEFAULT_SPEC (perilaku lama). Kalau spec.snapshot_hash None, pencocokan
    hash dilakukan oleh pemanggil terhadap hash snapshot yang dihitung.
    """
    spec = spec or DEFAULT_SPEC
    split_file = Path(spec.split_file)
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file belum dibuat. Jalankan: "
            f"python -m early_fusion.splits.create_temporal_split"
        )

    split = json.loads(split_file.read_text())
    if spec.snapshot_hash is not None:
        assert split["snapshot_hash"] == spec.snapshot_hash, (
            f"Split dibuat untuk snapshot {split['snapshot_hash']}, "
            f"tapi sekarang {spec.snapshot_hash}"
        )

    if verbose:
        print(f"loaded split: {split_file}")
        print(f"  mode: {split['split_mode']}")
        print(f"  drop_columns: {split['drop_columns']}")
        print(f"  n_train={split['n_train']} n_val={split['n_val']} n_test={split['n_test']}")
        print(f"  train_hash={split['train_ids_hash']}")
        print(f"  val_hash={split['val_ids_hash']}")
        print(f"  test_hash={split['test_ids_hash']}")
        print(f"  created_at={split['created_at']}")
    return split


def apply_split_to_df(df, split):
    """Assign idx berdasarkan split ids. Return (train_idx, val_idx, test_idx)."""
    import numpy as np
    train_set = set(split["train_ids"])
    val_set = set(split["val_ids"])
    test_set = set(split["test_ids"])

    train_idx = np.where(df["video_id"].isin(train_set))[0]
    val_idx = np.where(df["video_id"].isin(val_set))[0]
    test_idx = np.where(df["video_id"].isin(test_set))[0]

    assert len(train_idx) == split["n_train"], \
        f"train count mismatch: {len(train_idx)} vs {split['n_train']}"
    assert len(val_idx) == split["n_val"], \
        f"val count mismatch: {len(val_idx)} vs {split['n_val']}"
    assert len(test_idx) == split["n_test"], \
        f"test count mismatch: {len(test_idx)} vs {split['n_test']}"
    return train_idx, val_idx, test_idx