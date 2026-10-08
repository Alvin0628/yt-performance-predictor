"""Where the early-fusion pipeline reads its data from, as one object.

Until now the snapshot path/hash, the split file and the token-cache directory were
constants spread over several modules. Automatic retraining needs a *new* snapshot on
every run, so they travel together in a DataSpec instead.

DEFAULT_SPEC reproduces the old constants exactly: code that passes no spec behaves
as it did before (same files, same checks).

This module is deliberately dependency-light (no torch, no sklearn).
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

LEGACY_SNAPSHOT_HASH = "c14dba895034fc4c"


@dataclass(frozen=True)
class DataSpec:
    snapshot_path: Path = Path("data_snapshots/snapshot.parquet")
    # None = do not compare against a fixed hash; the hash computed from the file is used.
    snapshot_hash: Optional[str] = LEGACY_SNAPSHOT_HASH
    split_file: Path = Path("early_fusion/splits/temporal_no_subs.json")
    cache_dir: Path = Path("data_snapshots/token_cache")
    # "snapshot": legacy cache built for exactly this snapshot (same row order, hash in meta.json)
    # "store":    TokenStore keyed by video_id (rows are looked up in snapshot order)
    cache_kind: str = "snapshot"

    def __post_init__(self):
        if self.cache_kind not in ("snapshot", "store"):
            raise ValueError(f"cache_kind must be 'snapshot' or 'store', got {self.cache_kind!r}")


DEFAULT_SPEC = DataSpec()


def hash_df(df: pd.DataFrame) -> str:
    """Snapshot fingerprint (same function the snapshot export uses)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def temporal_split_sizes(n, frac_train=0.8, frac_val=0.1):
    """(n_train, n_val, n_test) for a chronologically ordered snapshot.

    Same arithmetic as scripts/export_snapshot.py: int() truncation for train and val,
    the remainder is test.
    """
    n_tr = int(frac_train * n)
    n_va = int(frac_val * n)
    return n_tr, n_va, n - n_tr - n_va


def assign_temporal_split(n, frac_train=0.8, frac_val=0.1):
    """Split label per row for a snapshot already ordered by (published_at, video_id)."""
    n_tr, n_va, n_te = temporal_split_sizes(n, frac_train, frac_val)
    return ["train"] * n_tr + ["val"] * n_va + ["test"] * n_te
