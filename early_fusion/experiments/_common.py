"""Shared utilities untuk semua training final (M3'/M4a/M5/M6).

- set_seed
- get_git_info
- load_snapshot (verify hash)
- prepare_tabular_no_subs (drop subs)
- prepare_tabular_with_subs (kalau perlu)
"""
import hashlib
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.dataset import (build_tabular_matrix, TABULAR_LOG_COLS,
                             TABULAR_NUMERIC_COLS, TABULAR_BOOL_COLS)


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
SNAPSHOT_HASH = "c14dba895034fc4c"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hash_df(df):
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def hash_ids(ids):
    return hashlib.sha256(",".join(sorted(map(str, ids))).encode()).hexdigest()[:16]


def get_git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return sha, bool(dirty)
    except Exception:
        return "unknown", False


def parse_vector(s):
    if isinstance(s, str):
        return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(s, dtype=np.float32)


def load_snapshot(verbose=True):
    """Load snapshot + verify hash. Return (df, image_emb, text_emb).

    df sudah punya kolom 'target' (log ratio views vs trailing).
    """
    from features.target import compute_target   # ← tambah import

    df = pd.read_parquet(SNAPSHOT)
    computed = hash_df(df)
    if verbose:
        print(f"snapshot: {len(df)} rows, hash={computed}")
    assert computed == SNAPSHOT_HASH, \
        f"snapshot hash mismatch: {computed} vs {SNAPSHOT_HASH}"

    image_emb = np.stack(df["image_embedding"].apply(parse_vector).values)
    text_emb = np.stack(df["text_embedding"].apply(parse_vector).values)
    df = df.drop(columns=["image_embedding", "text_embedding"]).reset_index(drop=True)

    df["target"] = compute_target(df["views"], df["trailing_avg_views"])   # ← TAMBAH INI

    return df, image_emb, text_emb


def prepare_tabular_drop_subs(df, train_idx, val_idx, test_idx,
                               genre_categories=None, verbose=True):
    """Build tabular matrix, lalu drop kolom subscriber_count_at_upload.

    Return (train_tab, val_tab, test_tab, scaler, genre_categories, tabular_dim).
    """
    if genre_categories is None:
        genre_categories = sorted(df.iloc[train_idx]["genre"].dropna().unique().tolist())

    train_tab, scaler = build_tabular_matrix(
        df.iloc[train_idx], genre_categories, fit_scaler=True)
    val_tab, _ = build_tabular_matrix(
        df.iloc[val_idx], genre_categories, scaler=scaler)
    test_tab, _ = build_tabular_matrix(
        df.iloc[test_idx], genre_categories, scaler=scaler)

    # Drop subs: kolom log pertama (index 0)
    # TABULAR_LOG_COLS = ["subscriber_count_at_upload", "trailing_avg_views"]
    # Setelah drop, trailing jadi index 0.
    SUBS_IDX = 0
    train_tab = np.delete(train_tab, SUBS_IDX, axis=1)
    val_tab = np.delete(val_tab, SUBS_IDX, axis=1)
    test_tab = np.delete(test_tab, SUBS_IDX, axis=1)

    # Update scaler? scaler adalah tuple (log_scaler, numeric_scaler).
    # log_scaler fit pada 2 kolom (subs + trailing). Setelah drop subs,
    # kita tetap pakai log_scaler karena kolomnya sudah di-transform sebelum di-drop.
    # numeric_scaler tidak terpengaruh.
    # Konsisten: semua eksperimen (M3'/M4a/M5/M6) drop index 0 dari output ini.

    if verbose:
        print(f"tabular: {train_tab.shape[1]} dims (drop subs from {train_tab.shape[1]+1})")

    return train_tab, val_tab, test_tab, scaler, genre_categories, train_tab.shape[1]