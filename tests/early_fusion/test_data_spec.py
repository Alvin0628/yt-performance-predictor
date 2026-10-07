"""Fase 2: parametrisasi data tidak boleh mengubah perilaku lama.

Jalankan dari root repo:  python -m pytest tests/early_fusion/test_data_spec.py -q

Tes tanpa torch selalu jalan. Tes yang butuh torch (import _common/m6_core) di-skip kalau
torch tidak ada. Tes yang butuh snapshot asli (data_snapshots/) di-skip kalau file tidak ada.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from early_fusion.data_spec import (DEFAULT_SPEC, DataSpec, LEGACY_SNAPSHOT_HASH,
                                    assign_temporal_split, hash_df, temporal_split_sizes)
from early_fusion.splits import create_temporal_split
from early_fusion.splits.load_split import load_canonical_split, SPLIT_FILE, SNAPSHOT_HASH


def test_default_spec_matches_legacy_constants():
    assert LEGACY_SNAPSHOT_HASH == "c14dba895034fc4c"
    assert DEFAULT_SPEC.snapshot_hash == "c14dba895034fc4c"
    assert Path(DEFAULT_SPEC.snapshot_path).as_posix() == "data_snapshots/snapshot.parquet"
    assert Path(DEFAULT_SPEC.split_file).as_posix() == "early_fusion/splits/temporal_no_subs.json"
    assert Path(DEFAULT_SPEC.cache_dir).as_posix() == "data_snapshots/token_cache"
    # nama lama di modul lain tetap ada dan bernilai sama
    assert SNAPSHOT_HASH == DEFAULT_SPEC.snapshot_hash
    assert SPLIT_FILE == Path(DEFAULT_SPEC.split_file)


def test_split_sizes_match_export_rule_and_the_real_snapshot_numbers():
    # aturan scripts/export_snapshot.py: int(0.8*n), int(0.1*n), sisanya test
    for n in (10, 99, 1000, 11285, 12345):
        n_tr, n_va, n_te = temporal_split_sizes(n)
        assert n_tr == int(0.8 * n) and n_va == int(0.1 * n)
        assert n_tr + n_va + n_te == n
    # angka snapshot c14dba895034fc4c yang sudah ada
    assert temporal_split_sizes(11285) == (9028, 1128, 1129)
    labels = assign_temporal_split(11285)
    assert labels.count("train") == 9028 and labels.count("val") == 1128 and labels.count("test") == 1129
    assert labels[0] == "train" and labels[-1] == "test"


def test_hash_df_is_deterministic_and_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert hash_df(df) == hash_df(df.copy())
    df2 = df.copy(); df2.loc[1, "a"] = 99
    assert hash_df(df) != hash_df(df2)


def _tiny_split_file(tmp, snapshot_hash):
    p = Path(tmp) / "split.json"
    p.write_text(json.dumps({
        "snapshot_hash": snapshot_hash, "split_mode": "temporal_no_subs",
        "drop_columns": ["subscriber_count_at_upload"], "n_total": 3,
        "n_train": 1, "n_val": 1, "n_test": 1,
        "train_ids_hash": "a", "val_ids_hash": "b", "test_ids_hash": "c",
        "train_ids": ["v1"], "val_ids": ["v2"], "test_ids": ["v3"], "created_at": "x",
    }))
    return p


def test_load_split_with_spec_checks_hash_only_when_fixed():
    with tempfile.TemporaryDirectory() as tmp:
        p = _tiny_split_file(tmp, "aaaa")
        # hash tetap cocok -> lolos
        load_canonical_split(verbose=False, spec=DataSpec(split_file=p, snapshot_hash="aaaa"))
        # hash tetap beda -> ditolak (perilaku lama)
        with pytest.raises(AssertionError):
            load_canonical_split(verbose=False, spec=DataSpec(split_file=p, snapshot_hash="bbbb"))
        # hash None -> pemanggil yang mencocokkan; tidak ditolak di sini
        s = load_canonical_split(verbose=False, spec=DataSpec(split_file=p, snapshot_hash=None))
        assert s["snapshot_hash"] == "aaaa"
        # file tidak ada -> pesan lama
        with pytest.raises(FileNotFoundError):
            load_canonical_split(verbose=False, spec=DataSpec(split_file=Path(tmp) / "nope.json"))


def test_create_split_cli_default_hash_vs_auto():
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(10)],
        "split": assign_temporal_split(10),
        "views": range(10),
    })
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / "snap.parquet"; df.to_parquet(snap, index=False)
        out1 = Path(tmp) / "s1.json"; out2 = Path(tmp) / "s2.json"
        create_temporal_split.main(["--snapshot", str(snap), "--out", str(out1)])
        create_temporal_split.main(["--snapshot", str(snap), "--snapshot-hash", "auto", "--out", str(out2)])
        s1 = json.loads(out1.read_text()); s2 = json.loads(out2.read_text())
        assert s1["snapshot_hash"] == LEGACY_SNAPSHOT_HASH            # perilaku lama
        assert s2["snapshot_hash"] == hash_df(pd.read_parquet(snap))   # mode baru
        assert (s1["n_train"], s1["n_val"], s1["n_test"]) == (8, 1, 1)
        assert s1["train_ids"] == s2["train_ids"]


def test_load_snapshot_spec_and_embeddings_flag():
    pytest.importorskip("torch"); pytest.importorskip("sklearn"); pytest.importorskip("pyarrow")
    from early_fusion.experiments._common import load_snapshot
    n = 5
    df = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(n)],
        "views": [100, 200, 300, 400, 500], "trailing_avg_views": [90.0, 150.0, 250.0, 350.0, 450.0],
        "image_embedding": ["[0.1,0.2]"] * n, "text_embedding": ["[0.3,0.4]"] * n,
    })
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / "snap.parquet"; df.to_parquet(snap, index=False)
        h = hash_df(pd.read_parquet(snap))
        # hash tetap salah -> ditolak
        with pytest.raises(AssertionError):
            load_snapshot(verbose=False, spec=DataSpec(snapshot_path=snap, snapshot_hash="0000"))
        # hash tetap benar + embedding
        d1, i1, t1 = load_snapshot(verbose=False, spec=DataSpec(snapshot_path=snap, snapshot_hash=h))
        assert i1.shape == (n, 2) and t1.shape == (n, 2) and "target" in d1.columns
        assert d1.attrs["snapshot_hash"] == h and "image_embedding" not in d1.columns
        # hash None + tanpa embedding: df identik, embedding None
        d2, i2, t2 = load_snapshot(verbose=False, load_embeddings=False,
                                   spec=DataSpec(snapshot_path=snap, snapshot_hash=None))
        assert i2 is None and t2 is None
        pd.testing.assert_frame_equal(d1, d2)
        assert d2.attrs["snapshot_hash"] == h


def test_real_snapshot_reproduces_existing_split_column():
    pytest.importorskip("pyarrow")
    snap = Path(DEFAULT_SPEC.snapshot_path)
    if not snap.exists():
        pytest.skip("data_snapshots/snapshot.parquet tidak ada di mesin ini")
    df = pd.read_parquet(snap)
    assert hash_df(df) == LEGACY_SNAPSHOT_HASH
    # aturan baru harus mereproduksi kolom split yang dibuat export lama, baris demi baris
    assert df["split"].tolist() == assign_temporal_split(len(df))
    # urutan kronologis: published_at tidak menurun. Urutan antar-video dengan published_at
    # yang sama mengikuti collation Postgres (ORDER BY di export), bukan urutan string Python,
    # jadi itu sengaja tidak dibandingkan di sini.
    assert df["published_at"].is_monotonic_increasing
