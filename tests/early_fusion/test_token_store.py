"""Fase 3: TokenStore (cache token per video_id) dan update_store.

    python -m pytest tests/early_fusion/test_token_store.py -q

Tanpa torch/CLIP/MinIO: ekstraktor diganti fungsi palsu yang deterministik.
Butuh pyarrow (index.parquet), seperti cache lama.
"""
import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from early_fusion.data_spec import DEFAULT_SPEC, DataSpec
from early_fusion.datasets.token_cache import TokenCacheWriter, load_cache
from early_fusion.datasets.token_store import (TokenStore, StoreLockedError, IMG_FILE, TXT_FILE,
                                               MASK_FILE, META_FILE, LOCK_FILE)
from early_fusion.datasets.token_update import update_store

L, IT, ID, TD = 6, 3, 4, 5          # dimensi kecil untuk tes


def _rows(ids):
    """Isi deterministik per video_id: nilai dasar = angka di id."""
    out_img, out_txt, out_mask, out_ok = [], [], [], []
    for v in ids:
        k = int(v[1:])
        out_img.append(np.full((IT, ID), k + 0.5, dtype=np.float32))
        out_txt.append(np.full((L, TD), -k - 0.25, dtype=np.float32))
        m = np.zeros(L, dtype=bool); m[: 1 + k % L] = True
        out_mask.append(m)
        out_ok.append(k % 7 != 3)                      # beberapa thumbnail "gagal"
    return (np.stack(out_img), np.stack(out_txt), np.stack(out_mask), np.array(out_ok, dtype=bool))


def _legacy(tmp, n=10, snapshot_hash="deadbeefdeadbeef"):
    d = Path(tmp) / "legacy"
    ids = [f"v{i}" for i in range(n)]
    w = TokenCacheWriter(d, n, L, img_dim=ID, txt_dim=TD, img_tokens=IT)
    img, txt, mask, ok = _rows(ids)
    for i in range(n):
        w.write(i, img[i], txt[i], mask[i], ok[i])
    meta = {"n": n, "L": L, "img_tokens": IT, "img_dim": ID, "txt_dim": TD, "encoder": "clip_b32",
            "image_mode": "squash", "snapshot_hash": snapshot_hash, "snapshot_n_total": n,
            "n_thumb_ok": int(ok.sum()), "n_thumb_failed": int((~ok).sum())}
    w.close(meta, ids)
    del w                      # lepaskan memmap penulis (Windows tidak bisa menghapus file yang masih ter-map)
    gc.collect()
    return d, ids


def _read_cache(d):
    """load_cache() mengembalikan memmap yang terbuka. Salin ke array biasa lalu lepaskan mapping,
    supaya TemporaryDirectory bisa menghapus file-nya di Windows."""
    img, txt, mask, ok, index_df, meta = load_cache(d)
    out = (np.array(img), np.array(txt), np.array(mask), np.array(ok), index_df, meta)
    del img, txt, mask
    gc.collect()
    return out


def _empty(tmp):
    return TokenStore.create_empty(Path(tmp) / "store", L=L, img_tokens=IT, img_dim=ID, txt_dim=TD)


def _fake_extract_factory(calls):
    def extract(ids, titles):
        calls.append((list(ids), list(titles)))
        return _rows(ids)
    return extract


def test_create_empty_and_append_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        s = _empty(tmp)
        assert len(s) == 0 and s.take([])["img"].shape == (0, IT, ID)
        ids = ["v1", "v2", "v3"]
        assert s.append(ids, *_rows(ids)) == 3
        t = s.take(["v3", "v1"])
        e_img, e_txt, e_mask, e_ok = _rows(["v3", "v1"])
        assert np.array_equal(t["img"], e_img.astype(np.float16))
        assert np.array_equal(t["txt"], e_txt.astype(np.float16))
        assert np.array_equal(t["mask"], e_mask) and np.array_equal(t["thumb_ok"], e_ok)


def test_bootstrap_from_legacy_is_exact_and_drops_snapshot_hash():
    with tempfile.TemporaryDirectory() as tmp:
        legacy, ids = _legacy(tmp)
        s = TokenStore.create_from_legacy(legacy, Path(tmp) / "store")
        assert len(s) == len(ids) and s.video_ids == ids
        img, txt, mask, ok, index_df, meta = _read_cache(legacy)
        t = s.take(ids)
        assert np.array_equal(t["img"], np.asarray(img)) and np.array_equal(t["txt"], np.asarray(txt))
        assert np.array_equal(t["mask"], np.asarray(mask)) and np.array_equal(t["thumb_ok"], ok)
        assert "snapshot_hash" not in s.meta
        assert s.meta["provenance"]["legacy_snapshot_hash"] == "deadbeefdeadbeef"
        # cache lama tidak disentuh
        assert json.loads((legacy / "meta.json").read_text())["snapshot_hash"] == "deadbeefdeadbeef"


def test_take_follows_requested_order_and_reports_missing():
    with tempfile.TemporaryDirectory() as tmp:
        legacy, ids = _legacy(tmp)
        s = TokenStore.create_from_legacy(legacy, Path(tmp) / "store")
        order = ["v7", "v0", "v7", "v3"]                  # urutan bebas, boleh ganda
        t = s.take(order)
        e = _rows(order)
        assert np.array_equal(t["img"], e[0].astype(np.float16)) and np.array_equal(t["mask"], e[2])
        with pytest.raises(KeyError):
            s.take(["v1", "nope"])
        assert s.missing(["v2", "x1", "x2", "x1", "v9"]) == ["x1", "x2"]


def test_append_rejects_duplicates_and_bad_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        s = _empty(tmp)
        s.append(["v1"], *_rows(["v1"]))
        with pytest.raises(ValueError):
            s.append(["v1"], *_rows(["v1"]))                       # sudah ada
        with pytest.raises(ValueError):
            s.append(["v2", "v2"], *_rows(["v2", "v2"]))           # ganda dalam batch
        img, txt, mask, ok = _rows(["v5"])
        with pytest.raises(ValueError):
            s.append(["v5"], img[:, :, :-1], txt, mask, ok)        # bentuk salah
        assert len(s) == 1


def test_reopen_sees_appended_rows_and_old_rows_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        legacy, ids = _legacy(tmp, n=8)
        s = TokenStore.create_from_legacy(legacy, Path(tmp) / "store")
        before = s.take(ids)
        s.append(["v100", "v101"], *_rows(["v100", "v101"]))
        s2 = TokenStore(Path(tmp) / "store")
        assert len(s2) == 10 and s2.video_ids[-2:] == ["v100", "v101"]
        after = s2.take(ids)
        for k in before:
            assert np.array_equal(before[k], after[k])
        assert s2.meta["n"] == 10 and s2.meta["n_thumb_ok"] + s2.meta["n_thumb_failed"] == 10


def test_partial_append_is_ignored_and_cleaned_up():
    with tempfile.TemporaryDirectory() as tmp:
        legacy, ids = _legacy(tmp, n=6)
        d = Path(tmp) / "store"
        s = TokenStore.create_from_legacy(legacy, d)
        # simulasi crash: data sudah ditulis tetapi meta.json belum diperbarui
        for name, junk in ((IMG_FILE, 1000), (TXT_FILE, 777), (MASK_FILE, 13)):
            with open(d / name, "ab") as f:
                f.write(b"\xff" * junk)
        s2 = TokenStore(d)
        assert len(s2) == 6
        t = s2.take(ids)                                            # pembaca mengabaikan sisa byte
        assert np.array_equal(t["mask"], _rows(ids)[2])
        s2.append(["v50"], *_rows(["v50"]))                         # append berikutnya memotong sisa
        s3 = TokenStore(d)
        assert len(s3) == 7
        assert np.array_equal(s3.take(["v50"])["img"], _rows(["v50"])[0].astype(np.float16))
        assert np.array_equal(s3.take(ids)["img"], _rows(ids)[0].astype(np.float16))
        assert (d / IMG_FILE).stat().st_size == 7 * IT * ID * 2


def test_lock_blocks_a_second_writer_and_is_released():
    with tempfile.TemporaryDirectory() as tmp:
        s = _empty(tmp)
        (Path(tmp) / "store" / LOCK_FILE).write_text("pid=1")       # writer lain sedang aktif
        with pytest.raises(StoreLockedError):
            s.append(["v1"], *_rows(["v1"]))
        (Path(tmp) / "store" / LOCK_FILE).unlink()
        s.append(["v1"], *_rows(["v1"]))
        assert not (Path(tmp) / "store" / LOCK_FILE).exists()


def test_update_store_extracts_only_missing_in_snapshot_order_and_is_idempotent():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        legacy, ids = _legacy(tmp, n=6)
        s = TokenStore.create_from_legacy(legacy, Path(tmp) / "store")
        new_ids = [f"v{i}" for i in range(20, 27)]
        snap = ids[:3] + new_ids[:4] + ids[3:] + new_ids[4:]        # campuran lama + baru
        df = pd.DataFrame({"video_id": snap, "title": [f"judul {v}" for v in snap]})
        df.loc[df.index[-1], "title"] = None                         # judul kosong tidak boleh membuat gagal
        calls = []
        stats = update_store(s, df, _fake_extract_factory(calls), batch=3, flush_every=4, log=lambda *_: None)
        assert stats["n_new"] == 7 and stats["n_total"] == 13
        # hanya video baru yang diekstrak, urutan snapshot, dalam batch 3
        called = [v for c in calls for v in c[0]]
        assert called == new_ids and [len(c[0]) for c in calls] == [3, 3, 1]
        assert calls[-1][1] == [""]                                  # judul None -> ""
        # isi cocok dengan ekstraksi langsung, dan baris lama tidak berubah
        t = s.take(snap)
        e = _rows(snap)
        assert np.array_equal(t["img"], e[0].astype(np.float16)) and np.array_equal(t["thumb_ok"], e[3])
        assert stats["n_thumb_failed_new"] == int((~_rows(new_ids)[3]).sum())
        # dijalankan lagi: tidak ada yang diekstrak
        calls.clear()
        stats2 = update_store(s, df, _fake_extract_factory(calls), batch=3, log=lambda *_: None)
        assert stats2["n_new"] == 0 and calls == []


def test_update_store_resumes_after_interruption():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        s = _empty(tmp)
        ids = [f"v{i}" for i in range(10)]
        df = pd.DataFrame({"video_id": ids, "title": ids})
        boom = {"n": 0}

        def flaky(vids, titles):
            boom["n"] += 1
            if boom["n"] == 3:
                raise RuntimeError("koneksi putus")
            return _rows(vids)
        with pytest.raises(RuntimeError):
            update_store(s, df, flaky, batch=2, flush_every=2, log=lambda *_: None)
        assert len(s) == 4                                           # dua batch pertama sudah tersimpan
        calls = []
        update_store(s, df, _fake_extract_factory(calls), batch=2, log=lambda *_: None)
        assert [v for c in calls for v in c[0]] == ids[4:] and len(s) == 10
        assert np.array_equal(s.take(ids)["img"], _rows(ids)[0].astype(np.float16))


def test_legacy_load_cache_can_read_a_store():
    with tempfile.TemporaryDirectory() as tmp:
        legacy, ids = _legacy(tmp, n=5)
        s = TokenStore.create_from_legacy(legacy, Path(tmp) / "store")
        s.append(["v90"], *_rows(["v90"]))
        img, txt, mask, ok, index_df, meta = _read_cache(Path(tmp) / "store")
        assert img.shape == (6, IT, ID) and index_df["video_id"].tolist() == ids + ["v90"]
        assert meta["n"] == 6


def test_dataspec_cache_kind_default_and_validation():
    assert DEFAULT_SPEC.cache_kind == "snapshot"
    assert DataSpec(cache_kind="store").cache_kind == "store"
    with pytest.raises(ValueError):
        DataSpec(cache_kind="lain")


def test_take_leaves_no_open_file_mapping():
    """Tidak boleh ada mapping file store yang tersisa setelah take() (penting di Windows)."""
    if not sys.platform.startswith("linux"):
        pytest.skip("pemeriksaan ini memakai /proc/self/maps (Linux)")
    with tempfile.TemporaryDirectory() as tmp:
        legacy, ids = _legacy(tmp, n=9)
        d = Path(tmp) / "store"
        s = TokenStore.create_from_legacy(legacy, d)
        s.take(ids[:5])
        s.take(ids)
        maps = open("/proc/self/maps").read()
        assert str(d) not in maps and str(legacy) not in maps


def test_numerics_context_sets_and_restores_flags():
    torch = pytest.importorskip("torch")
    from early_fusion.datasets.token_update import numerics
    m0, c0 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True               # seperti yang dilakukan m6_core saat di-import
        with numerics("legacy"):
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is True       # dikembalikan
        with numerics("tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is True
        with numerics("ambient"):
            assert torch.backends.cuda.matmul.allow_tf32 is True
        torch.backends.cuda.matmul.allow_tf32 = False
        with numerics("legacy"):
            pass
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m0, c0
