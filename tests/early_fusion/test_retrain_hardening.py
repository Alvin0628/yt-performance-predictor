"""Fase 4.1: penjaga thumbnail, `all` yang bisa dilanjutkan, dan opsi --threads (tanpa torch/Postgres/MinIO).

Sejak Fase 5, `all` berakhir di tahap `gate` (keputusan promosi = baris JSON terakhir) dan mewajibkan
`--live PATH` atau `--first-promotion` (divalidasi sebelum run dibuat). Tes `all` di sini memakai tahap palsu
untuk export..verify DAN gate, dan memberi argumen gate yang valid, supaya yang diuji tetap resume/pembuatan run.

    python -m pytest tests/early_fusion/test_retrain_hardening.py -q
"""
import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from early_fusion import retrain as rt
from early_fusion.datasets.token_store import TokenStore

REPO = Path(__file__).resolve().parent.parent.parent
quiet = lambda *a, **k: None


def make_raw(n=100, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    return pd.DataFrame({
        "video_id": [f"v{i:04d}" for i in range(n)],
        "published_at": [t0 + pd.Timedelta(hours=int(h)) for h in np.arange(n) * 5],
        "title": [f"title {i}" for i in range(n)], "channel_ref": rng.integers(0, 7, n),
        "views": rng.integers(100, 10_000, n), "trailing_avg_views": rng.integers(100, 10_000, n),
        "genre": rng.choice(["a", "b", "c"], n),
    })


def exported(tmp_path, n=100):
    run = rt.Run.new(tmp_path / "runs", "r1")
    rt.stage_export(run, read_sql_fn=lambda: make_raw(n), log=quiet)
    return run


def extractor(ok_mask_fn):
    def extract(ids, titles):
        n = len(ids)
        return (np.zeros((n, 3, 4), np.float32), np.zeros((n, 6, 5), np.float32),
                np.ones((n, 6), bool), np.array([ok_mask_fn(v) for v in ids], dtype=bool))
    return extract


# ------------------------------------------------------------------ penjaga thumbnail
def test_tokens_refuse_to_store_a_minio_outage_and_recover_afterwards(tmp_path):
    run = exported(tmp_path)
    store_dir = tmp_path / "store"
    TokenStore.create_empty(store_dir, L=6, img_tokens=3, img_dim=4, txt_dim=5)
    with pytest.raises(RuntimeError, match="thumbnails failed"):
        rt.stage_tokens(run, store_dir=store_dir, extractor=extractor(lambda v: False), batch=8, log=quiet)
    assert len(TokenStore(store_dir)) == 0                       # tidak ada token nol yang tertulis
    assert "tokens" not in run.load_state()["stages"]            # tahap tidak ditandai selesai
    info = rt.stage_tokens(run, store_dir=store_dir, extractor=extractor(lambda v: True), log=quiet)
    assert info["n_new"] == 100 and info["n_thumb_failed_new"] == 0
    assert int(TokenStore(store_dir).thumb_ok.sum()) == 100


def test_tokens_tolerate_a_few_failed_thumbnails(tmp_path):
    run = exported(tmp_path)
    store_dir = tmp_path / "store"
    TokenStore.create_empty(store_dir, L=6, img_tokens=3, img_dim=4, txt_dim=5)
    bad = {"v0003", "v0050"}
    info = rt.stage_tokens(run, store_dir=store_dir, extractor=extractor(lambda v: v not in bad), log=quiet)
    assert info["n_new"] == 100 and info["n_thumb_failed_new"] == 2


def test_max_fail_frac_option_and_default():
    p = rt.build_parser()
    assert p.parse_args(["tokens", "--run-id", "x"]).max_fail_frac == 0.05
    assert p.parse_args(["all"]).max_fail_frac == 0.05
    assert p.parse_args(["tokens", "--run-id", "x", "--max-fail-frac", "0.5"]).max_fail_frac == 0.5


# ------------------------------------------------------------------ all: lanjutkan run yang ada
DECISION = dict(promoted=False, versioned_path="x.pt", reason="r", new_spearman=0.4, run_id="rA", stage="gate")
VERIFY_PAYLOAD = dict(promoted=False, versioned_path="x.pt", reason="verified", new_spearman=0.3, run_id="rA",
                      stage="verify")
GATE_ARGS = ("--first-promotion",)        # kombinasi gate yang sah; stage_gate palsu tidak membacanya


def _install_fake_stages(monkeypatch, calls, fail_train_once):
    state = {"failed": False}

    def make(name, ret=None):
        def f(run, **kw):
            if name == "train" and fail_train_once and not state["failed"]:
                state["failed"] = True
                raise RuntimeError("OOM di tengah training")
            calls.append(name)
            run.mark(name, ret if ret is not None else {"ok": True})
            return ret if ret is not None else {"ok": True}
        return f
    for n, attr in (("export", "stage_export"), ("tokens", "stage_tokens"), ("split", "stage_split"),
                    ("train", "stage_train"), ("package", "stage_package")):
        monkeypatch.setattr(rt, attr, make(n))
    monkeypatch.setattr(rt, "stage_verify", make("verify", dict(VERIFY_PAYLOAD)))
    monkeypatch.setattr(rt, "stage_gate", make("gate", dict(DECISION)))      # `all` berakhir di gate


def _run_all(tmp_path, *extra):
    out = io.StringIO()
    with redirect_stdout(out):
        rc = rt.main(["all", "--root", str(tmp_path / "runs"), "--store", str(tmp_path / "store"),
                      "--config", str(REPO / rt.FINAL_CFG), "--run-id", "rA", *GATE_ARGS, *extra])
    return rc, json.loads(out.getvalue().strip().splitlines()[-1])


def test_all_resumes_an_interrupted_run_and_skips_finished_stages(tmp_path, monkeypatch):
    calls = []
    _install_fake_stages(monkeypatch, calls, fail_train_once=True)
    with pytest.raises(RuntimeError, match="OOM"):
        _run_all(tmp_path)
    assert calls == ["export", "tokens", "split"]                # crash di train
    rc, last = _run_all(tmp_path)                                # run yang sama dilanjutkan
    assert rc == 0 and last["promoted"] is False and last["versioned_path"] == "x.pt"
    assert last["stage"] == "gate" and last["new_spearman"] == 0.4                # JSON akhir = keputusan gate
    assert calls == ["export", "tokens", "split", "train", "package", "verify", "gate"]   # tahap lama tidak diulang
    n_before = len(calls)
    rc, last2 = _run_all(tmp_path)                               # sudah tuntas: tidak ada tahap yang jalan
    assert len(calls) == n_before and last2["new_spearman"] == 0.4 and "finished_at" not in last2


def test_all_without_run_id_still_creates_a_new_run_each_time(tmp_path, monkeypatch):
    calls = []
    _install_fake_stages(monkeypatch, calls, fail_train_once=False)
    for rid in ("r1", "r2"):
        out = io.StringIO()
        with redirect_stdout(out):
            rt.main(["all", "--root", str(tmp_path / "runs"), "--store", str(tmp_path / "store"),
                     "--config", str(REPO / rt.FINAL_CFG), "--run-id", rid, *GATE_ARGS])
    assert calls.count("export") == 2 and calls.count("verify") == 2 and calls.count("gate") == 2
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["r1", "r2"]


def test_all_without_gate_args_fails_before_any_run_or_stage(tmp_path, monkeypatch):
    """Validasi produksi tetap utuh: tanpa --live/--first-promotion, tidak ada run dibuat dan tidak ada tahap jalan."""
    calls = []
    _install_fake_stages(monkeypatch, calls, fail_train_once=False)
    with pytest.raises(SystemExit, match="tepat satu"):
        rt.main(["all", "--root", str(tmp_path / "runs"), "--store", str(tmp_path / "store"),
                 "--config", str(REPO / rt.FINAL_CFG), "--run-id", "rA"])
    assert calls == [] and not (tmp_path / "runs" / "rA").exists()


# ------------------------------------------------------------------ --threads
def test_threads_option_is_applied_only_when_requested(monkeypatch):
    seen = []
    fake = types.ModuleType("torch")
    fake.set_num_threads = lambda n: seen.append(n)
    monkeypatch.setitem(sys.modules, "torch", fake)
    rt._apply_threads(types.SimpleNamespace(threads=None))
    assert seen == []
    rt._apply_threads(types.SimpleNamespace(threads=2))
    assert seen == [2]
    p = rt.build_parser()
    assert p.parse_args(["all", "--threads", "1"]).threads == 1
    assert p.parse_args(["train", "--run-id", "x", "--threads", "2"]).threads == 2
