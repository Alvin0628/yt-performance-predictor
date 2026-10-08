"""Fase 4: early_fusion/retrain.py (tahap-tahap tanpa torch + logika resume/package dengan dependensi palsu).

    python -m pytest tests/early_fusion/test_retrain.py -q

Tidak butuh torch, CLIP, Postgres, atau MinIO. Tahap train/package/verify yang sebenarnya
(torch) diuji lewat uji coba di PC, bukan di sini.
"""
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from early_fusion import retrain as rt
from early_fusion.data_spec import DataSpec, assign_temporal_split, hash_df
from early_fusion.datasets.token_store import TokenStore
from early_fusion.splits.load_split import apply_split_to_df, load_canonical_split

REPO = Path(__file__).resolve().parent.parent.parent
quiet = lambda *a, **k: None


def make_raw(n=100, seed=0, shuffle=True, with_emb=True):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    df = pd.DataFrame({
        "video_id": [f"v{i:04d}" for i in range(n)],
        "published_at": [t0 + pd.Timedelta(hours=int(h)) for h in np.arange(n) * 5],
        "title": [f"title {i}" for i in range(n)],
        "channel_ref": rng.integers(0, 7, n),
        "views": rng.integers(100, 10_000, n),
        "trailing_avg_views": rng.integers(100, 10_000, n),
        "genre": rng.choice(["a", "b", "c"], n),
    })
    if with_emb:
        df["image_embedding"] = "[0.1,0.2]"
        df["text_embedding"] = "[0.3,0.4]"
    return df.sample(frac=1, random_state=1).reset_index(drop=True) if shuffle else df


def new_run(tmp_path):
    return rt.Run.new(tmp_path / "runs", "r1")


def exported(tmp_path, n=100, **kw):
    run = new_run(tmp_path)
    rt.stage_export(run, read_sql_fn=lambda: make_raw(n), log=quiet, **kw)
    return run


# ------------------------------------------------------------------ resep produksi
def test_production_recipe_matches_live_bundle_sidecar():
    side = json.loads((REPO / "early_fusion/models/final/m6_granular_ensemble_v1.json").read_text())
    cfg = rt.load_production_cfg(REPO / rt.FINAL_CFG)
    assert cfg == side["cfg"], "resep retrain menyimpang dari bundle live"
    assert rt.PRODUCTION_SEEDS == side["member_seeds"]
    assert side["trained_on"] == "train+val"
    assert cfg["gate_lr_mult"] == 0           # final_cfg.json sendiri berisi 10.0: override ini disengaja


def test_refit_epochs_match_live_refit_records_and_train_only_median():
    res = REPO / "early_fusion/results"
    rows = [json.loads(l) for l in (res / "m6_final_runs.jsonl").read_text().splitlines() if l.strip()]
    live = [r for r in rows if r["tag"] == "refit_m6_c24_gateoff" and r["fit"] == "trainval"]
    assert sorted(r["seed"] for r in live) == rt.PRODUCTION_SEEDS
    assert {r["refit_epochs"] for r in live} == {rt.PRODUCTION_REFIT_EPOCHS}, "epoch refit live != resep retrain"
    best = json.loads((res / "summary_final_m6_c24_gateoff.json").read_text())["best_epochs"]
    assert int(np.median(best)) == rt.PRODUCTION_REFIT_EPOCHS      # median best_epoch train-only


def test_snapshot_query_matches_export_script():
    src = (REPO / "scripts/export_snapshot.py").read_text(encoding="utf-8")
    flat = re.sub(r'["\s]+', " ", src)
    for clause in ("label_finalized = true", "image_embedding IS NOT NULL",
                   "trailing_avg_views IS NOT NULL", "ORDER BY published_at, video_id"):
        assert clause in flat and clause in rt.SNAPSHOT_QUERY


def test_parse_overrides():
    assert rt.parse_overrides(["a=1", "b=0.5", "c=x"]) == {"a": 1, "b": 0.5, "c": "x"}


# ------------------------------------------------------------------ export
def test_build_snapshot_frame():
    df = rt.build_snapshot_frame(make_raw(100))
    assert list(df.columns).count("image_embedding") == 0 and "text_embedding" not in df.columns
    assert df["published_at"].is_monotonic_increasing
    assert df["split"].tolist() == assign_temporal_split(100)
    assert df["split"].value_counts().to_dict() == {"train": 80, "val": 10, "test": 10}


def test_build_snapshot_frame_rejects_bad_input():
    with pytest.raises(ValueError, match="duplikat"):
        rt.build_snapshot_frame(pd.concat([make_raw(20), make_raw(20)]))
    with pytest.raises(ValueError, match="kolom wajib"):
        rt.build_snapshot_frame(make_raw(20).drop(columns=["genre"]))
    bad = make_raw(20)
    bad.loc[3, "published_at"] = pd.NaT
    with pytest.raises(ValueError, match="null"):
        rt.build_snapshot_frame(bad)


def test_max_rows_keeps_newest_chronological():
    full = rt.build_snapshot_frame(make_raw(100))
    sub = rt.build_snapshot_frame(make_raw(100), max_rows=30)
    assert len(sub) == 30
    assert sub["video_id"].tolist() == full["video_id"].tolist()[-30:]
    assert sub["split"].tolist() == assign_temporal_split(30)


def test_export_hash_is_of_file_as_read_back(tmp_path):
    run = exported(tmp_path)
    m = run.stage_info("export")
    assert m["snapshot_hash"] == hash_df(pd.read_parquet(run.snapshot_path))
    assert (m["n_total"], m["n_train"], m["n_val"], m["n_test"]) == (100, 80, 10, 10)
    df = pd.read_parquet(run.snapshot_path)
    assert m["train_end"] == str(df.loc[df["split"] != "test", "published_at"].max())
    assert pd.Timestamp(m["test_start"]) > pd.Timestamp(m["train_end"])


def test_export_is_deterministic(tmp_path):
    a = rt.Run.new(tmp_path / "a", "r")
    b = rt.Run.new(tmp_path / "b", "r")
    rt.stage_export(a, read_sql_fn=lambda: make_raw(60, shuffle=True), log=quiet)
    rt.stage_export(b, read_sql_fn=lambda: make_raw(60, shuffle=False), log=quiet)
    assert a.stage_info("export")["snapshot_hash"] == b.stage_info("export")["snapshot_hash"]


def test_export_from_parquet_keeps_hash(tmp_path):
    df = rt.build_snapshot_frame(make_raw(50, with_emb=True), max_rows=None)
    src = tmp_path / "legacy.parquet"
    df.to_parquet(src, index=False)
    run = new_run(tmp_path)
    rt.stage_export(run, from_parquet=src, log=quiet)
    assert run.stage_info("export")["snapshot_hash"] == hash_df(pd.read_parquet(src))


def test_export_from_parquet_rejects_unordered_or_wrong_split(tmp_path):
    df = rt.build_snapshot_frame(make_raw(50))
    bad_order = df.iloc[::-1].reset_index(drop=True)
    p1 = tmp_path / "rev.parquet"
    bad_order.to_parquet(p1, index=False)
    with pytest.raises(SystemExit, match="berurutan"):
        rt.stage_export(rt.Run.new(tmp_path / "x", "r"), from_parquet=p1, log=quiet)
    bad_split = df.copy()
    bad_split["split"] = "train"
    p2 = tmp_path / "split.parquet"
    bad_split.to_parquet(p2, index=False)
    with pytest.raises(SystemExit, match="split"):
        rt.stage_export(rt.Run.new(tmp_path / "y", "r"), from_parquet=p2, log=quiet)
    with pytest.raises(SystemExit, match="max-rows"):
        rt.stage_export(rt.Run.new(tmp_path / "z", "r"), from_parquet=p2, max_rows=10, log=quiet)


# ------------------------------------------------------------------ run + urutan tahap
def test_run_id_collision_and_stage_order(tmp_path):
    run = new_run(tmp_path)
    with pytest.raises(FileExistsError):
        rt.Run.new(tmp_path / "runs", "r1")
    with pytest.raises(SystemExit, match="export"):
        rt.stage_split(run, log=quiet)
    with pytest.raises(SystemExit, match="tidak ditemukan"):
        rt.Run.existing(tmp_path / "runs", "nope")


# ------------------------------------------------------------------ split
def test_split_is_loadable_by_training_code(tmp_path):
    run = exported(tmp_path)
    info = rt.stage_split(run, log=quiet)
    spec = DataSpec(snapshot_path=run.snapshot_path, snapshot_hash=None, split_file=run.split_path,
                    cache_dir=tmp_path, cache_kind="store")
    split = load_canonical_split(verbose=False, spec=spec)
    assert split["snapshot_hash"] == run.stage_info("export")["snapshot_hash"]
    df = pd.read_parquet(run.snapshot_path)
    tr, va, te = apply_split_to_df(df, split)
    assert (len(tr), len(va), len(te)) == (80, 10, 10)
    assert df.loc[te, "published_at"].min() > df.loc[np.concatenate([tr, va]), "published_at"].max()
    assert info["split_hashes"]["test_ids_hash"] == split["test_ids_hash"]


# ------------------------------------------------------------------ tokens
def fake_extractor(calls):
    def extract(ids, titles):
        calls.extend(ids)
        n = len(ids)
        return (np.zeros((n, 3, 4), np.float32), np.zeros((n, 6, 5), np.float32),
                np.ones((n, 6), bool), np.ones(n, bool))
    return extract


def test_tokens_extracts_only_missing_and_is_idempotent(tmp_path):
    run = exported(tmp_path)
    store_dir = tmp_path / "store"
    TokenStore.create_empty(store_dir, L=6, img_tokens=3, img_dim=4, txt_dim=5)
    calls = []
    info = rt.stage_tokens(run, store_dir=store_dir, extractor=fake_extractor(calls), batch=7, log=quiet)
    assert info["n_missing_before"] == 100 and len(calls) == 100
    calls.clear()
    info2 = rt.stage_tokens(run, store_dir=store_dir, extractor=fake_extractor(calls), log=quiet)
    assert info2["n_missing_before"] == 0 and calls == []


def test_tokens_requires_store_unless_create_empty(tmp_path):
    run = exported(tmp_path)
    with pytest.raises(SystemExit, match="token store"):
        rt.stage_tokens(run, store_dir=tmp_path / "none", extractor=fake_extractor([]), log=quiet)


# ------------------------------------------------------------------ train (resume) + package (dependensi palsu)
class FakeDeps:
    """Pengganti torch: 'checkpoint' disimpan sebagai pickle."""

    def __init__(self):
        self.trained = []

    def load_data(self, spec):
        assert spec.cache_kind == "store" and spec.snapshot_hash is None
        return dict(scaler="SCALER", genres_train=["a", "b"], n_cont=5, n_genres=3,
                    split_hashes={"train_ids_hash": "t", "val_ids_hash": "v", "test_ids_hash": "x"}, git_sha="abc")

    def train_one(self, cfg, seed, data, **kw):
        assert kw["fit"] == "trainval" and kw["fixed_epochs"] == rt.PRODUCTION_REFIT_EPOCHS and kw["eval_test"]
        self.trained.append(seed)
        rng = np.random.default_rng(seed)
        tgt = np.linspace(-1, 1, 20)
        return dict(test_spearman=0.3, seconds=1.0, test_preds=tgt + rng.normal(0, 0.3, 20),
                    test_targets=tgt, state_dict={"w": FakeTensor(10 + seed)})

    def metrics(self, p, t):
        from scipy.stats import spearmanr
        return dict(spearman=float(spearmanr(p, t)[0]), auc=0.6, mae=float(np.mean(np.abs(p - t))))

    def save(self, obj, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class FakeTensor:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


@pytest.fixture
def torch_or_stub(monkeypatch):
    """package_final_model.py meng-import torch di tingkat atas; stub hanya bila torch tidak terpasang."""
    try:
        import torch  # noqa: F401
    except ImportError:
        import sys
        import types
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.delitem(sys.modules, "early_fusion.experiments.package_final_model", raising=False)


@pytest.fixture
def staged(tmp_path):
    run = exported(tmp_path)
    store = tmp_path / "store"
    TokenStore.create_empty(store, L=6, img_tokens=3, img_dim=4, txt_dim=5)
    rt.stage_tokens(run, store_dir=store, extractor=fake_extractor([]), log=quiet)
    rt.stage_split(run, log=quiet)
    return run, store


def test_train_resumes_and_refuses_changed_config(staged):
    run, store = staged
    cfg = {"d": 8, "gate_lr_mult": 0}
    deps = FakeDeps()
    rt.stage_train(run, cfg=cfg, seeds=[100, 101], refit_epochs=rt.PRODUCTION_REFIT_EPOCHS, store_dir=store, deps=deps, log=quiet)
    assert deps.trained == [100, 101]
    # simulasi crash setelah seed 101 hilang: hanya seed itu yang diulang
    (run.ckpt_dir / "seed101.pt").unlink()
    rt.stage_train(run, cfg=cfg, seeds=[100, 101], refit_epochs=rt.PRODUCTION_REFIT_EPOCHS, store_dir=store, deps=deps, log=quiet)
    assert deps.trained == [100, 101, 101]
    with pytest.raises(SystemExit, match="berbeda"):
        rt.stage_train(run, cfg={**cfg, "d": 16}, seeds=[100], refit_epochs=rt.PRODUCTION_REFIT_EPOCHS, store_dir=store,
                       deps=deps, log=quiet)


def test_package_writes_metadata_needed_by_serving(staged, torch_or_stub):
    run, store = staged
    deps = FakeDeps()
    rt.stage_train(run, cfg={"d": 8}, seeds=[100, 101, 102], refit_epochs=rt.PRODUCTION_REFIT_EPOCHS, store_dir=store, deps=deps, log=quiet)
    info = rt.stage_package(run, deps=deps, log=quiet)
    blob = deps.load(info["bundle"])
    exp = run.stage_info("export")
    assert blob["format_version"] == 1 and blob["model_class"] == "RATF_M6_Granular_V2"
    assert blob["snapshot_hash"] == exp["snapshot_hash"] and blob["train_end"] == exp["train_end"]
    assert blob["n_rows"] == dict(total=100, train=80, val=10, test=10)
    assert blob["seeds"] == [100, 101, 102] and len(blob["members"]) == 3
    assert "test_mae_ensemble" in blob["refit_metrics"]            # dipakai LoadedRatfBundle untuk pita galat
    side = json.loads(Path(info["bundle"]).with_suffix(".json").read_text())
    assert side["sha256"] == info["sha256"] and side["trained_on"] == "train+val"


def test_package_rejects_inconsistent_checkpoints(staged, torch_or_stub):
    run, store = staged
    deps = FakeDeps()
    rt.stage_train(run, cfg={"d": 8}, seeds=[100, 101], refit_epochs=rt.PRODUCTION_REFIT_EPOCHS, store_dir=store, deps=deps, log=quiet)
    bad = deps.load(run.ckpt_dir / "seed101.pt")
    bad["genres_train"] = ["zzz"]
    deps.save(bad, run.ckpt_dir / "seed101.pt")
    with pytest.raises(SystemExit, match="tidak konsisten"):
        rt.stage_package(run, deps=deps, log=quiet)


# ------------------------------------------------------------------ kontrak stdout
def test_last_stdout_line_is_json_and_satisfies_check_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "_read_sql_from_db", lambda: make_raw(40))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = rt.main(["export", "--root", str(tmp_path / "runs"), "--run-id", "r9"])
    assert rc == 0
    last = buf.getvalue().strip().splitlines()[-1]
    out = json.loads(last)
    assert out["stage"] == "export" and out["status"] == "ok" and out["n_total"] == 40
    assert out["promoted"] is False and "reason" in out and "versioned_path" in out   # kontrak ada di tiap tahap

    # kontrak verify: persis kunci dan format yang dibaca dags/train_model.py::check_promotion
    decision = dict(promoted=False, versioned_path="x.pt", reason="r", new_spearman=0.31234)
    line = json.dumps(decision)
    parsed = json.loads(line)
    assert parsed.get("promoted") is False
    "%s -- %s (new_spearman=%.4f)" % (parsed.get("versioned_path"), parsed.get("reason"), parsed.get("new_spearman", float("nan")))


def test_every_stage_payload_survives_check_promotion_formatting(tmp_path):
    """Meniru persis cara dags/train_model.py::check_promotion memakai XCom."""
    run = new_run(tmp_path)
    for stage in rt.STAGES[:-1]:
        d = json.loads(json.dumps(rt.stage_payload(stage, run, {"n": 1})))
        assert not d.get("promoted")
        "NOT PROMOTED %s -- %s (new_spearman=%.4f)" % (
            d.get("versioned_path"), d.get("reason"), d.get("new_spearman", float("nan")))


def test_parser_requires_run_id_for_later_stages(tmp_path):
    with pytest.raises(SystemExit, match="run-id"):
        rt.main(["split", "--root", str(tmp_path)])
