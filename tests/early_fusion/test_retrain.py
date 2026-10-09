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

    live_meta = {"train_end": "2000-01-01 00:00:00+00:00"}
    cand_noise, live_noise = 0.2, 0.2
    shuffled = []

    def shuffle_labels(self, data, seed):
        self.shuffled.append(seed)

    def score(self, path, df, idx, store, device):
        """Prediksi = target + noise; noise kandidat/live diatur per tes. Meta live = self.live_meta."""
        is_live = Path(path).name.startswith("live")
        tgt = df["target"].values[idx]
        rng = np.random.default_rng(2 if is_live else 1)
        noise = self.live_noise if is_live else self.cand_noise
        return tgt + rng.normal(0, noise * tgt.std(), len(tgt)), (self.live_meta if is_live else {"train_end": "x"})

    def load_data(self, spec):
        assert spec.cache_kind == "store" and spec.snapshot_hash is None
        return dict(store={}, device="cpu", scaler="SCALER", genres_train=["a", "b"], n_cont=5, n_genres=3,
                    split_hashes={"train_ids_hash": "t", "val_ids_hash": "v", "test_ids_hash": "x"}, git_sha="abc")

    def train_one(self, cfg, seed, data, **kw):
        assert kw["fit"] == "trainval" and kw["fixed_epochs"] >= 1 and kw["eval_test"]
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


# ------------------------------------------------------------------ Fase 5: tahap gate
@pytest.fixture
def gated(tmp_path, monkeypatch, torch_or_stub):
    """Run lengkap (data 1000 baris -> test 100) sampai 'verify' ditandai, dengan dependensi palsu."""
    monkeypatch.chdir(REPO)
    run = rt.Run.new(tmp_path / "runs", "g1")
    rt.stage_export(run, read_sql_fn=lambda: make_raw(1000), log=quiet)
    store = tmp_path / "store"
    TokenStore.create_empty(store, L=6, img_tokens=3, img_dim=4, txt_dim=5)
    rt.stage_tokens(run, store_dir=store, extractor=fake_extractor([]), log=quiet)
    rt.stage_split(run, log=quiet)
    deps = FakeDeps()
    deps.shuffled = []                                           # per-instance
    rt.stage_train(run, cfg=rt.load_production_cfg(), seeds=rt.PRODUCTION_SEEDS,
                   refit_epochs=rt.PRODUCTION_REFIT_EPOCHS, store_dir=store, deps=deps, log=quiet)
    rt.stage_package(run, deps=deps, log=quiet)
    run.mark("verify", {"ok": True})                             # verify butuh torch; di sini dianggap lulus
    live = tmp_path / "live_v1.pt"
    live.write_bytes(b"fake-live-bundle")
    return SimpleNamespace(run=run, store=store, deps=deps, live=live, log=tmp_path / "promo.jsonl")


from types import SimpleNamespace   # noqa: E402


def gate(g, **kw):
    args = dict(store_dir=g.store, live_path=g.live, log_path=g.log, deps=g.deps, log=quiet,
                live_train_end="2000-01-01 00:00:00+00:00")
    args.update(kw)
    return rt.stage_gate(g.run, **args)


def test_standard_run_has_no_deviations(gated):
    assert rt.run_deviations(gated.run) == {}


def test_gate_promotes_equal_quality_and_logs_contract(gated):
    d = gate(gated)
    assert d["promoted"] is True and d["stage"] == "gate" and d["run_id"] == "g1"
    for k in ("promoted", "versioned_path", "reason", "new_spearman", "old_spearman", "ci_lower", "ci_upper",
              "linear_baseline_spearman", "live_sha256", "candidate_sha256", "n_test", "snapshot_hash"):
        assert k in d, k
    assert d["n_test"] == 100 and d["baseline_fit"] == "train+val"
    rows = [json.loads(l) for l in gated.log.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["promoted"] is True


def test_gate_rejects_broken_candidate_via_hard_gate(gated):
    gated.deps.cand_noise = 50.0                                  # prediksi ~ noise murni
    d = gate(gated)
    assert d["promoted"] is False and d["reason"] == rt.pm.REASON_HARD_GATE


def test_gate_rejects_confidently_worse_candidate_via_ci(gated):
    gated.deps.cand_noise, gated.deps.live_noise = 0.9, 0.05
    d = gate(gated)
    assert d["new_spearman"] > d["linear_baseline_spearman"]
    assert d["promoted"] is False and d["reason"] == rt.pm.REASON_REJECT


def test_gate_first_promotion_requires_explicit_flag(gated):
    with pytest.raises(SystemExit, match="tepat satu"):
        gate(gated, live_path=None)
    with pytest.raises(SystemExit, match="tepat satu"):
        gate(gated, first_promotion=True)                          # live DAN first-promotion sekaligus
    with pytest.raises(SystemExit, match="tidak ditemukan"):
        gate(gated, live_path=gated.live.parent / "live_typo.pt")
    d = gate(gated, live_path=None, first_promotion=True)
    assert d["promoted"] is True and d["reason"] == rt.pm.REASON_FIRST and d["live_path"] is None


def test_gate_refuses_when_live_may_have_seen_test(gated):
    gated.deps.live_meta = {}                                      # bundle lama: hanya flag yang tersedia
    with pytest.raises(SystemExit, match="tidak adil"):
        gate(gated, live_train_end="2999-01-01 00:00:00+00:00")


def test_gate_needs_live_train_end_for_legacy_bundle(gated):
    gated.deps.live_meta = {}                                      # bundle lama tanpa train_end
    with pytest.raises(SystemExit, match="--live-train-end"):
        gate(gated, live_train_end=None)
    assert gate(gated)["promoted"] is True                         # flag diberikan -> jalan
    gated.deps.live_meta = {"train_end": "2999-01-01 00:00:00+00:00"}      # meta menang atas flag
    with pytest.raises(SystemExit, match="tidak adil"):
        gate(gated)


def test_nonstandard_run_cannot_log_to_production_file(gated):
    gated.run.mark("train", {**gated.run.stage_info("train"), "refit_epochs": 1})
    assert rt.run_deviations(gated.run) == {"refit_epochs": 1}
    with pytest.raises(SystemExit, match="menyimpang"):
        gate(gated, log_path=rt.pm.PROMOTION_LOG_PATH)
    d = gate(gated)                                                # ke berkas lain: boleh, dan diberi label
    assert d["deviations"] == {"refit_epochs": 1}


def test_shuffle_labels_marks_run_and_bundle_as_sabotage(tmp_path, monkeypatch, torch_or_stub):
    monkeypatch.chdir(REPO)
    run = rt.Run.new(tmp_path / "runs", "s1")
    rt.stage_export(run, read_sql_fn=lambda: make_raw(200), log=quiet)
    store = tmp_path / "store"
    TokenStore.create_empty(store, L=6, img_tokens=3, img_dim=4, txt_dim=5)
    rt.stage_tokens(run, store_dir=store, extractor=fake_extractor([]), log=quiet)
    rt.stage_split(run, log=quiet)
    deps = FakeDeps()
    deps.shuffled = []
    kw = dict(cfg=rt.load_production_cfg(), seeds=[100], refit_epochs=rt.PRODUCTION_REFIT_EPOCHS, store_dir=store,
              deps=deps, log=quiet)
    rt.stage_train(run, shuffle_labels=7, **kw)
    assert deps.shuffled == [7] and run.stage_info("train")["sabotage"] == {"shuffle_labels": 7}
    info = rt.stage_package(run, deps=deps, log=quiet)
    assert deps.load(info["bundle"])["sabotage"] == {"shuffle_labels": 7}
    assert "sabotage" in rt.run_deviations(run)
    with pytest.raises(SystemExit, match="berbeda"):               # ckpt sabotase tidak boleh dilanjutkan tanpa flag
        rt.stage_train(run, shuffle_labels=None, **kw)


def test_all_fails_fast_without_live_args(tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(SystemExit, match="tepat satu"):
        rt.main(["all", "--root", str(root)])
    assert not root.exists() or not any(root.iterdir())            # tidak ada run yang dibuat


def test_cli_exposes_gate_and_shuffle_flags():
    a = rt.build_parser().parse_args(["gate", "--run-id", "r", "--live", "x.pt", "--live-train-end", "t"])
    assert (a.stage, a.live, a.first_promotion, a.live_train_end) == ("gate", "x.pt", False, "t")
    b = rt.build_parser().parse_args(["train", "--run-id", "r", "--shuffle-labels", "5"])
    assert b.shuffle_labels == 5


# ------------------------------------------------------------------ Fase 5: skrip simulasi (dependensi palsu)
class SimDeps(FakeDeps):
    """Kualitas kandidat mengikuti bundle-nya: 1 epoch atau sabotase -> prediksi noise murni."""

    def __init__(self):
        super().__init__()
        self.shuffled = []

    def score(self, path, df, idx, store, device):
        if Path(path).name.startswith("live"):
            return super().score(path, df, idx, store, device)
        blob = self.load(path)
        broken = blob["refit_epochs"] == 1 or blob.get("sabotage")
        tgt = df["target"].values[idx]
        noise = 50.0 if broken else 0.2
        return tgt + np.random.default_rng(1).normal(0, noise * tgt.std(), len(tgt)), {"train_end": "x"}


def test_simulation_covers_mentor_scenarios_and_checks_live_untouched(tmp_path, monkeypatch, torch_or_stub, capsys):
    from early_fusion.experiments import simulate_promotion_gate as sim
    assert set(sim.DEFAULT_CASES) == {"seed_berbeda", "satu_epoch", "label_acak"}
    assert sim.CASES["seed_berbeda"]["expect"] is True and sim.CASES["satu_epoch"]["expect"] is False
    assert sim.CASES["label_acak"]["expect"] is False

    monkeypatch.chdir(REPO)
    snap = tmp_path / "legacy.parquet"
    rt.build_snapshot_frame(make_raw(1000)).to_parquet(snap, index=False)
    store = tmp_path / "store"
    TokenStore.create_empty(store, L=6, img_tokens=3, img_dim=4, txt_dim=5)
    seed_run = rt.Run.new(tmp_path / "seedrun", "t")                       # isi token store dulu
    rt.stage_export(seed_run, from_parquet=snap, log=quiet)
    rt.stage_tokens(seed_run, store_dir=store, extractor=fake_extractor([]), log=quiet)
    live = tmp_path / "live_v1.pt"
    live.write_bytes(b"fake-live")

    monkeypatch.setattr(rt, "_torch_deps", lambda: SimDeps())
    monkeypatch.setattr(rt, "stage_verify", lambda run, store_dir, log=quiet: run.mark("verify", {"ok": True}))
    log = tmp_path / "sim.jsonl"
    rc = sim.main(["--live", str(live), "--live-train-end", "2000-01-01 00:00:00+00:00", "--snapshot", str(snap),
                   "--store", str(store), "--root", str(tmp_path / "runs"), "--log-path", str(log),
                   "--cases", "seed_berbeda", "satu_epoch", "label_acak", "identik"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["all_ok"] and out["live_unchanged"]
    got = {c["case"]: c["promoted"] for c in out["cases"]}
    assert got == {"seed_berbeda": True, "satu_epoch": False, "label_acak": False, "identik": True}
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 4 and all("deviations" in r or r["promoted"] for r in rows)
    assert live.read_bytes() == b"fake-live"


# ------------------------------------------------------------------ Phase 4.1 hardening: max_fail_frac, --threads, resume `all`
def test_max_fail_frac_is_passed_to_update_store(tmp_path, monkeypatch):
    from early_fusion.datasets import token_update
    seen = {}
    real = token_update.update_store

    def spy(store, df, extractor, **kw):
        seen.update(kw)
        return real(store, df, extractor, **{k: v for k, v in kw.items() if k != "max_fail_frac"})

    monkeypatch.setattr(token_update, "update_store", spy)
    for frac, expect in ((None, 0.05), (0.2, 0.2)):
        run = rt.Run.new(tmp_path / f"r{expect}", "r")
        rt.stage_export(run, read_sql_fn=lambda: make_raw(30), log=quiet)
        store = tmp_path / f"s{expect}"
        TokenStore.create_empty(store, L=6, img_tokens=3, img_dim=4, txt_dim=5)
        kw = {} if frac is None else {"max_fail_frac": frac}
        rt.stage_tokens(run, store_dir=store, extractor=fake_extractor([]), log=quiet, **kw)
        assert seen["max_fail_frac"] == expect


def test_cli_hardening_flags_parse():
    p = rt.build_parser()
    assert p.parse_args(["tokens", "--run-id", "r"]).max_fail_frac == 0.05
    assert p.parse_args(["tokens", "--run-id", "r", "--max-fail-frac", "0.1"]).max_fail_frac == 0.1
    assert p.parse_args(["train", "--run-id", "r", "--threads", "3"]).threads == 3
    a = p.parse_args(["all", "--threads", "2", "--max-fail-frac", "0.3", "--live", "x.pt", "--shuffle-labels", "9"])
    assert (a.threads, a.max_fail_frac, a.live, a.shuffle_labels) == (2, 0.3, "x.pt", 9)


def test_apply_threads_sets_torch_threads(monkeypatch):
    import sys
    import types
    calls = []
    fake = types.ModuleType("torch")
    fake.set_num_threads = lambda n: calls.append(n)
    monkeypatch.setitem(sys.modules, "torch", fake)
    rt._apply_threads(SimpleNamespace(threads=3))
    rt._apply_threads(SimpleNamespace(threads=None))           # tidak diset -> tidak dipanggil
    assert calls == [3]


@pytest.fixture
def recorded_all(monkeypatch):
    """Ganti semua tahap dengan perekam; `all` hanya diuji urutan/lewati/lanjutkan-nya."""
    calls = []

    def make(name, ret):
        def fn(run, **kw):
            calls.append((name, kw.get("max_fail_frac"), kw.get("shuffle_labels")))
            run.mark(name, ret or {"fake": True})        # stage_gate asli menyimpan keputusan lengkapnya
            return ret
        return fn

    decision = {"promoted": True, "versioned_path": "b.pt", "reason": "r", "new_spearman": 0.4,
                "stage": "gate", "run_id": "x"}
    for name in ("export", "tokens", "split", "train", "package", "verify"):
        monkeypatch.setattr(rt, f"stage_{name}", make(name, {}))
    monkeypatch.setattr(rt, "stage_gate", make("gate", decision))
    return calls, decision


def _all_args(tmp_path, *extra):
    return rt.build_parser().parse_args(["all", "--root", str(tmp_path / "runs"), "--first-promotion", *extra])


def test_all_runs_every_stage_in_order_including_gate(tmp_path, recorded_all):
    calls, decision = recorded_all
    out = rt.run_stage(_all_args(tmp_path, "--run-id", "a1", "--max-fail-frac", "0.2", "--shuffle-labels", "5"))
    assert [c[0] for c in calls] == ["export", "tokens", "split", "train", "package", "verify", "gate"]
    assert out == decision
    assert dict((c[0], c[1]) for c in calls)["tokens"] == 0.2           # max_fail_frac sampai ke tahap tokens
    assert dict((c[0], c[2]) for c in calls)["train"] == 5              # shuffle_labels sampai ke tahap train


def test_all_resumes_skipping_completed_stages(tmp_path, recorded_all):
    calls, _ = recorded_all
    run = rt.Run.new(tmp_path / "runs", "a2")
    for s in ("export", "tokens", "split"):
        run.mark(s, {"done": True})
    rt.run_stage(_all_args(tmp_path, "--run-id", "a2"))
    assert [c[0] for c in calls] == ["train", "package", "verify", "gate"]


def test_all_resume_runs_only_gate_when_verify_already_done(tmp_path, recorded_all):
    calls, _ = recorded_all
    run = rt.Run.new(tmp_path / "runs", "a3")
    for s in ("export", "tokens", "split", "train", "package", "verify"):
        run.mark(s, {"done": True})
    rt.run_stage(_all_args(tmp_path, "--run-id", "a3"))
    assert [c[0] for c in calls] == ["gate"]


def test_all_on_finished_run_echoes_stored_decision_without_rerunning(tmp_path, recorded_all):
    calls, decision = recorded_all
    first = rt.run_stage(_all_args(tmp_path, "--run-id", "a4"))
    calls.clear()
    again = rt.run_stage(_all_args(tmp_path, "--run-id", "a4"))
    assert calls == [] and again == first == decision
    assert "finished_at" not in again


def test_all_still_fails_fast_on_resume_without_live_args(tmp_path, recorded_all):
    calls, _ = recorded_all
    args = rt.build_parser().parse_args(["all", "--root", str(tmp_path / "runs"), "--run-id", "a5"])
    with pytest.raises(SystemExit, match="tepat satu"):
        rt.run_stage(args)
    assert calls == [] and not (tmp_path / "runs" / "a5").exists()


def test_single_gate_stage_is_wired_to_cli(tmp_path, recorded_all):
    calls, decision = recorded_all
    run = rt.Run.new(tmp_path / "runs", "a6")
    args = rt.build_parser().parse_args(["gate", "--root", str(tmp_path / "runs"), "--run-id", "a6",
                                         "--first-promotion"])
    assert rt.run_stage(args) == decision and [c[0] for c in calls] == ["gate"]
