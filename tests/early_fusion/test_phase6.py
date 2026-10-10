"""Fase 6: gerbang anchor, mekanisme promosi berkas (live_bundle), integrasi stage_gate --promote,
dan pemeriksaan data baru (--min-new-rows). Tanpa torch/Postgres/MinIO/Airflow.

    python -m pytest tests/early_fusion/test_phase6.py -q
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from early_fusion import live_bundle as lb
from early_fusion import promotion as pm
from early_fusion import retrain as rt
from early_fusion.promotion import sha256_file
from tests.early_fusion.test_retrain import (FakeDeps, REPO, gate, gated, make_raw, quiet,  # noqa: F401
                                             torch_or_stub)  # fixture dipakai ulang


def _preds(y, noise, seed):
    return y + np.random.default_rng(seed).normal(0, noise * y.std(), len(y))


@pytest.fixture
def y():
    return np.random.default_rng(0).normal(size=1129)


# ------------------------------------------------------------------ gerbang anchor (murni)
def test_without_anchor_nothing_changes(y):
    d = pm.decide(new_preds=_preds(y, 0.5, 1), old_preds=_preds(y, 0.5, 2), targets=y, baseline_spearman=0.0)
    assert d["promoted"] is True and "anchor_spearman" not in d and "anchor_ci_lower" not in d


def test_candidate_equal_to_anchor_promotes(y):
    p = _preds(y, 0.5, 1)
    d = pm.decide(new_preds=p, old_preds=_preds(y, 0.6, 2), anchor_preds=p.copy(), targets=y, baseline_spearman=0.0)
    assert d["promoted"] is True and d["reason"] == pm.REASON_PROMOTE
    assert (d["anchor_ci_lower"], d["anchor_ci_upper"]) == (0.0, 0.0) and d["anchor_margin"] == 0.0


def test_candidate_worse_than_anchor_is_rejected_even_when_it_beats_live(y):
    """Inti gerbang ke-3: live sudah menurun, kandidat sedikit lebih baik dari live tetapi jelas di bawah anchor."""
    live, cand, anchor = _preds(y, 2.5, 1), _preds(y, 1.2, 2), _preds(y, 0.4, 3)
    d = pm.decide(new_preds=cand, old_preds=live, anchor_preds=anchor, targets=y, baseline_spearman=0.0)
    assert d["new_spearman"] > d["old_spearman"] and d["ci_upper"] >= pm.CI_REJECT_MARGIN   # lolos gerbang lunak
    assert d["promoted"] is False and d["reason"] == pm.REASON_ANCHOR
    assert d["anchor_ci_upper"] < pm.ANCHOR_MARGIN and d["anchor_spearman"] > d["new_spearman"]


def test_anchor_is_not_evaluated_when_earlier_gates_already_reject(y):
    anchor = _preds(y, 0.4, 3)
    hard = pm.decide(new_preds=_preds(y, 30, 1), old_preds=_preds(y, 0.5, 2), anchor_preds=anchor,
                     targets=y, baseline_spearman=0.9)
    assert hard["reason"] == pm.REASON_HARD_GATE and "anchor_ci_lower" not in hard
    soft = pm.decide(new_preds=_preds(y, 2.0, 1), old_preds=_preds(y, 0.3, 2), anchor_preds=anchor,
                     targets=y, baseline_spearman=0.0)
    assert soft["reason"] == pm.REASON_REJECT and "anchor_ci_lower" not in soft


def test_first_promotion_ignores_the_anchor(y):
    d = pm.decide(new_preds=_preds(y, 0.5, 1), anchor_preds=_preds(y, 0.1, 3), targets=y,
                  baseline_spearman=0.0, first_promotion=True)
    assert d["promoted"] is True and d["reason"] == pm.REASON_FIRST and "anchor_ci_lower" not in d


def test_anchor_margin_is_a_knob_and_defaults_to_zero(y):
    assert pm.ANCHOR_MARGIN == 0.0
    live, cand, anchor = _preds(y, 2.5, 1), _preds(y, 1.2, 2), _preds(y, 0.4, 3)
    d = pm.decide(new_preds=cand, old_preds=live, anchor_preds=anchor, targets=y, baseline_spearman=0.0,
                  anchor_margin=-0.9)
    assert d["promoted"] is True and d["anchor_margin"] == -0.9


def test_anchor_length_mismatch_is_an_error(y):
    with pytest.raises(ValueError, match="anchor_preds"):
        pm.decide(new_preds=y, old_preds=y, anchor_preds=y[:-1], targets=y, baseline_spearman=0.0)


# ------------------------------------------------------------------ live_bundle: berkas
def _bundle(path, payload=b"bundle-bytes", **side):
    path.write_bytes(payload)
    meta = {"sha256": sha256_file(path), "train_end": "2026-02-01 00:00:00+00:00", **side}
    lb.sidecar_path(path).write_text(json.dumps(meta))
    return path


def test_bootstrap_creates_latest_and_anchor_without_touching_the_source(tmp_path):
    src = tmp_path / "v1.pt"
    src.write_bytes(b"v1-weights")
    sha_before = sha256_file(src)
    res = lb.bootstrap(src, tmp_path / "bundles", train_end="2026-07-04 22:15:09+00:00", snapshot_n_total=11285)
    assert res == {"m6_latest": "dibuat", "m6_anchor": "dibuat"} and sha256_file(src) == sha_before
    for name, role in (("m6_latest", "live"), ("m6_anchor", "anchor")):
        pt = tmp_path / "bundles" / f"{name}.pt"
        side = lb.read_sidecar(pt)
        assert pt.read_bytes() == b"v1-weights" and side["sha256"] == sha_before and side["role"] == role
        assert side["train_end"] == "2026-07-04 22:15:09+00:00" and side["snapshot_n_total"] == 11285
    assert not list((tmp_path / "bundles").glob("*.new"))
    again = lb.bootstrap(src, tmp_path / "bundles", train_end="x", snapshot_n_total=1)
    assert again == {"m6_latest": "sudah ada (identik)", "m6_anchor": "sudah ada (identik)"}


def test_bootstrap_refuses_to_overwrite_a_different_latest_without_force(tmp_path):
    src = tmp_path / "v1.pt"; src.write_bytes(b"v1")
    b = tmp_path / "bundles"; b.mkdir()
    (b / "m6_latest.pt").write_bytes(b"something else")
    with pytest.raises(FileExistsError, match="--force"):
        lb.bootstrap(src, b, train_end="t", snapshot_n_total=1)
    assert (b / "m6_latest.pt").read_bytes() == b"something else"
    lb.bootstrap(src, b, train_end="t", snapshot_n_total=1, force=True)
    assert (b / "m6_latest.pt").read_bytes() == b"v1"


def test_bootstrap_checks_the_source_sidecar_hash(tmp_path):
    src = _bundle(tmp_path / "v1.pt")
    lb.sidecar_path(src).write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="tidak cocok"):
        lb.bootstrap(src, tmp_path / "b", train_end="t", snapshot_n_total=1)


def test_verify_sidecar_detects_tampering_and_tolerates_missing(tmp_path):
    pt = _bundle(tmp_path / "x.pt")
    assert lb.verify_sidecar(pt)["train_end"] == "2026-02-01 00:00:00+00:00"
    pt.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        lb.verify_sidecar(pt)
    bare = tmp_path / "bare.pt"; bare.write_bytes(b"1")
    assert lb.verify_sidecar(bare) is None


def test_snapshot_n_total_reads_both_formats():
    assert lb.snapshot_n_total({"snapshot_n_total": 11285}) == 11285
    assert lb.snapshot_n_total({"n_rows": {"total": 11700}}) == 11700           # sidecar hasil retrain.py
    assert lb.snapshot_n_total({}) is None and lb.snapshot_n_total(None) is None


def test_promote_first_time_then_rotates_the_old_live_into_prev(tmp_path):
    bundles = tmp_path / "bundles"
    cand1 = _bundle(tmp_path / "c1.pt", b"candidate-one", n_rows={"total": 11500})
    r1 = lb.promote(cand1, bundles)
    assert r1["prev"] is None and (bundles / "m6_latest.pt").read_bytes() == b"candidate-one"
    side = lb.read_sidecar(bundles / "m6_latest.pt")
    assert side["role"] == "live" and "promoted_at" in side and side["snapshot_n_total"] == 11500   # dari n_rows.total
    cand2 = _bundle(tmp_path / "c2.pt", b"candidate-two", n_rows={"total": 11900})
    r2 = lb.promote(cand2, bundles)
    assert (bundles / "m6_latest.pt").read_bytes() == b"candidate-two"
    assert (bundles / "m6_prev.pt").read_bytes() == b"candidate-one" and r2["prev"] == str(bundles / "m6_prev.pt")
    assert lb.read_sidecar(bundles / "m6_prev.pt")["role"] == "prev"
    assert lb.verify_sidecar(bundles / "m6_latest.pt") and lb.verify_sidecar(bundles / "m6_prev.pt")
    assert not list(bundles.glob("*.new"))
    assert lb.status(bundles)["m6_anchor"] is None and lb.status(bundles)["m6_latest"]["sidecar_ok"] is True


def test_promote_refuses_a_candidate_whose_hash_does_not_match_and_leaves_live_alone(tmp_path):
    bundles = tmp_path / "bundles"
    lb.promote(_bundle(tmp_path / "ok.pt", b"live-now"), bundles)
    bad = _bundle(tmp_path / "bad.pt", b"payload")
    bad.write_bytes(b"corrupted after packaging")
    with pytest.raises(ValueError, match="sha256"):
        lb.promote(bad, bundles)
    assert (bundles / "m6_latest.pt").read_bytes() == b"live-now" and not (bundles / "m6_prev.pt").exists()
    with pytest.raises(FileNotFoundError):
        lb.promote(tmp_path / "missing.pt", bundles)
    assert (bundles / "m6_latest.pt").read_bytes() == b"live-now" and not list(bundles.glob("*.new"))


def test_live_bundle_cli_bootstrap_and_status(tmp_path):
    src = tmp_path / "v1.pt"; src.write_bytes(b"w")
    out = io.StringIO()
    with redirect_stdout(out):
        assert lb.main(["bootstrap", "--src", str(src), "--bundles-dir", str(tmp_path / "b"),
                        "--train-end", "2026-07-04 22:15:09+00:00", "--snapshot-n-total", "11285"]) == 0
        assert lb.main(["status", "--bundles-dir", str(tmp_path / "b")]) == 0
    assert "dibuat" in out.getvalue() and '"snapshot_n_total": 11285' in out.getvalue()


# ------------------------------------------------------------------ stage_gate --promote (dependensi palsu)
class Deps(FakeDeps):
    """FakeDeps + model anchor (nama berkas diawali 'anchor')."""
    anchor_noise = 0.05
    anchor_meta = {"train_end": "2000-01-01 00:00:00+00:00"}

    def score(self, path, df, idx, store, device):
        if Path(path).name.startswith("anchor"):
            tgt = df["target"].values[idx]
            rng = np.random.default_rng(3)
            return tgt + rng.normal(0, self.anchor_noise * tgt.std(), len(tgt)), self.anchor_meta
        return super().score(path, df, idx, store, device)


def _use_anchor_deps(g):
    d = Deps()
    d.shuffled = []
    g.deps = d


def _anchor(g, tmp_name="anchor_v1.pt"):
    p = g.live.parent / tmp_name
    p.write_bytes(b"fake-anchor")
    return p


def test_gate_promote_replaces_latest_and_logs_applied(gated, tmp_path):
    bundles = tmp_path / "bundles"
    d = gate(gated, promote=True, bundles_dir=bundles)
    cand = Path(gated.run.stage_info("package")["bundle"])
    assert d["promoted"] is True and d["applied"] is True and d["promote_requested"] is True
    assert (bundles / "m6_latest.pt").read_bytes() == cand.read_bytes()
    assert d["applied_latest"] == str(bundles / "m6_latest.pt") and d["applied_prev"] is None
    row = json.loads(gated.log.read_text().splitlines()[0])
    assert row["applied"] is True and row["promote_requested"] is True
    assert gated.run.stage_info("gate")["applied"] is True


def test_gate_without_promote_never_touches_the_bundles_dir(gated, tmp_path):
    bundles = tmp_path / "bundles"
    d = gate(gated, bundles_dir=bundles)                 # promote tidak diberikan
    assert d["promoted"] is True and d["applied"] is False and d["promote_requested"] is False
    assert not bundles.exists()


def test_gate_promote_does_not_copy_a_rejected_candidate(gated, tmp_path):
    gated.deps.cand_noise = 50.0
    d = gate(gated, promote=True, bundles_dir=tmp_path / "bundles")
    assert d["promoted"] is False and d["applied"] is False and not (tmp_path / "bundles").exists()


def test_gate_promote_refused_for_a_nonstandard_run(gated, tmp_path):
    gated.run.mark("train", {**gated.run.stage_info("train"), "refit_epochs": 1})
    with pytest.raises(SystemExit, match="hanya untuk run standar"):
        gate(gated, promote=True, bundles_dir=tmp_path / "bundles")
    assert not gated.log.exists() and not (tmp_path / "bundles").exists()


def test_gate_apply_failure_is_logged_and_fails_the_task(gated, tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("bukan direktori")
    with pytest.raises(RuntimeError, match="gagal diganti"):
        gate(gated, promote=True, bundles_dir=blocker / "bundles")
    row = json.loads(gated.log.read_text().splitlines()[0])
    assert row["promoted"] is True and row["applied"] is False and "apply_error" in row
    assert "gate" not in gated.run.load_state()["stages"]


def test_gate_second_promotion_keeps_the_previous_live_as_prev(gated, tmp_path):
    bundles = tmp_path / "bundles"
    gate(gated, promote=True, bundles_dir=bundles)
    first = (bundles / "m6_latest.pt").read_bytes()
    d = gate(gated, promote=True, bundles_dir=bundles)
    assert d["applied"] is True and d["applied_prev"] == str(bundles / "m6_prev.pt")
    assert (bundles / "m6_prev.pt").read_bytes() == first


def test_gate_reads_live_train_end_from_the_sidecar_and_rejects_a_tampered_one(gated):
    gated.deps.live_meta = {}                                      # bundle tanpa train_end
    lb.sidecar_path(gated.live).write_text(json.dumps(
        {"train_end": "2000-01-01 00:00:00+00:00", "sha256": sha256_file(gated.live)}))
    assert gate(gated, live_train_end=None)["promoted"] is True    # train_end dari sidecar, tanpa flag
    lb.sidecar_path(gated.live).write_text(json.dumps({"train_end": "2000-01-01 00:00:00+00:00", "sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="sha256"):
        gate(gated, live_train_end=None)


# ------------------------------------------------------------------ stage_gate --anchor
def test_gate_anchor_rejects_a_regression_that_the_live_comparison_misses(gated):
    _use_anchor_deps(gated)
    gated.deps.cand_noise, gated.deps.live_noise = 0.5, 1.0       # kandidat lebih baik dari live...
    anchor = _anchor(gated)                                       # ...tetapi anchor jauh lebih baik
    d = gate(gated, anchor_path=anchor)
    assert d["new_spearman"] > d["old_spearman"] and d["promoted"] is False and d["reason"] == pm.REASON_ANCHOR
    assert d["anchor_sha256"] == sha256_file(anchor) and d["anchor_train_end"].startswith("2000-01-01")
    row = json.loads(gated.log.read_text().splitlines()[0])
    assert row["anchor_ci_upper"] < 0 and row["reason"] == pm.REASON_ANCHOR


def test_gate_anchor_passes_when_candidate_is_at_least_as_good(gated):
    _use_anchor_deps(gated)
    gated.deps.cand_noise, gated.deps.live_noise, gated.deps.anchor_noise = 0.2, 0.2, 0.2
    d = gate(gated, anchor_path=_anchor(gated))
    assert d["promoted"] is True and "anchor_ci_lower" in d


def test_gate_anchor_guards(gated):
    _use_anchor_deps(gated)
    with pytest.raises(SystemExit, match="anchor tidak ditemukan"):
        gate(gated, anchor_path=gated.live.parent / "anchor_missing.pt")
    anchor = _anchor(gated)
    gated.deps.anchor_meta = {"train_end": "2999-01-01 00:00:00+00:00"}
    with pytest.raises(SystemExit, match="anchor"):
        gate(gated, anchor_path=anchor)                            # anchor "sudah melihat" test: tidak adil
    gated.deps.anchor_meta = {}
    with pytest.raises(SystemExit, match="train_end"):
        gate(gated, anchor_path=anchor)


def test_gate_anchor_is_ignored_for_first_promotion(gated):
    _use_anchor_deps(gated)
    d = gate(gated, live_path=None, first_promotion=True, anchor_path=_anchor(gated))
    assert d["promoted"] is True and "anchor_sha256" not in d


# ------------------------------------------------------------------ --min-new-rows
class _Args:
    def __init__(self, live, min_new_rows):
        self.live, self.min_new_rows = live, min_new_rows


def _exported(tmp_path, n_total=1000):
    run = rt.Run.new(tmp_path / "runs", "m1")
    run.mark("export", {"n_total": n_total, "snapshot_hash": "h"})
    return run


def test_maybe_skip_when_too_few_new_videos(tmp_path):
    run = _exported(tmp_path, 1000)
    live = _bundle(tmp_path / "m6_latest.pt", snapshot_n_total=950)
    r = rt._maybe_skip(run, _Args(live, 100), log=quiet)
    assert r["skipped"] is True and r["promoted"] is False and r["applied"] is False and r["n_new"] == 50
    assert run.stage_info("gate")["skipped"] is True and r["stage"] == "skip"


def test_maybe_skip_proceeds_with_enough_new_videos_or_when_disabled_or_unknown(tmp_path):
    run = _exported(tmp_path, 1000)
    live = _bundle(tmp_path / "m6_latest.pt", snapshot_n_total=800)
    assert rt._maybe_skip(run, _Args(live, 100), log=quiet) is None            # 200 baru
    assert rt._maybe_skip(run, _Args(live, 0), log=quiet) is None              # dimatikan
    bare = _bundle(tmp_path / "bare.pt")                                       # tidak mencatat ukuran snapshot
    assert rt._maybe_skip(run, _Args(bare, 100), log=quiet) is None
    after_promotion = _bundle(tmp_path / "after.pt", n_rows={"total": 990})   # format sidecar hasil retrain.py
    assert rt._maybe_skip(run, _Args(after_promotion, 100), log=quiet)["n_new"] == 10


def test_all_with_min_new_rows_stops_after_export_and_prints_the_skip_decision(tmp_path, monkeypatch):
    calls = []

    def fake_export(run, **kw):
        calls.append("export")
        run.mark("export", {"n_total": 1000, "snapshot_hash": "h"})
        return {}

    def must_not_run(name):
        def f(*a, **k):
            raise AssertionError(f"tahap {name} tidak boleh jalan saat dilewati")
        return f
    monkeypatch.setattr(rt, "stage_export", fake_export)
    for n in ("stage_tokens", "stage_split", "stage_train", "stage_package", "stage_verify", "stage_gate"):
        monkeypatch.setattr(rt, n, must_not_run(n))
    live = _bundle(tmp_path / "m6_latest.pt", snapshot_n_total=990)
    out = io.StringIO()
    with redirect_stdout(out):
        rc = rt.main(["all", "--root", str(tmp_path / "runs"), "--store", str(tmp_path / "store"),
                      "--config", str(REPO / rt.FINAL_CFG),
                      "--live", str(live), "--min-new-rows", "100", "--run-id", "rK"])
    last = json.loads(out.getvalue().strip().splitlines()[-1])
    assert rc == 0 and calls == ["export"] and last["skipped"] is True and last["promoted"] is False
    assert last["n_new"] == 10 and "dilewati" in last["reason"]


# ------------------------------------------------------------------ CLI
def test_cli_accepts_the_phase6_flags():
    a = rt.build_parser().parse_args(["all", "--live", "l.pt", "--anchor", "a.pt", "--promote",
                                      "--bundles-dir", "b", "--min-new-rows", "50"])
    assert a.anchor == "a.pt" and a.promote is True and a.bundles_dir == "b" and a.min_new_rows == 50
    g = rt.build_parser().parse_args(["gate", "--run-id", "x", "--live", "l.pt"])
    assert g.anchor is None and g.promote is False and g.bundles_dir == str(lb.DEFAULT_BUNDLES)
