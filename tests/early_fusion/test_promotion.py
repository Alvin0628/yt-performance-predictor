"""Fase 5: early_fusion/promotion.py (aturan gerbang; tanpa torch).

    python -m pytest tests/early_fusion/test_promotion.py -q
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from early_fusion import promotion as pm

REPO = Path(__file__).resolve().parent.parent.parent
TRAIN_SRC = (REPO / "models/train.py").read_text()


def _preds(targets, noise, seed):
    rng = np.random.default_rng(seed)
    return targets + rng.normal(0, noise * targets.std(), len(targets))


@pytest.fixture
def y():
    return np.random.default_rng(0).normal(size=1129)


# ------------------------------------------------------------------ tidak menyimpang dari milik teman
def test_paired_bootstrap_ci_identical_to_models_train():
    """Ambil fungsi persis dari sumber models/train.py (tanpa meng-import modul berat itu)."""
    tree = ast.parse(TRAIN_SRC)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "paired_bootstrap_ci")
    ns = {"np": np, "spearmanr": spearmanr}
    exec(compile(ast.Module([fn], []), "train_src", "exec"), ns)
    rng = np.random.default_rng(1)
    t = rng.normal(size=300)
    a, b = t + rng.normal(0, 0.8, 300), t + rng.normal(0, 1.0, 300)
    assert ns["paired_bootstrap_ci"](a, b, t, 200, seed=42) == pm.paired_bootstrap_ci(a, b, t, 200, seed=42)


def test_rule_constants_and_reasons_match_models_train():
    assert 'os.environ.get("PROMOTION_CI_REJECT_MARGIN", "-0.01")' in TRAIN_SRC and pm.CI_REJECT_MARGIN == -0.01
    assert 'os.environ.get("PROMOTION_N_BOOTSTRAP", "2000")' in TRAIN_SRC and pm.N_BOOTSTRAP == 2000
    for reason in (pm.REASON_HARD_GATE, pm.REASON_FIRST, pm.REASON_REJECT, pm.REASON_PROMOTE):
        assert reason in TRAIN_SRC, reason
    assert "if upper < CI_REJECT_MARGIN:" in TRAIN_SRC          # aturan tolak satu arah: batas ATAS < margin


# ------------------------------------------------------------------ putusan
def test_identical_candidate_is_a_tie_and_promotes(y):
    p = _preds(y, 0.5, 1)
    d = pm.decide(new_preds=p, old_preds=p.copy(), targets=y, baseline_spearman=0.1)
    assert d["promoted"] is True and d["reason"] == pm.REASON_PROMOTE
    assert (d["ci_lower"], d["ci_upper"]) == (0.0, 0.0)


def test_different_seed_same_quality_promotes(y):
    d = pm.decide(new_preds=_preds(y, 0.8, 1), old_preds=_preds(y, 0.8, 2), targets=y, baseline_spearman=0.1)
    assert d["promoted"] is True and d["ci_upper"] >= pm.CI_REJECT_MARGIN


def test_confidently_worse_is_rejected_by_ci(y):
    d = pm.decide(new_preds=_preds(y, 2.0, 1), old_preds=_preds(y, 0.3, 2), targets=y, baseline_spearman=0.0)
    assert d["new_spearman"] > 0 and d["promoted"] is False and d["reason"] == pm.REASON_REJECT
    assert d["ci_upper"] < pm.CI_REJECT_MARGIN


def test_slightly_worse_inside_margin_still_promotes(y):
    """Perilaku yang disengaja milik temanmu: hanya 'yakin lebih buruk dari margin' yang ditolak."""
    old = _preds(y, 0.6, 2)
    new = old + np.random.default_rng(3).normal(0, 0.12 * y.std(), len(y))        # sedikit lebih bising
    d = pm.decide(new_preds=new, old_preds=old, targets=y, baseline_spearman=0.0)
    assert d["new_spearman"] < d["old_spearman"] and d["ci_upper"] >= pm.CI_REJECT_MARGIN
    assert d["promoted"] is True


def test_margin_is_a_real_knob(y):
    old, new = _preds(y, 0.6, 2), _preds(y, 0.6, 1)                 # kualitas sama, seed beda = seri
    assert pm.decide(new_preds=new, old_preds=old, targets=y, baseline_spearman=0.0)["promoted"] is True
    assert pm.decide(new_preds=new, old_preds=old, targets=y, baseline_spearman=0.0, margin=0.5)["promoted"] is False


def test_hard_gate_beats_a_better_than_live_candidate(y):
    d = pm.decide(new_preds=_preds(y, 3.0, 1), old_preds=_preds(y, 30.0, 2), targets=y, baseline_spearman=0.9)
    assert d["promoted"] is False and d["reason"] == pm.REASON_HARD_GATE
    assert "ci_lower" not in d and "old_spearman" in d       # CI dilewati seperti milik teman; old tetap dicatat


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_nonfinite_new_spearman_is_rejected(y):
    d = pm.decide(new_preds=np.zeros(len(y)), old_preds=_preds(y, 0.5, 2), targets=y, baseline_spearman=0.0)
    assert d["promoted"] is False and d["reason"] == pm.REASON_HARD_GATE


def test_first_promotion_must_be_explicit(y):
    p = _preds(y, 0.5, 1)
    with pytest.raises(ValueError, match="first_promotion"):
        pm.decide(new_preds=p, targets=y, baseline_spearman=0.1)
    d = pm.decide(new_preds=p, targets=y, baseline_spearman=0.1, first_promotion=True)
    assert d["promoted"] is True and d["reason"] == pm.REASON_FIRST and "old_spearman" not in d


def test_length_mismatch_is_an_error(y):
    with pytest.raises(ValueError, match="panjang"):
        pm.decide(new_preds=y[:-1], old_preds=y, targets=y, baseline_spearman=0.0)


def test_extra_fields_are_added_after_core_keys(y):
    d = pm.decide(new_preds=y, old_preds=y, targets=y, baseline_spearman=0.0, extra={"run_id": "r1"})
    assert d["run_id"] == "r1" and d["promoted"] is True


# ------------------------------------------------------------------ log + kebocoran + baseline
def test_log_rows_are_valid_jsonl_and_superset_of_historical_keys(tmp_path, y):
    old_row = json.loads((REPO / "experiments/promotions.jsonl").read_text().splitlines()[0])
    d = pm.decide(new_preds=_preds(y, 0.8, 1), old_preds=_preds(y, 0.8, 2), targets=y,
                  baseline_spearman=0.1, versioned_path=tmp_path / "b.pt")
    log = tmp_path / "sub" / "p.jsonl"
    pm.log_decision(d, log)
    pm.log_decision(d, log)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 2 and set(old_row) <= set(rows[0])


def test_check_no_leak():
    pm.check_no_leak("2026-07-04 22:15:09+00:00", "2026-07-05 00:00:02+00:00")
    with pytest.raises(ValueError, match="tidak adil"):
        pm.check_no_leak("2026-07-05 00:00:02+00:00", "2026-07-05 00:00:02+00:00")
    with pytest.raises(ValueError, match="tidak diketahui"):
        pm.check_no_leak(None, "2026-07-05 00:00:02+00:00")


def test_linear_baseline_spearman():
    n = 200
    tv = np.random.default_rng(0).integers(100, 10_000, n)
    df = pd.DataFrame({"trailing_avg_views": tv, "target": 2 * np.log1p(tv)})
    assert pm.linear_baseline_spearman(df.iloc[:150], df.iloc[150:]) == pytest.approx(1.0)


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert pm.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
