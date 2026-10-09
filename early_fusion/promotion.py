"""Gerbang promosi untuk ensemble M6 (Fase 5): aturan yang sama dengan models/train.py::promote_if_better.

Dua gerbang, urutannya sama dengan milik temanmu:
  1. Gerbang keras: Spearman kandidat harus > baseline linear (target ~ log1p(trailing_avg_views)).
     Menangkap run yang rusak total, tidak peduli bagaimana kandidat dibanding model live.
  2. Gerbang lunak: bootstrap berpasangan atas (Spearman baru - Spearman live) di baris test yang
     sama. DITOLAK hanya jika batas ATAS CI 95% < CI_REJECT_MARGIN (yakin lebih buruk dari margin).
     Seri (CI melintasi nol, atau berada di pita negatif kecil sampai margin) tetap PROMOSI:
     "ties go to newer".

Kenapa tidak `from models.train import ...`: modul itu meng-import torch, matplotlib, dotenv,
meminta env POSTGRES_* saat di-import, dan membuat engine DB. Di sini aturan disalin apa adanya
(paired_bootstrap_ci, string alasan, margin, n_bootstrap) dan tests/early_fusion/test_promotion.py
membandingkannya dengan sumber models/train.py supaya tidak menyimpang diam-diam.

Modul ini sengaja TIDAK meng-import torch: murni numpy/scipy/sklearn. Penilaian bundle (yang butuh
torch) ada di early_fusion/retrain.py::stage_gate dan disuntikkan ke sini sebagai array prediksi.

Modul ini hanya MEMUTUSKAN dan MENCATAT. Menyalin bundle ke m6_latest.pt (dan m6_prev.pt untuk
rollback) adalah Fase 6.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Nama env sama dengan milik temanmu supaya satu knob mengatur keduanya.
CI_REJECT_MARGIN = float(os.environ.get("PROMOTION_CI_REJECT_MARGIN", "-0.01"))
N_BOOTSTRAP = int(os.environ.get("PROMOTION_N_BOOTSTRAP", "2000"))
BOOTSTRAP_SEED = int(os.environ.get("PROMOTION_BOOTSTRAP_SEED", "42"))   # train.py memakai env SEED (42)
PROMOTION_LOG_PATH = "experiments/promotions.jsonl"

# String alasan identik dengan models/train.py (log lama dan baru bisa dibaca seragam).
REASON_HARD_GATE = "failed hard gate: did not beat linear (trailing_avg_views) baseline"
REASON_FIRST = "no currently-served bundle to compare against (first promotion)"
REASON_REJECT = "CI confidently below reject margin -- new model is worse than live model"
REASON_PROMOTE = "beat hard gate; not confidently worse than live model (ties go to newer)"


def paired_bootstrap_ci(new_preds, old_preds, targets, n_bootstrap, seed, ci=0.95):
    """CI for (new Spearman - old Spearman) on the shared test set. Each
    resample draws test ROWS (with replacement), not predictions
    independently, so the pairing between the two models is preserved --
    every resample compares both models on the exact same (resampled) rows.
    """
    rng = np.random.default_rng(seed)
    n = len(targets)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        new_corr, _ = spearmanr(new_preds[idx], targets[idx])
        old_corr, _ = spearmanr(old_preds[idx], targets[idx])
        diffs[i] = new_corr - old_corr
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(diffs, [alpha, 1 - alpha])
    return float(lower), float(upper)


def linear_baseline_spearman(train_df, test_df):
    """Baseline yang sama dengan models/baseline.py::run_linear (target ~ log1p(trailing_avg_views)).

    Di sini dipasang pada train+val snapshot baru (kandidat juga dilatih di train+val), dinilai di test.
    """
    from sklearn.linear_model import LinearRegression

    X_train = np.log1p(train_df["trailing_avg_views"].values).reshape(-1, 1)
    X_test = np.log1p(test_df["trailing_avg_views"].values).reshape(-1, 1)
    model = LinearRegression().fit(X_train, train_df["target"].values)
    corr, _ = spearmanr(model.predict(X_test), test_df["target"].values)
    return float(corr)


def decide(*, new_preds, targets, baseline_spearman, old_preds=None, first_promotion=False,
           versioned_path=None, margin=None, n_bootstrap=None, seed=None, extra=None):
    """Putusan promosi. Mengembalikan dict (kunci inti sama dengan promotions.jsonl lama).

    old_preds=None hanya sah bila first_promotion=True (tidak ada bundle live). Tanpa itu: ValueError,
    supaya path live yang salah ketik tidak diam-diam menjadi "promosi pertama".
    """
    margin = CI_REJECT_MARGIN if margin is None else margin
    n_bootstrap = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    seed = BOOTSTRAP_SEED if seed is None else seed
    new_preds, targets = np.asarray(new_preds), np.asarray(targets)
    if old_preds is None and not first_promotion:
        raise ValueError("old_preds wajib diisi kecuali first_promotion=True")
    if old_preds is not None:
        old_preds = np.asarray(old_preds)
        if not (len(new_preds) == len(old_preds) == len(targets)):
            raise ValueError("panjang new_preds/old_preds/targets harus sama (baris test yang sama)")

    new_sp = float(spearmanr(new_preds, targets)[0])
    decision = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "versioned_path": str(versioned_path) if versioned_path is not None else None,
        "new_spearman": new_sp,
        "linear_baseline_spearman": float(baseline_spearman),
    }
    if old_preds is not None:
        decision["old_spearman"] = float(spearmanr(old_preds, targets)[0])

    if not np.isfinite(new_sp) or new_sp <= baseline_spearman:
        decision.update(promoted=False, reason=REASON_HARD_GATE)
    elif old_preds is None:
        decision.update(promoted=True, reason=REASON_FIRST)
    else:
        lower, upper = paired_bootstrap_ci(new_preds, old_preds, targets, n_bootstrap, seed)
        decision.update(ci_lower=lower, ci_upper=upper, ci_reject_margin=margin, n_bootstrap=n_bootstrap)
        if upper < margin:
            decision.update(promoted=False, reason=REASON_REJECT)
        else:
            decision.update(promoted=True, reason=REASON_PROMOTE)
    decision.update(extra or {})
    return decision


def check_no_leak(live_train_end, test_start):
    """Model live tidak boleh pernah melihat baris test: train_end live harus < awal test baru.

    Kalau tidak, perbandingan bias ke model live (ia dinilai di data latihnya sendiri).
    """
    if live_train_end is None:
        raise ValueError("train_end bundle live tidak diketahui")
    if pd.Timestamp(live_train_end) >= pd.Timestamp(test_start):
        raise ValueError(f"bundle live dilatih sampai {live_train_end}, padahal test baru mulai "
                         f"{test_start}: live sudah melihat sebagian test; perbandingan tidak adil")


def log_decision(decision, path=PROMOTION_LOG_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
