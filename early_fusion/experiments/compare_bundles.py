"""Diagnosis: kandidat retrain vs bundle live pada data yang SAMA (read-only, tidak menulis apa pun).

Simpan sebagai early_fusion/experiments/compare_bundles.py, jalankan dari akar repo:

    python -m early_fusion.experiments.compare_bundles ^
        --run-id 20261008T102701Z ^
        --live early_fusion/models/final/m6_granular_ensemble_v1.pt

Yang dicetak: (1) metadata kedua bundle berdampingan (cfg, seed, epoch, hash snapshot/split),
(2) Spearman test tiap anggota dan ensemble, (3) per seed yang sama: korelasi dan selisih
maksimum prediksi kandidat vs live. Kalau resep, data, dan seed sama dan training deterministik,
(3) harus hampir identik. Selisih besar = ada perbedaan nyata yang harus dicari akarnya.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from early_fusion.experiments.m6_core import iterate_batches, load_data
from early_fusion.models.m6_ensemble import M6Ensemble
from early_fusion.retrain import DEFAULT_ROOT, DEFAULT_STORE, Run


def member_preds(path, data):
    ens = M6Ensemble.load(path, device=data["device"])
    parts, ys = [], []
    for b in iterate_batches(data["store"], data["test_idx"], 256, data["device"], shuffle=False):
        parts.append(ens.predict_batch(b["image_tokens"], b["text_tokens"], b["text_mask"],
                                       b["tabular"], b["genre_idx"]))
        ys.append(b["target"].cpu().numpy())
    return ens, np.concatenate(parts, axis=1), np.concatenate(ys)


def sp(a, b):
    return float(spearmanr(a, b)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--live", required=True, help="path bundle live (.pt)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    args = ap.parse_args()

    run = Run.existing(args.root, args.run_id)
    cand_path = Path(run.stage_info("package")["bundle"])
    data = load_data(verbose=False, spec=run.spec(args.store))

    cand, cp, y = member_preds(cand_path, data)
    live, lp, y2 = member_preds(Path(args.live), data)
    assert np.allclose(y, y2)

    print("== METADATA (kandidat | live) ==")
    for k in ("snapshot_hash", "seeds", "fit", "variant", "refit_epochs", "split_hashes"):
        a, b = cand.meta.get(k), live.meta.get(k)
        print(f"{k:14s}: {a} | {b}   {'SAMA' if a == b else '<-- BEDA'}")
    ca, cb = cand.meta.get("cfg", {}), live.meta.get("cfg", {})
    diff = {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
    print(f"cfg beda      : {diff or 'tidak ada'}")

    print("\n== SPEARMAN TEST (data test sama, n=%d) ==" % len(y))
    print(f"ensemble      : kandidat {sp(cp.mean(0), y):.4f} | live {sp(lp.mean(0), y):.4f}")
    cs, ls = cand.meta.get("seeds", []), live.meta.get("seeds", [])
    for i, s in enumerate(cs):
        line = f"seed {s}: kandidat {sp(cp[i], y):.4f}"
        if s in ls:
            j = ls.index(s)
            line += (f" | live {sp(lp[j], y):.4f} | korelasi pred {pearsonr(cp[i], lp[j])[0]:.4f}"
                     f" | selisih maks {np.max(np.abs(cp[i] - lp[j])):.4f}")
        print(line)


if __name__ == "__main__":
    main()
