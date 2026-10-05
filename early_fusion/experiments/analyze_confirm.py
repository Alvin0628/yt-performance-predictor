"""Ringkasan Phase 4 (konfirmasi di seed segar) — hanya VAL, tidak menyentuh test.

Lokasi yang disarankan: early_fusion/experiments/analyze_confirm.py

Membaca:  early_fusion/results/m6_final_runs.jsonl  (skor per seed + cfg)
          early_fusion/results/preds/<tag>_seed<N>.npz  (butuh --save-preds saat run)

Mencetak, per kandidat:
  mean ± std per-seed, skor risiko (mean - 0.5*std), Spearman ENSEMBLE (rata-rata prediksi seed),
  selisih ensemble terhadap baseline + selang kepercayaan 95% dari bootstrap-berpasangan atas sampel val
  (menjawab: "apakah selisih ini lebih besar dari noise sampling val yang ~0.03?").

Aturan keputusan (ditetapkan sebelum melihat hasil):
  1. Kandidat terbaik = skor risiko tertinggi di antara --candidates.
  2. Dipakai menggantikan baseline HANYA bila skor risikonya >= skor risiko baseline + 0.005.
     Bila tidak, baseline (default) tetap dipakai.
  3. Pasangan gate-off: bila gate-off >= induknya - 0.005, gate belum terbukti berguna (catat untuk laporan).

Contoh:
  python early_fusion/experiments/analyze_confirm.py \
     --baseline confirm_default \
     --candidates confirm_c24 confirm_c0_stageB confirm_c6 confirm_stageA_t31 \
     --gate-pair confirm_c24 confirm_c24_gateoff --seeds 44 45 46
"""
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

KEYS = ("d", "num_layers", "cross_attn_layers", "ff_mult", "lr", "weight_decay", "dropout", "emb_noise",
        "mod_dropout", "ema_decay", "rank_lambda", "gate_lr_mult", "schedule")


def load_tag(runs, preds_dir, tag, seeds):
    recs = {}
    for r in runs:
        if r["tag"] == tag and r["fit"] == "train" and not r.get("test_eval"):
            recs[r["seed"]] = r                       # run ulang -> ambil yang terakhir
    seeds_ok = [s for s in seeds if s in recs]
    out = dict(tag=tag, seeds=seeds_ok, sp=[recs[s]["val_spearman"] for s in seeds_ok],
               cfg=recs[seeds_ok[0]]["cfg"] if seeds_ok else {}, preds=None, y=None)
    P = []
    for s in seeds_ok:
        f = preds_dir / f"{tag}_seed{s}.npz"
        if not f.exists():
            P = None
            break
        z = np.load(f)
        P.append(z["val_preds"])
        out["y"] = z["val_targets"]
    if P:
        out["preds"] = np.mean(np.stack(P, axis=0), axis=0)
    return out


def risk(sp, lam=0.5):
    a = np.asarray(sp, dtype=float)
    return float(a.mean() - lam * (a.std(ddof=1) if len(a) > 1 else 0.0))


def boot_diff(pa, pb, y, n_boot=2000, seed=0):
    """Bootstrap-berpasangan atas sampel val: Spearman(ens A) - Spearman(ens B)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = spearmanr(pa[idx], y[idx])[0] - spearmanr(pb[idx], y[idx])[0]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="tag lain yang ikut ditampilkan (mis. gate-off, tab-only)")
    ap.add_argument("--gate-pair", nargs=2, metavar=("PARENT", "GATEOFF"), default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46])
    ap.add_argument("--results-dir", default="early_fusion/results")
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    rd = Path(args.results_dir)
    runs = [json.loads(l) for l in open(rd / "m6_final_runs.jsonl") if l.strip()]
    tags = [args.baseline] + [t for t in args.candidates if t != args.baseline] + \
           [t for t in args.extra if t not in args.candidates and t != args.baseline]
    D = {t: load_tag(runs, rd / "preds", t, args.seeds) for t in tags}
    base = D[args.baseline]
    if not base["seeds"]:
        raise SystemExit(f"baseline '{args.baseline}' tidak punya run untuk seed {args.seeds}")

    print(f"\nseed konfirmasi: {args.seeds}   (hanya VAL)\n")
    print(f"{'tag':<26}{'n':>2} {'mean':>7} {'std':>7} {'risk':>7} {'ens':>7} {'Δens vs base [95% CI]':>30}  per-seed")
    for t in tags:
        d = D[t]
        if not d["seeds"]:
            print(f"{t:<26} (tidak ada run)")
            continue
        m, s = float(np.mean(d["sp"])), (float(np.std(d["sp"], ddof=1)) if len(d["sp"]) > 1 else 0.0)
        ens = float(spearmanr(d["preds"], d["y"])[0]) if d["preds"] is not None else float("nan")
        ci = ""
        if t != args.baseline and d["preds"] is not None and base["preds"] is not None and \
                len(d["y"]) == len(base["y"]):
            lo, hi, pg = boot_diff(d["preds"], base["preds"], d["y"], args.n_boot)
            ci = f"{ens - float(spearmanr(base['preds'], base['y'])[0]):+.4f} [{lo:+.4f},{hi:+.4f}] P>0={pg:.2f}"
        print(f"{t:<26}{len(d['sp']):>2} {m:>7.4f} {s:>7.4f} {risk(d['sp']):>7.4f} {ens:>7.4f} {ci:>30}  "
              + " ".join(f"{x:.4f}" for x in d["sp"]))

    print("\nkonfigurasi (kolom utama):")
    for t in tags:
        c = D[t]["cfg"]
        if c:
            print(f"  {t:<26} " + " ".join(f"{k}={c[k]:.4g}" if isinstance(c.get(k), float) else f"{k}={c.get(k)}"
                                           for k in KEYS if k in c))

    cands = [t for t in args.candidates if t != args.baseline and D[t]["seeds"]]
    if cands:
        best = max(cands, key=lambda t: risk(D[t]["sp"]))
        gain = risk(D[best]["sp"]) - risk(base["sp"])
        print(f"\nkandidat terbaik (skor risiko): {best}   selisih terhadap baseline: {gain:+.4f}")
        if gain >= args.margin:
            print(f"KEPUTUSAN: pakai '{best}' (>= baseline + {args.margin}).")
        else:
            print(f"KEPUTUSAN: selisih < {args.margin} -> pakai BASELINE ('{args.baseline}'). "
                  f"Tuning belum terbukti lebih baik di seed segar.")
    if args.gate_pair:
        p, g = args.gate_pair
        if D[p]["seeds"] and D[g]["seeds"]:
            dg = risk(D[g]["sp"]) - risk(D[p]["sp"])
            verdict = "gate BELUM terbukti berguna" if dg >= -args.margin else "gate tampak berguna"
            print(f"\ngate-off vs induk: risk {risk(D[g]['sp']):.4f} vs {risk(D[p]['sp']):.4f} "
                  f"(selisih {dg:+.4f}) -> {verdict}")


if __name__ == "__main__":
    main()
