"""Gerbang Fase 5: simulasi gerbang promosi di data yang ada (snapshot lama + bundle live).

Skenario dari mentor, dijalankan penuh (export -> tokens -> split -> train -> package -> verify -> gate)
terhadap bundle live yang SAMA:

  seed_berbeda  seed 200..204, konfigurasi sama           -> HARUS LOLOS (seri berarti yang baru)
  satu_epoch    refit 1 epoch                             -> HARUS DITOLAK (kandidat sengaja dirusak)
  label_acak    target train+val diacak (2 seed)          -> HARUS DITOLAK (kandidat sengaja dirusak)
  identik       seed 100..104 persis seperti live (opsional, --cases ... identik) -> seri, LOLOS

Keputusan dicatat ke data_snapshots/promotions_sim.jsonl (BUKAN experiments/promotions.jsonl produksi;
gate menolak mencatat run non-standar ke berkas produksi). Bundle live hanya dibaca: sha256-nya
dibandingkan sebelum dan sesudah, dan skrip gagal kalau berubah.

    python -m early_fusion.experiments.simulate_promotion_gate ^
        --live early_fusion/models/final/m6_granular_ensemble_v1.pt ^
        --live-train-end "2026-07-04 22:15:09+00:00"

Baris terakhir stdout: JSON {"all_ok": ..., "cases": [...]}; exit code 1 bila ada skenario yang
hasilnya tidak sesuai harapan.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_fusion import promotion as pm
from early_fusion import retrain as rt

CASES = {
    "seed_berbeda": dict(seeds=[200, 201, 202, 203, 204], refit_epochs=None, shuffle=None, expect=True,
                         why="seed berbeda, konfigurasi sama: seri/lebih baik -> lolos"),
    "satu_epoch": dict(seeds=None, refit_epochs=1, shuffle=None, expect=False,
                       why="kandidat 1 epoch -> ditolak"),
    "label_acak": dict(seeds=[100, 101], refit_epochs=None, shuffle=1234, expect=False,
                       why="label train+val diacak -> ditolak"),
    "identik": dict(seeds=None, refit_epochs=None, shuffle=None, expect=True,
                    why="identik dengan live: CI [0,0] = seri -> lolos"),
}
DEFAULT_CASES = ["seed_berbeda", "satu_epoch", "label_acak"]


def run_case(name, spec, args, deps):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_sim_{name}"
    run = rt.Run.new(args.root, run_id)
    print(f"\n===== skenario {name}: {spec['why']} (run {run_id}) =====", flush=True)
    rt.stage_export(run, from_parquet=args.snapshot)
    rt.stage_tokens(run, store_dir=args.store)
    rt.stage_split(run)
    rt.stage_train(run, cfg=rt.load_production_cfg(), seeds=spec["seeds"] or rt.PRODUCTION_SEEDS,
                   refit_epochs=spec["refit_epochs"] or rt.PRODUCTION_REFIT_EPOCHS, store_dir=args.store,
                   shuffle_labels=spec["shuffle"], deps=deps)
    rt.stage_package(run, deps=deps)
    rt.stage_verify(run, store_dir=args.store)
    return rt.stage_gate(run, store_dir=args.store, live_path=args.live, live_train_end=args.live_train_end,
                         log_path=args.log_path, deps=deps)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True, help="bundle live (hanya dibaca)")
    ap.add_argument("--live-train-end", default=None, help="wajib bila bundle live lama tanpa train_end")
    ap.add_argument("--snapshot", default="data_snapshots/snapshot.parquet")
    ap.add_argument("--store", default=str(rt.DEFAULT_STORE))
    ap.add_argument("--root", default=str(rt.DEFAULT_ROOT))
    ap.add_argument("--log-path", default="data_snapshots/promotions_sim.jsonl")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES, choices=sorted(CASES))
    args = ap.parse_args(argv)

    rt.check_gate_args(args.live, False)
    live_sha_before = pm.sha256_file(args.live)
    deps = rt._torch_deps()

    results = []
    for name in args.cases:
        d = run_case(name, CASES[name], args, deps)
        results.append(dict(case=name, expected_promoted=CASES[name]["expect"], promoted=d["promoted"],
                            ok=d["promoted"] == CASES[name]["expect"], new_spearman=d["new_spearman"],
                            old_spearman=d.get("old_spearman"), ci_lower=d.get("ci_lower"),
                            ci_upper=d.get("ci_upper"), reason=d["reason"], run_id=d["run_id"]))

    live_unchanged = pm.sha256_file(args.live) == live_sha_before
    print("\n" + "=" * 100)
    print(f"{'skenario':14s} {'harap':8s} {'hasil':9s} {'new':>7s} {'old':>7s} {'CI95 (new-old)':>20s}  alasan")
    for r in results:
        ci = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]" if r["ci_lower"] is not None else "-"
        print(f"{r['case']:14s} {'LOLOS' if r['expected_promoted'] else 'TOLAK':8s} "
              f"{'LOLOS' if r['promoted'] else 'DITOLAK':9s} {r['new_spearman']:7.4f} "
              f"{(r['old_spearman'] if r['old_spearman'] is not None else float('nan')):7.4f} {ci:>20s}  "
              f"{'OK' if r['ok'] else '<-- TIDAK SESUAI'}  {r['reason'][:48]}")
    print(f"bundle live tidak berubah (sha256): {'OK' if live_unchanged else 'GAGAL'}")
    all_ok = all(r["ok"] for r in results) and live_unchanged
    print("HASIL GERBANG FASE 5: " + ("LULUS" if all_ok else "GAGAL"))
    print(json.dumps({"all_ok": all_ok, "live_unchanged": live_unchanged, "cases": results}, default=str), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
