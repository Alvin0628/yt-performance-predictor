"""Jalankan satu konfigurasi M6 granular v2 pada beberapa seed: konfirmasi, final, ablasi, refit.

Lokasi yang disarankan: early_fusion/experiments/run_m6_config.py

Mode pemakaian:
  # Phase 0 — reproduksi config C lama (tanpa --config => DEFAULT_CFG), val saja
  python run_m6_config.py --seeds 42 43 44 --tag p0_repro

  # Phase 4 — konfirmasi di seed segar, val saja
  python run_m6_config.py --config early_fusion/results/optuna/stageC_full_top3.json:0 \
         --seeds 44 45 46 --tag confirm_top1

  # Phase 5 — final train-only: test dibuka SEKALI per --test-role
  python run_m6_config.py --config final_cfg.json --seeds 100 101 102 103 104 \
         --tag final_m6 --eval-test --test-role m6_final --save-preds --save-ckpt

  # Phase 6 — refit produksi pada train+val (epoch tetap), test dibuka sekali per role
  python run_m6_config.py --config final_cfg.json --seeds 100 101 102 103 104 --fit trainval \
         --refit-epochs 24 --tag refit_m6 --eval-test --test-role m6_refit --save-preds --save-ckpt

Ledger penjaga test (early_fusion/results/test_eval_ledger.jsonl):
  Satu --test-role hanya boleh membuka test untuk SATU konfigurasi (hash cfg+variant+fit).
  Seed tambahan untuk konfigurasi yang sama diperbolehkan; konfigurasi berbeda ditolak kecuali --force-test.
  Tujuannya mencegah "mengintip test lalu mengutak-atik" tanpa sengaja.

Ensemble seed: prediksi seluruh seed dirata-ratakan -> metrik val (dan test). Ini calon model produksi
(deep ensemble) dan hampir selalu lebih baik/stabil daripada rata-rata model tunggal.
"""
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

RESULTS_DIR = Path("early_fusion/results")
RUNS_LOG = RESULTS_DIR / "m6_final_runs.jsonl"
LEDGER = RESULTS_DIR / "test_eval_ledger.jsonl"
PRED_DIR = RESULTS_DIR / "preds"
CKPT_DIR = Path("early_fusion/models/checkpoints")


def load_cfg(spec):
    """'path.json' atau 'path.json:K' (K = indeks di file top3.json). None -> {} (pakai DEFAULT_CFG)."""
    if spec is None:
        return {}
    idx = None
    if ":" in spec and not spec.endswith(":"):
        spec, idx_s = spec.rsplit(":", 1)
        idx = int(idx_s)
    obj = json.load(open(spec))
    if isinstance(obj, list):
        obj = obj[idx if idx is not None else 0]["cfg"]
    return obj


def cfg_hash(cfg, variant, fit):
    payload = json.dumps({"cfg": cfg, "variant": variant, "fit": fit}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def check_ledger(role, chash, force):
    """Tolak bila role ini sudah membuka test untuk konfigurasi LAIN."""
    if not LEDGER.exists():
        return
    prior = [json.loads(l) for l in open(LEDGER) if l.strip()]
    other = [p for p in prior if p["role"] == role and p["cfg_hash"] != chash]
    if other and not force:
        raise SystemExit(
            f"[LEDGER] role '{role}' sudah membuka test untuk konfigurasi lain "
            f"(hash {other[0]['cfg_hash']}, tag {other[0]['tag']}). Menolak membuka test lagi untuk "
            f"hash {chash}. Pakai role berbeda bila ini memang pembanding lain, atau --force-test "
            f"(dan catat alasannya di laporan).")


def append_ledger(role, chash, tag, seeds):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, cfg_hash=chash,
                                tag=tag, seeds=seeds)) + "\n")


def ensemble_metrics(pred_list, targets, metrics_fn):
    return metrics_fn(np.mean(np.stack(pred_list, axis=0), axis=0), targets)


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="json cfg, atau file_top3.json:K; default = DEFAULT_CFG")
    ap.add_argument("--variant", choices=["full", "no_image", "no_text", "tab_only"], default="full")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V",
                    help="override cfg cepat untuk ablasi, mis. --set gate_lr_mult=0 rank_lambda=0")
    ap.add_argument("--fit", choices=["train", "trainval"], default="train")
    ap.add_argument("--refit-epochs", type=int, default=None)
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--test-role", default=None)
    ap.add_argument("--force-test", action="store_true")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--save-ckpt", action="store_true")
    return ap.parse_args(argv)


def apply_overrides(cfg, kvs):
    for kv in kvs:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def run(args, default_cfg, data, train_one, metrics_fn):
    cfg = {**default_cfg, **load_cfg(args.config)}
    cfg = apply_overrides(cfg, args.set)
    chash = cfg_hash(cfg, args.variant, args.fit)

    if args.eval_test:
        if not args.test_role:
            raise SystemExit("--eval-test wajib disertai --test-role (mis. m6_final, tab_baseline, m6_refit)")
        check_ledger(args.test_role, chash, args.force_test)
    if args.fit == "trainval" and cfg["schedule"] == "const" and args.refit_epochs is None:
        raise SystemExit("--fit trainval dengan schedule const butuh --refit-epochs "
                         "(mis. median best_epoch dari run train-only)")

    print(f"tag={args.tag} variant={args.variant} fit={args.fit} cfg_hash={chash} "
          f"eval_test={args.eval_test} seeds={args.seeds}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in args.seeds:
        res = train_one(cfg, seed, data, variant=args.variant, fit=args.fit,
                        fixed_epochs=args.refit_epochs, eval_test=args.eval_test,
                        return_state=args.save_ckpt, verbose=False)
        runs.append(res)
        line = f"seed {seed}: "
        if args.fit == "train":
            line += f"val_sp={res['val_spearman']:.4f} val_auc={res['val_auc']:.4f} best_ep={res['best_epoch']} "
            line += f"| {res.get('seconds', 0):.0f}s "
        if args.eval_test:
            line += f"test_sp={res['test_spearman']:.4f} test_auc={res['test_auc']:.4f} test_mae={res['test_mae']:.4f}"
        print(line)

        rec = dict(tag=args.tag, cfg_hash=chash, variant=args.variant, fit=args.fit, seed=seed,
                   protocol=res["protocol"], best_epoch=res["best_epoch"], epochs_run=res["epochs_run"],
                   n_params=res["n_params"], cfg=cfg, refit_epochs=args.refit_epochs,
                   split_hashes=data["split_hashes"], git_sha=data["git_sha"], git_dirty=data["git_dirty"],
                   test_eval=bool(args.eval_test), test_role=args.test_role)
        for k in ("val_spearman", "val_auc", "val_mae", "val_loss", "test_spearman", "test_auc", "test_mae"):
            if k in res:
                rec[k] = float(res[k])
        if "gate_stats" in res:
            rec["gate_stats"] = res["gate_stats"]
        with open(RUNS_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if args.save_preds:
            arrs = {}
            for k in ("val_preds", "val_targets", "test_preds", "test_targets"):
                if k in res:
                    arrs[k] = res[k]
            np.savez(PRED_DIR / f"{args.tag}_seed{seed}.npz", **arrs)
        if args.save_ckpt:
            import torch
            torch.save(dict(state_dict=res["state_dict"], cfg=cfg, variant=args.variant, fit=args.fit,
                            seed=seed, scaler=data["scaler"], genres_train=data["genres_train"],
                            n_cont=data["n_cont"], n_genres=data["n_genres"],
                            split_hashes=data["split_hashes"], git_sha=data["git_sha"]),
                       CKPT_DIR / f"m6_granular_{args.tag}_seed{seed}.pt")

    if args.eval_test:
        append_ledger(args.test_role, chash, args.tag, args.seeds)

    summary = dict(tag=args.tag, cfg_hash=chash, variant=args.variant, fit=args.fit, seeds=args.seeds, cfg=cfg)
    print("\n=== ringkasan ===")
    if args.fit == "train":
        m, s = summarize([r["val_spearman"] for r in runs])
        summary["val_spearman_mean"], summary["val_spearman_std"] = m, s
        print(f"val Spearman  single-model: {m:.4f} ± {s:.4f}   per-seed: "
              + " ".join(f"{r['val_spearman']:.4f}" for r in runs))
        if len(runs) > 1:
            em = ensemble_metrics([r["val_preds"] for r in runs], runs[0]["val_targets"], metrics_fn)
            summary["val_spearman_ensemble"] = em["spearman"]
            print(f"val Spearman  ensemble({len(runs)}): {em['spearman']:.4f}   AUC {em['auc']:.4f}")
        bes = [r["best_epoch"] for r in runs]
        summary["best_epochs"] = bes
        print(f"best/final epoch: {bes} (median {int(np.median(bes))})")
    if args.eval_test:
        m, s = summarize([r["test_spearman"] for r in runs])
        summary["test_spearman_mean"], summary["test_spearman_std"] = m, s
        print(f"test Spearman single-model: {m:.4f} ± {s:.4f}   per-seed: "
              + " ".join(f"{r['test_spearman']:.4f}" for r in runs))
        if len(runs) > 1:
            em = ensemble_metrics([r["test_preds"] for r in runs], runs[0]["test_targets"], metrics_fn)
            summary["test_spearman_ensemble"], summary["test_auc_ensemble"], summary["test_mae_ensemble"] = \
                em["spearman"], em["auc"], em["mae"]
            print(f"test Spearman ensemble({len(runs)}): {em['spearman']:.4f}   "
                  f"AUC {em['auc']:.4f}   MAE {em['mae']:.4f}")
    with open(RESULTS_DIR / f"summary_{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {RESULTS_DIR / f'summary_{args.tag}.json'}  (log: {RUNS_LOG})")
    return summary


def main():
    args = parse_args()
    from early_fusion.experiments.m6_core import DEFAULT_CFG, load_data, train_one, metrics   # lazy: butuh torch
    data = load_data()
    run(args, DEFAULT_CFG, data, train_one, metrics)


if __name__ == "__main__":
    main()
