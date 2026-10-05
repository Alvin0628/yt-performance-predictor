"""Optuna bertahap untuk M6 granular v2 (sigmoid2, temporal_no_subs).

Lokasi yang disarankan: early_fusion/experiments/tune_m6_optuna.py

Kenapa bertahap (A -> B -> C) dan bukan satu studi besar?
  Dengan budget puluhan trial, TPE tidak bisa menjelajah >12 dimensi secara andal. Dimensi
  dikelompokkan menurut sebab-akibat (optimisasi/regularisasi dulu karena hambatan utama = overfit
  cepat; arsitektur kemudian; refine lokal terakhir) dan tiap tahap memakai hasil tahap sebelumnya
  sebagai basis (di-enqueue sebagai trial #0 supaya selalu ada pembanding pada kondisi identik).

  Stage A : lr, weight_decay, batch_size, schedule(+epochs), dropout, emb_noise, mod_dropout,
            ema_decay, rank_lambda
  Stage B : d, num_layers, cross_attn_layers, ff_mult
  Stage C : refine di sekitar basis (lr, emb_noise, mod_dropout menyempit; weight_decay & dropout lebih lebar
            karena model Stage B lebih besar) + ema_decay, rank_lambda, gate_lr_mult, gate_dropout

Objective mode (--objective):
  risk     : mean - risk_lambda*std atas seed tuning (dipakai Stage A)
  ensemble : Spearman val dari RATA-RATA PREDIKSI seluruh seed tuning (selaras dengan model produksi = ensemble
             seed; lebih kecil terpengaruh noise inisialisasi). Disarankan untuk Stage B/C dengan 3 seed:
             python tune_m6_optuna.py --stage B --objective ensemble --seeds 42 43 44

Objective (anti-overfit-ke-val):
  skor trial = mean(val Spearman atas seed tuning) - risk_lambda * std   (default 2 seed: 42, 43; lambda 0.5)
  -> konfigurasi yang kadang kolaps pada satu seed (mis. 0.237 di seed 44 sebelum modality dropout)
     dihukum. Rung hemat: bila seed pertama di bawah persentil ke-prune_pct dari trial sebelumnya,
     seed kedua dilewati (TrialPruned) -> sekitar 30-40% komputasi hemat.

TEST SET TIDAK PERNAH DISENTUH di file ini (tidak ada kode jalur test sama sekali).

Contoh:
  python tune_m6_optuna.py --stage A --n-trials 60
  python tune_m6_optuna.py --stage B --n-trials 24      # otomatis memakai stageA_full_best.json
  python tune_m6_optuna.py --stage C --n-trials 30      # otomatis memakai stageB_full_best.json
  # baseline tabular-only dengan budget IDENTIK (adil):
  python tune_m6_optuna.py --stage A --variant tab_only --n-trials 60   (lalu B, C)
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna

OUT_DIR = Path("early_fusion/results/optuna")
AUTO_BASE_STAGE = {"B": "A", "C": "B"}
CONST_MAX_EPOCHS = 60          # tuning: cap epoch (early stop biasanya berhenti ~20-30; config bagus best_epoch 7-12)
STARTUP = {"A": 12, "B": 8, "C": 10}
EARLY_ABORT = dict(min_epoch=8, sp_floor=0.24)   # run yang best val Spearman-nya < 0.24 di epoch >= 8 dihentikan


# ----------------------------------------------------------------------------------
# Helper rentang
# ----------------------------------------------------------------------------------
def _grid_with(base, grid):
    return sorted(set(float(x) for x in grid) | {float(base)})


def _rng(base, lo_mult, hi_mult, abs_lo, abs_hi):
    lo, hi = max(abs_lo, base * lo_mult), min(abs_hi, base * hi_mult)
    return min(lo, base), max(hi, base)


def _span(base, delta, abs_lo, abs_hi):
    lo, hi = max(abs_lo, base - delta), min(abs_hi, base + delta)
    return min(lo, base), max(hi, base)


# ----------------------------------------------------------------------------------
# Ruang pencarian per tahap. Return: dict override cfg.
# ----------------------------------------------------------------------------------
def suggest_A(trial, base, has_mod):
    c = {}
    c["lr"] = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
    c["weight_decay"] = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    c["batch_size"] = trial.suggest_categorical("batch_size", [64, 128])      # 32 dibuang: 2x lebih lambat
    c["schedule"] = trial.suggest_categorical("schedule", ["const", "cosine"])
    if c["schedule"] == "cosine":
        c["epochs"] = trial.suggest_categorical("epochs_cos", [15, 20, 30, 40])
    else:
        c["epochs"] = CONST_MAX_EPOCHS            # early stop yang menentukan
    c["dropout"] = trial.suggest_float("dropout", 0.1, 0.35)
    c["emb_noise"] = trial.suggest_float("emb_noise", 0.0, 0.1)
    if has_mod:
        c["mod_dropout"] = trial.suggest_float("mod_dropout", 0.0, 0.3)
    c["ema_decay"] = trial.suggest_categorical("ema_decay", [0.0, 0.99, 0.995])
    c["rank_lambda"] = trial.suggest_categorical("rank_lambda", [0.0, 0.1, 0.3])
    return c


def suggest_B(trial, base, has_mod):
    c = {}
    c["d"] = trial.suggest_categorical("d", [64, 96, 128, 192])
    c["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 3])
    if has_mod:
        c["cross_attn_layers"] = trial.suggest_categorical("cross_attn_layers", [1, 2])
    c["ff_mult"] = trial.suggest_categorical("ff_mult", [2, 4])
    return c


def suggest_C(trial, base, has_mod):
    c = {}
    lo, hi = _rng(base["lr"], 0.5, 2.0, 1e-5, 1e-3)
    c["lr"] = trial.suggest_float("lr", lo, hi, log=True)
    # rentang regularisasi sengaja lebih lebar dari "refine lokal" murni: model hasil Stage B lebih besar
    # (kapasitas naik) dengan dropout rendah, jadi pencarian harus bebas bergerak ke regularisasi lebih kuat.
    lo, hi = _rng(base["weight_decay"], 0.3, 4.0, 1e-4, 0.3)
    c["weight_decay"] = trial.suggest_float("weight_decay", lo, hi, log=True)
    lo, hi = _span(base["dropout"], 0.15, 0.05, 0.4)
    c["dropout"] = trial.suggest_float("dropout", lo, hi)
    c["ema_decay"] = trial.suggest_categorical("ema_decay", _grid_with(base["ema_decay"], [0.0, 0.99, 0.995]))
    lo, hi = _span(base["emb_noise"], 0.05, 0.0, 0.3)
    c["emb_noise"] = trial.suggest_float("emb_noise", lo, hi)
    c["rank_lambda"] = trial.suggest_categorical(
        "rank_lambda", _grid_with(base["rank_lambda"], [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]))
    if has_mod:
        lo, hi = _span(base["mod_dropout"], 0.1, 0.0, 0.5)
        c["mod_dropout"] = trial.suggest_float("mod_dropout", lo, hi)
        c["gate_lr_mult"] = trial.suggest_categorical(
            "gate_lr_mult", _grid_with(base["gate_lr_mult"], [0.0, 3.0, 10.0, 30.0]))
        c["gate_dropout"] = trial.suggest_categorical(
            "gate_dropout", _grid_with(base["gate_dropout"], [0.0, 0.1, 0.3]))
    if base["schedule"] == "cosine":
        e = int(base["epochs"])
        c["epochs"] = trial.suggest_categorical(
            "epochs", sorted({max(10, e + k) for k in (-10, -5, 0, 5, 10)}))
    return c


SUGGEST = {"A": suggest_A, "B": suggest_B, "C": suggest_C}


def default_enqueue(stage, base, has_mod):
    """Parameter trial #0 = basis (supaya selalu ada pembanding di kondisi identik)."""
    p = {}
    if stage == "A":
        p = dict(lr=base["lr"], weight_decay=base["weight_decay"], batch_size=base["batch_size"],
                 schedule=base["schedule"], dropout=base["dropout"], emb_noise=base["emb_noise"],
                 ema_decay=float(base["ema_decay"]), rank_lambda=float(base["rank_lambda"]))
        if base["schedule"] == "cosine":
            p["epochs_cos"] = base["epochs"]
        if has_mod:
            p["mod_dropout"] = base["mod_dropout"]
    elif stage == "B":
        p = dict(d=base["d"], num_layers=base["num_layers"], ff_mult=base["ff_mult"])
        if has_mod:
            p["cross_attn_layers"] = base["cross_attn_layers"]
    else:
        p = dict(lr=base["lr"], weight_decay=base["weight_decay"], dropout=base["dropout"],
                 emb_noise=base["emb_noise"], rank_lambda=float(base["rank_lambda"]),
                 ema_decay=float(base["ema_decay"]))
        if has_mod:
            p.update(mod_dropout=base["mod_dropout"], gate_lr_mult=float(base["gate_lr_mult"]),
                     gate_dropout=float(base["gate_dropout"]))
        if base["schedule"] == "cosine":
            p["epochs"] = int(base["epochs"])
    return p


# ----------------------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------------------
def make_objective(stage, variant, base_cfg, data, seeds, risk_lambda, prune_pct, min_prune,
                   train_one, log_path, study_name, abs_floor=0.26, objective_mode="risk"):
    has_mod = variant != "tab_only"

    def objective(trial):
        overrides = SUGGEST[stage](trial, base_cfg, has_mod)
        cfg = {**base_cfg, **overrides}
        trial.set_user_attr("cfg", cfg)

        scores = []
        preds_list, targets_ref = [], None
        for k, seed in enumerate(seeds):
            try:
                res = train_one(cfg, seed, data, variant=variant, fit="train", early_abort=EARLY_ABORT)
            except RuntimeError as e:                      # OOM pada d/batch besar -> lewati trial
                if "out of memory" in str(e).lower():
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    raise optuna.TrialPruned()
                raise
            sp = res["val_spearman"]
            if not np.isfinite(sp):
                sp = -1.0
            scores.append(float(sp))
            if res.get("val_preds") is not None:
                preds_list.append(np.asarray(res["val_preds"]))
                targets_ref = np.asarray(res["val_targets"])
            trial.set_user_attr(f"sp_seed{seed}", float(sp))
            trial.set_user_attr(f"best_epoch_seed{seed}", int(res["best_epoch"]))
            with open(log_path, "a") as f:
                f.write(json.dumps(dict(
                    study=study_name, trial=trial.number, seed=seed, val_spearman=float(sp),
                    val_auc=float(res["val_auc"]), val_loss=float(res["val_loss"]),
                    best_epoch=int(res["best_epoch"]), epochs_run=int(res["epochs_run"]),
                    protocol=res["protocol"], n_params=int(res["n_params"]),
                    aborted=bool(res.get("aborted", False)), seconds=float(res.get("seconds", 0.0)),
                    cfg=cfg)) + "\n")

            if k == 0:
                prior = [t.user_attrs["s1"] for t in trial.study.get_trials(deepcopy=False)
                         if "s1" in t.user_attrs and t.number != trial.number]
                trial.set_user_attr("s1", float(sp))
                trial.report(float(sp), step=1)            # supaya TPE tetap belajar dari trial yang di-prune
                if len(seeds) > 1 and (sp < abs_floor or
                                       (len(prior) >= min_prune and sp < np.percentile(prior, prune_pct))):
                    raise optuna.TrialPruned()

        m = float(np.mean(scores))
        s = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        trial.set_user_attr("mean_sp", m)
        trial.set_user_attr("std_sp", s)
        ens = None
        if len(preds_list) == len(seeds) and targets_ref is not None:
            from scipy.stats import spearmanr
            ens = float(spearmanr(np.mean(np.stack(preds_list, axis=0), axis=0), targets_ref)[0])
            if not np.isfinite(ens):
                ens = -1.0
            trial.set_user_attr("ens_sp", ens)
        if objective_mode == "ensemble":
            if ens is None:
                raise RuntimeError("objective=ensemble butuh val_preds dari train_one")
            return ens
        return m - risk_lambda * s

    return objective


# ----------------------------------------------------------------------------------
# Laporan
# ----------------------------------------------------------------------------------
def report_and_save(study, stage, variant, seeds, k_top=5):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: t.value, reverse=True)
    # TPE pada ruang kategorikal kecil (Stage B) bisa mengusulkan konfigurasi yang SAMA berulang kali
    # (hasilnya identik karena seed dan GPU deterministik) -> dedupe supaya top-3 benar-benar 3 kandidat berbeda.
    seen, uniq = set(), []
    for t in done:
        key = json.dumps(t.user_attrs["cfg"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    n_complete_all = len(done)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"\n=== stage {stage} ({variant}): {n_complete_all} complete "
          f"({len(uniq)} konfigurasi unik), {n_pruned} pruned ===")
    default = next((t for t in done if t.user_attrs.get("is_default")), None)
    if default is not None:
        print(f"basis (trial #{default.number}): objective={default.value:.4f} "
              f"mean={default.user_attrs['mean_sp']:.4f} std={default.user_attrs['std_sp']:.4f} "
              f"ens={default.user_attrs.get('ens_sp', float('nan')):.4f}")
    print(f"{'rank':>4} {'trial':>5} {'objective':>9} {'mean':>7} {'std':>6} {'ens':>7}  per-seed")
    for r, t in enumerate(uniq[:k_top], 1):
        per = " ".join(f"{t.user_attrs.get(f'sp_seed{s}', float('nan')):.4f}" for s in seeds)
        print(f"{r:>4} {t.number:>5} {t.value:>9.4f} {t.user_attrs['mean_sp']:>7.4f} "
              f"{t.user_attrs['std_sp']:>6.4f} {t.user_attrs.get('ens_sp', float('nan')):>7.4f}  {per}")
    if not done:
        print("belum ada trial selesai")
        return None

    top = uniq[:3]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"stage{stage}_{variant}_top3.json", "w") as f:
        json.dump([dict(trial=t.number, objective=t.value, mean=t.user_attrs["mean_sp"],
                        std=t.user_attrs["std_sp"], cfg=t.user_attrs["cfg"]) for t in top], f, indent=2)
    with open(OUT_DIR / f"stage{stage}_{variant}_best.json", "w") as f:
        json.dump(top[0].user_attrs["cfg"], f, indent=2)
    if default is not None:
        gain = top[0].value - default.value
        print(f"\nselisih terbaik - basis (objective): {gain:+.4f}")
        if gain < 0.005:
            print("  -> < 0.005: peningkatan belum bisa dibedakan dari noise. "
                  "Pertimbangkan berhenti di tahap ini / mempertahankan basis.")
    print(f"saved: {OUT_DIR / f'stage{stage}_{variant}_best.json'} (+ _top3.json)")
    return top


# ----------------------------------------------------------------------------------
# Pemulihan setelah crash / laptop mati
# ----------------------------------------------------------------------------------
def recover_stale_trials(study, stage, base_cfg, has_mod):
    """Proses ini satu-satunya yang menulis ke study, jadi trial berstatus RUNNING saat start = sisa crash.
    Tandai FAIL (tidak memakan budget), dan bila yang mati adalah trial basis (#0 default), enqueue ulang
    supaya pembanding basis tetap ada."""
    TS = optuna.trial.TrialState
    stale = [t for t in study.trials if t.state == TS.RUNNING]
    for t in stale:
        try:
            study._storage.set_trial_state_values(t._trial_id, TS.FAIL)
        except Exception as e:                                  # API internal; jangan sampai menghentikan run
            print(f"[recover] gagal menandai trial {t.number}: {e}")
    if stale:
        print(f"[recover] {len(stale)} trial sisa crash ditandai FAIL: {[t.number for t in stale]}")
    have_default = any(t.user_attrs.get("is_default") and t.state in (TS.COMPLETE, TS.WAITING)
                       for t in study.trials)
    if not have_default:
        study.enqueue_trial(default_enqueue(stage, base_cfg, has_mod), user_attrs={"is_default": True})
        print("[recover] trial basis belum selesai -> di-enqueue ulang")


# ----------------------------------------------------------------------------------
# Runner (dipisah dari main() supaya bisa diuji tanpa torch)
# ----------------------------------------------------------------------------------
def run_stage(args, default_cfg, data, train_one):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variant, stage = args.variant, args.stage

    if args.base_config:
        base_cfg = {**default_cfg, **json.load(open(args.base_config))}
    elif stage == "A":
        base_cfg = dict(default_cfg)
    else:
        auto = OUT_DIR / f"stage{AUTO_BASE_STAGE[stage]}_{variant}_best.json"
        if not auto.exists():
            raise SystemExit(f"basis tidak ditemukan: {auto}. Jalankan stage {AUTO_BASE_STAGE[stage]} dulu "
                             f"atau beri --base-config.")
        base_cfg = {**default_cfg, **json.load(open(auto))}
        print(f"basis: {auto}")

    study_name = args.study_name or f"m6_granular_{variant}_stage{stage}"
    storage = args.storage or f"sqlite:///{OUT_DIR.as_posix()}/m6_granular.db"
    try:
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True, group=True,
                                             n_startup_trials=STARTUP[stage])
    except TypeError:
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True,
                                             n_startup_trials=STARTUP[stage])
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.NopPruner())

    has_mod = variant != "tab_only"
    recover_stale_trials(study, stage, base_cfg, has_mod)       # no-op pada study baru (enqueue basis di sini)

    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                       optuna.trial.TrialState.PRUNED)]
    remaining = max(0, args.n_trials - len(finished))
    print(f"study={study_name} stage={stage} variant={variant} seeds={args.seeds} "
          f"objective={args.objective} risk_lambda={args.risk_lambda} | selesai={len(finished)} sisa={remaining}")

    obj = make_objective(stage, variant, base_cfg, data, args.seeds, args.risk_lambda,
                         args.prune_pct, args.min_prune, train_one,
                         OUT_DIR / "m6_tuning_runs.jsonl", study_name, abs_floor=args.abs_floor,
                         objective_mode=args.objective)
    if remaining > 0:
        study.optimize(obj, n_trials=remaining, timeout=args.timeout)
    return report_and_save(study, stage, variant, args.seeds)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"], required=True)
    ap.add_argument("--variant", choices=["full", "tab_only"], default="full")
    ap.add_argument("--n-trials", type=int, default=40, help="target total trial (resume-able)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--risk-lambda", type=float, default=0.5)
    ap.add_argument("--objective", choices=["risk", "ensemble"], default="risk")
    ap.add_argument("--prune-pct", type=float, default=35.0)
    ap.add_argument("--min-prune", type=int, default=8)
    ap.add_argument("--abs-floor", type=float, default=0.26,
                    help="seed pertama < floor -> seed kedua dilewati (tanpa menunggu min-prune)")
    ap.add_argument("--base-config", default=None)
    ap.add_argument("--storage", default=None)
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=None, help="detik; opsional")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    from early_fusion.experiments.m6_core import DEFAULT_CFG, load_data, train_one   # lazy: butuh torch
    data = load_data()
    run_stage(args, DEFAULT_CFG, data, train_one)


if __name__ == "__main__":
    main()
