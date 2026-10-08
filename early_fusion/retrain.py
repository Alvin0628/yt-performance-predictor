"""Retrain otomatis ensemble M6 (Fase 4): enam tahap yang bisa dijalankan terpisah.

    export   Ekspor snapshot dari Postgres ke data_snapshots/retrain/<run_id>/snapshot.parquet
             (urut kronologis, kolom `split` 80/10/10, hash dihitung dari file hasil baca ulang).
    tokens   Perbarui TokenStore (cache token per video_id): hanya video baru yang diproses.
    split    Tulis split.json (format yang sama dengan early_fusion/splits/temporal_no_subs.json).
    train    Refit N seed di train+val dengan resep produksi; checkpoint + prediksi test per seed.
    package  Kemas N checkpoint jadi SATU bundle .pt (+ .json): tulis train_end, hash snapshot,
             jumlah baris, hash split ke dalam bundle.
    verify   Muat bundle, cek hash/metadata/prediksi/jalur tabular; cetak kontrak JSON.
    all      export -> tokens -> split -> train -> package -> verify dalam satu perintah.

Contoh (PC, dari akar repo, env POSTGRES_* dan MINIO_* sudah di-set):
    python -m early_fusion.retrain all
    python -m early_fusion.retrain export
    python -m early_fusion.retrain tokens  --run-id 20261009T020000Z
    python -m early_fusion.retrain train   --run-id 20261009T020000Z --seeds 100 101
    python -m early_fusion.retrain verify  --run-id 20261009T020000Z

Pembangkitan ulang dari snapshot lama (tanpa Postgres), mis. untuk uji reproduksi:
    python -m early_fusion.retrain export --from-parquet data_snapshots/snapshot.parquet

KONTRAK STDOUT: baris TERAKHIR stdout selalu satu baris JSON.
  * semua tahap membawa kunci promoted/versioned_path/reason (promoted selalu false di Fase 4);
    tahap selain verify tidak punya new_spearman, dan versioned_path-nya null
  * verify dan all: kontrak lengkap yang dibaca dags/train_model.py::check_promotion
        {"promoted": false, "versioned_path": ..., "reason": ..., "new_spearman": ..., ...}
    Di Fase 4 `promoted` SELALU false: tidak ada kode di sini yang menyentuh bundle live.
    Gerbang promosi (bootstrap berpasangan vs model live) baru datang di Fase 5.
  Tidak ada yang boleh dicetak setelah baris JSON itu.

Resep produksi = resep bundle live (early_fusion/models/final/m6_granular_ensemble_v1.json):
final_cfg.json + gate_lr_mult=0 (gate dimatikan), 8 epoch tetap, seed 100..104. Tes
tests/early_fusion/test_retrain.py menjaga agar resep ini tidak menyimpang dari sidecar bundle live.

Modul ini sengaja tidak meng-import torch di tingkat atas: export/tokens(split) tidak butuh torch
sampai ekstraktor CLIP benar-benar dipanggil.
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from early_fusion.data_spec import DataSpec, assign_temporal_split, hash_df, temporal_split_sizes

DEFAULT_ROOT = Path("data_snapshots/retrain")
DEFAULT_STORE = Path("data_snapshots/token_store")
FINAL_CFG = Path("early_fusion/results/final_cfg.json")

# Resep produksi (lihat docstring). Diubah hanya lewat commit, bukan lewat flag diam-diam.
PRODUCTION_OVERRIDES = {"gate_lr_mult": 0}
PRODUCTION_SEEDS = [100, 101, 102, 103, 104]
# 8 = median best_epoch run train-only ([8,14,6,6,8] di summary_final_m6_c24_gateoff.json); itulah epoch
# refit bundle live (m6_final_runs.jsonl, tag refit_m6_c24_gateoff). "c24" di nama tag BUKAN jumlah epoch.
PRODUCTION_REFIT_EPOCHS = 8

EMBEDDING_COLS = ("image_embedding", "text_embedding")
REQUIRED_COLS = ("video_id", "published_at", "title", "channel_ref", "views",
                 "trailing_avg_views", "genre")
PRED_TOL = 1e-3          # sama dengan early_fusion/experiments/verify_final_model.py

# Sama dengan query di scripts/export_snapshot.py (dijaga tes agar tidak menyimpang).
SNAPSHOT_QUERY = (
    "SELECT * FROM videos "
    "WHERE label_finalized = true "
    "AND image_embedding IS NOT NULL "
    "AND trailing_avg_views IS NOT NULL "
    "ORDER BY published_at, video_id"
)

STAGES = ("export", "tokens", "split", "train", "package", "verify")


# ----------------------------------------------------------------------------------
# Utilitas kecil
# ----------------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def emit(payload):
    """Cetak kontrak JSON. HARUS pemanggilan print terakhir sebelum proses selesai."""
    print(json.dumps(payload, default=str), flush=True)


def parse_overrides(kvs):
    out = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise SystemExit(f"--set butuh K=V, dapat: {kv!r}")
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_production_cfg(path=FINAL_CFG, overrides=None):
    cfg = json.loads(Path(path).read_text())
    cfg.update(PRODUCTION_OVERRIDES)
    cfg.update(overrides or {})
    return cfg


# ----------------------------------------------------------------------------------
# Direktori run + status
# ----------------------------------------------------------------------------------
class Run:
    """data_snapshots/retrain/<run_id>/ : snapshot, split, ckpt, preds, bundle, run.json."""

    def __init__(self, root, run_id):
        self.root = Path(root)
        self.run_id = run_id
        self.dir = self.root / run_id
        self.snapshot_path = self.dir / "snapshot.parquet"
        self.split_path = self.dir / "split.json"
        self.ckpt_dir = self.dir / "ckpt"
        self.preds_dir = self.dir / "preds"
        self.bundle_dir = self.dir / "bundle"
        self.state_path = self.dir / "run.json"

    @classmethod
    def new(cls, root, run_id=None):
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = cls(root, run_id)
        run.dir.mkdir(parents=True, exist_ok=False)     # tabrakan id = error, bukan menimpa
        run.save_state({"run_id": run_id, "created_at": _now(), "stages": {}})
        return run

    @classmethod
    def existing(cls, root, run_id):
        run = cls(root, run_id)
        if not run.state_path.exists():
            raise SystemExit(f"run {run_id!r} tidak ditemukan di {run.root} "
                             f"(jalankan `export` dulu atau periksa --run-id)")
        return run

    def load_state(self):
        return json.loads(self.state_path.read_text())

    def save_state(self, state):
        _atomic_write_text(self.state_path, json.dumps(state, indent=2, default=str))

    def mark(self, stage, info):
        st = self.load_state()
        st["stages"][stage] = {**info, "finished_at": _now()}
        self.save_state(st)

    def stage_info(self, stage):
        return self.load_state()["stages"][stage]

    def require(self, *stages):
        done = self.load_state()["stages"]
        for s in stages:
            if s not in done:
                raise SystemExit(f"tahap '{s}' belum selesai untuk run {self.run_id}. "
                                 f"Jalankan: python -m early_fusion.retrain {s} --run-id {self.run_id}")

    def spec(self, store_dir):
        """DataSpec untuk run ini: snapshot + split milik run, cache = TokenStore."""
        return DataSpec(snapshot_path=self.snapshot_path, snapshot_hash=None,
                        split_file=self.split_path, cache_dir=Path(store_dir), cache_kind="store")


# ----------------------------------------------------------------------------------
# Tahap 1: export
# ----------------------------------------------------------------------------------
def build_snapshot_frame(df, max_rows=None):
    """Mentah dari DB -> snapshot: tanpa kolom embedding, urut (published_at, video_id), kolom split.

    Urutan di-sortir ulang di pandas (stabil) supaya tidak bergantung pada collation SQL.
    max_rows=N mengambil N video TERBARU (tetap kronologis); dipakai uji coba kering.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"kolom wajib tidak ada di hasil query: {missing}")
    df = df.drop(columns=[c for c in EMBEDDING_COLS if c in df.columns])
    if df["video_id"].duplicated().any():
        raise ValueError("video_id duplikat di snapshot")
    if df["published_at"].isna().any():
        raise ValueError("published_at null di snapshot")
    df = df.sort_values(["published_at", "video_id"], kind="mergesort").reset_index(drop=True)
    if max_rows is not None:
        if max_rows < 10:
            raise ValueError("--max-rows minimal 10")
        df = df.tail(int(max_rows)).reset_index(drop=True)
    df["split"] = assign_temporal_split(len(df))
    return df


def write_snapshot(df, path):
    """Tulis parquet secara atomik; kembalikan hash dari file yang DIBACA ULANG.

    Hash itu yang akan dihitung load_snapshot() nanti (ia membaca parquet, bukan df di memori),
    jadi hash di manifest dan di split.json pasti cocok dengan yang dilihat training.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return hash_df(pd.read_parquet(path))


def compute_train_end(df):
    """Waktu publikasi terbaru di train+val = batas akhir data yang dilihat kandidat."""
    return str(df.loc[df["split"].isin(["train", "val"]), "published_at"].max())


def snapshot_manifest(df, snapshot_hash, source, max_rows=None):
    n = len(df)
    n_tr, n_va, n_te = temporal_split_sizes(n)
    return dict(
        snapshot_hash=snapshot_hash, n_total=n, n_train=n_tr, n_val=n_va, n_test=n_te,
        n_channels=int(df["channel_ref"].nunique()),
        published_min=str(df["published_at"].min()), published_max=str(df["published_at"].max()),
        train_end=compute_train_end(df),
        test_start=str(df.loc[df["split"] == "test", "published_at"].min()),
        source=source, max_rows=max_rows,
    )


def _read_sql_from_db():
    from models.precompute_embeddings import engine       # butuh POSTGRES_* di env
    return pd.read_sql(SNAPSHOT_QUERY, engine)


def stage_export(run, *, read_sql_fn=None, from_parquet=None, max_rows=None, log=print):
    if from_parquet and max_rows:
        raise SystemExit("--from-parquet tidak bisa dikombinasikan dengan --max-rows "
                         "(file sumber disalin apa adanya supaya hash tetap)")
    if from_parquet:
        src = Path(from_parquet)
        shutil.copyfile(src, run.snapshot_path)            # byte-identik -> hash identik
        df = pd.read_parquet(run.snapshot_path)
        snapshot_hash = hash_df(df)
        if "split" not in df.columns:
            raise SystemExit(f"{src}: tidak ada kolom 'split'")
        if not df["published_at"].is_monotonic_increasing:
            raise SystemExit(f"{src}: published_at tidak berurutan naik; bukan snapshot kronologis")
        if list(df["split"]) != assign_temporal_split(len(df)):
            raise SystemExit(f"{src}: kolom 'split' tidak sama dengan pembagian 80/10/10 kronologis")
        source = {"kind": "parquet", "path": str(src)}
    else:
        raw = (read_sql_fn or _read_sql_from_db)()
        log(f"baris hasil query: {len(raw)}")
        df = build_snapshot_frame(raw, max_rows=max_rows)
        snapshot_hash = write_snapshot(df, run.snapshot_path)
        source = {"kind": "postgres", "query": SNAPSHOT_QUERY}

    manifest = snapshot_manifest(df, snapshot_hash, source, max_rows)
    log(f"snapshot {snapshot_hash}: {manifest['n_total']} baris "
        f"(train/val/test = {manifest['n_train']}/{manifest['n_val']}/{manifest['n_test']}), "
        f"{manifest['n_channels']} kanal, train_end={manifest['train_end']}")
    run.mark("export", manifest)
    return manifest


# ----------------------------------------------------------------------------------
# Tahap 2: tokens
# ----------------------------------------------------------------------------------
def stage_tokens(run, *, store_dir, create_empty=False, batch=16, flush_every=128,
                 extractor=None, log=print):
    from early_fusion.datasets.token_store import TokenStore, META_FILE
    from early_fusion.datasets.token_update import update_store

    run.require("export")
    store_dir = Path(store_dir)
    if not (store_dir / META_FILE).exists():
        if not create_empty:
            raise SystemExit(f"tidak ada token store di {store_dir}. Di PC: salin/bootstrap dari cache lama "
                             f"(python -m scripts.update_token_store --bootstrap-from data_snapshots/token_cache). "
                             f"Kalau memang mau mulai kosong (semua token diekstrak): tambahkan --create-empty.")
        log(f"membuat token store kosong di {store_dir}")
        TokenStore.create_empty(store_dir)
    store = TokenStore(store_dir)

    df = pd.read_parquet(run.snapshot_path, columns=["video_id", "title"])
    n_missing_before = len(store.missing(df["video_id"]))
    stats = {"n_new": 0, "n_thumb_failed_new": 0, "n_total": len(store)}
    if n_missing_before:
        if extractor is None:
            from early_fusion.datasets.token_update import make_extractor
            extractor = make_extractor()
            log(f"extractor: {extractor.info}")
        stats = update_store(store, df, extractor, batch=batch, flush_every=flush_every, log=log)
    still = store.missing(df["video_id"])
    if still:
        raise RuntimeError(f"{len(still)} video snapshot masih tidak punya token setelah update "
                           f"(contoh: {still[:3]})")
    info = dict(store=str(store_dir), n_missing_before=n_missing_before, **stats)
    log(f"token store siap: {info}")
    run.mark("tokens", info)
    return info


# ----------------------------------------------------------------------------------
# Tahap 3: split
# ----------------------------------------------------------------------------------
def stage_split(run, log=print):
    from early_fusion.splits.create_temporal_split import build_split

    run.require("export")
    exp = run.stage_info("export")
    df = pd.read_parquet(run.snapshot_path, columns=["video_id", "split"])
    split = build_split(df, exp["snapshot_hash"])
    for k_run, k_split in (("n_train", "n_train"), ("n_val", "n_val"), ("n_test", "n_test")):
        if split[k_split] != exp[k_run]:
            raise RuntimeError(f"jumlah {k_split} di split ({split[k_split]}) != manifest export ({exp[k_run]})")
    _atomic_write_text(run.split_path, json.dumps(split, indent=2))
    info = dict(n_train=split["n_train"], n_val=split["n_val"], n_test=split["n_test"],
                split_hashes=dict(train_ids_hash=split["train_ids_hash"],
                                  val_ids_hash=split["val_ids_hash"],
                                  test_ids_hash=split["test_ids_hash"]))
    log(f"split ditulis: {run.split_path}  {info['split_hashes']}")
    run.mark("split", info)
    return info


# ----------------------------------------------------------------------------------
# Dependensi torch (dimuat malas; bisa diganti palsu di tes)
# ----------------------------------------------------------------------------------
def _torch_deps():
    import torch
    from early_fusion.experiments.m6_core import load_data, metrics, train_one

    def save(obj, path):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    return SimpleNamespace(load_data=load_data, train_one=train_one, metrics=metrics, save=save, load=load)


# ----------------------------------------------------------------------------------
# Tahap 4: train
# ----------------------------------------------------------------------------------
def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def stage_train(run, *, cfg, seeds, refit_epochs, store_dir, deps=None, log=print):
    """Refit tiap seed di train+val (epoch tetap). Test = 10% terbaru: tidak dilihat kandidat.

    Test dievaluasi di sini HANYA untuk dicatat dan dipakai gerbang Fase 5; tidak ada keputusan
    pelatihan (pemilihan epoch, early stop) yang memakainya, karena fit=trainval tanpa val.
    Idempoten per seed: seed yang ckpt+preds-nya sudah ada dilewati (resume setelah crash).
    """
    from early_fusion.experiments.run_m6_config import cfg_hash

    run.require("export", "tokens", "split")
    deps = deps or _torch_deps()
    data = deps.load_data(spec=run.spec(store_dir))
    chash = cfg_hash(cfg, "full", "trainval")
    run.ckpt_dir.mkdir(parents=True, exist_ok=True)
    run.preds_dir.mkdir(parents=True, exist_ok=True)

    per_seed, seconds = {}, 0.0
    for seed in seeds:
        ck, pr = run.ckpt_dir / f"seed{seed}.pt", run.preds_dir / f"seed{seed}.npz"
        if ck.exists() and pr.exists():
            prev = deps.load(ck)
            if prev.get("cfg_hash") != chash or prev.get("refit_epochs") != refit_epochs:
                raise SystemExit(f"checkpoint {ck} dibuat dengan konfigurasi/epoch berbeda "
                                 f"(cfg_hash {prev.get('cfg_hash')} vs {chash}); pakai --run-id baru")
            z = np.load(pr)
            per_seed[seed] = _spearman(z["test_preds"], z["test_targets"])
            log(f"seed {seed}: sudah ada, dilewati (test_sp={per_seed[seed]:.4f})")
            continue
        res = deps.train_one(cfg, seed, data, variant="full", fit="trainval", fixed_epochs=refit_epochs,
                             eval_test=True, return_state=True, verbose=False)
        seconds += res.get("seconds", 0.0)
        per_seed[seed] = float(res["test_spearman"])
        log(f"seed {seed}: test_sp={res['test_spearman']:.4f} (hanya dicatat) | {res.get('seconds', 0):.0f}s")
        np.savez(pr, test_preds=res["test_preds"], test_targets=res["test_targets"])
        deps.save(dict(state_dict=res["state_dict"], cfg=cfg, variant="full", fit="trainval", seed=seed,
                       scaler=data["scaler"], genres_train=data["genres_train"], n_cont=data["n_cont"],
                       n_genres=data["n_genres"], split_hashes=data["split_hashes"], git_sha=data["git_sha"],
                       cfg_hash=chash, refit_epochs=refit_epochs, run_id=run.run_id), ck)   # ckpt = penanda selesai

    info = dict(seeds=list(seeds), refit_epochs=refit_epochs, cfg=cfg, cfg_hash=chash,
                test_spearman_per_seed={str(k): v for k, v in per_seed.items()}, seconds=round(seconds, 1))
    run.mark("train", info)
    return info


# ----------------------------------------------------------------------------------
# Tahap 5: package
# ----------------------------------------------------------------------------------
def stage_package(run, *, deps=None, log=print):
    from early_fusion.experiments.package_final_model import FORMAT_VERSION, MUST_MATCH, sha256_file

    run.require("export", "split", "train")
    deps = deps or _torch_deps()
    exp, trn = run.stage_info("export"), run.stage_info("train")
    seeds = trn["seeds"]

    ref, ref_key, members, n_params, preds, targets = None, None, [], [], [], None
    for s in seeds:
        f = run.ckpt_dir / f"seed{s}.pt"
        if not f.exists():
            raise SystemExit(f"checkpoint tidak ditemukan: {f}")
        blob = deps.load(f)
        key = json.dumps({k: blob[k] for k in MUST_MATCH}, sort_keys=True, default=str)
        if ref is None:
            ref, ref_key = blob, key
        elif key != ref_key:
            raise SystemExit(f"checkpoint seed {s} tidak konsisten dengan seed {seeds[0]}; menolak mengemas")
        members.append(dict(seed=int(s), state_dict=blob["state_dict"]))
        n_params.append(int(sum(v.numel() for v in blob["state_dict"].values())))
        z = np.load(run.preds_dir / f"seed{s}.npz")
        preds.append(z["test_preds"])
        targets = z["test_targets"]

    single = [deps.metrics(p, targets)["spearman"] for p in preds]
    ens = deps.metrics(np.mean(np.stack(preds, axis=0), axis=0), targets)
    refit_metrics = dict(
        seeds=list(seeds), test_spearman_mean=float(np.mean(single)),
        test_spearman_std=float(np.std(single, ddof=1)) if len(single) > 1 else 0.0,
        test_spearman_ensemble=ens["spearman"], test_auc_ensemble=ens["auc"], test_mae_ensemble=ens["mae"],
    )
    n_rows = dict(total=exp["n_total"], train=exp["n_train"], val=exp["n_val"], test=exp["n_test"])

    name = f"m6_granular_ensemble_{run.run_id}"
    run.bundle_dir.mkdir(parents=True, exist_ok=True)
    out = run.bundle_dir / f"{name}.pt"
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    blob_out = dict(
        format_version=FORMAT_VERSION, name=name, model_class="RATF_M6_Granular_V2", created_at=created_at,
        cfg=ref["cfg"], variant=ref["variant"], fit=ref["fit"], n_cont=ref["n_cont"], n_genres=ref["n_genres"],
        genres_train=ref["genres_train"], scaler=ref["scaler"], split_hashes=ref["split_hashes"],
        snapshot_hash=exp["snapshot_hash"], git_sha=ref.get("git_sha"), seeds=list(seeds), tag=run.run_id,
        refit_metrics=refit_metrics, members=members,
        # tambahan retrain otomatis (M6Ensemble.load menaruhnya di .meta; LoadedRatfBundle membaca train_end)
        train_end=exp["train_end"], n_rows=n_rows, refit_epochs=trn["refit_epochs"], run_id=run.run_id,
    )
    deps.save(blob_out, out)

    digest = sha256_file(out)
    sidecar = dict(
        name=name, file=out.name, sha256=digest, size_mb=round(out.stat().st_size / 1e6, 2),
        created_at=created_at, model_class="RATF_M6_Granular_V2", format_version=FORMAT_VERSION,
        n_members=len(members), member_seeds=list(seeds), params_per_member=n_params[0], variant=ref["variant"],
        trained_on="train+val", cfg=ref["cfg"], split_hashes=ref["split_hashes"],
        snapshot_hash=exp["snapshot_hash"], n_rows=n_rows, train_end=exp["train_end"],
        test_start=exp["test_start"], refit_epochs=trn["refit_epochs"], git_sha=ref.get("git_sha"),
        run_id=run.run_id, refit_metrics=refit_metrics,
    )
    _atomic_write_text(out.with_suffix(".json"), json.dumps(sidecar, indent=2))
    log(f"bundle: {out} ({sidecar['size_mb']} MB, {len(members)} anggota)  sha256={digest[:16]}...")
    info = dict(bundle=str(out), sha256=digest, name=name, n_members=len(members),
                test_spearman_ensemble=ens["spearman"], test_mae_ensemble=ens["mae"])
    run.mark("package", info)
    return info


# ----------------------------------------------------------------------------------
# Tahap 6: verify
# ----------------------------------------------------------------------------------
def stage_verify(run, *, store_dir, log=print):
    """Empat pemeriksaan; semuanya harus lulus. Mengembalikan kontrak JSON (promoted=false)."""
    import torch
    from early_fusion.experiments._common import load_snapshot
    from early_fusion.experiments.m6_core import iterate_batches, load_data, metrics
    from early_fusion.experiments.package_final_model import sha256_file
    from early_fusion.models.m6_ensemble import M6Ensemble

    run.require("export", "split", "train", "package")
    exp, spl, pkg = run.stage_info("export"), run.stage_info("split"), run.stage_info("package")
    bundle = Path(pkg["bundle"])
    side = json.loads(bundle.with_suffix(".json").read_text())
    problems = []

    # 1. berkas = sidecar
    ok1 = sha256_file(bundle) == side["sha256"] == pkg["sha256"]
    log(f"[1/4] HASH FILE     : {'OK' if ok1 else 'GAGAL'}")
    if not ok1:
        problems.append("sha256 bundle tidak cocok dengan sidecar/run.json")

    # 2. bundle bisa dimuat dan metadatanya cocok dengan run
    data = load_data(verbose=False, spec=run.spec(store_dir))
    ens = M6Ensemble.load(bundle, device=data["device"])
    m = ens.meta
    seeds = run.stage_info("train")["seeds"]
    checks = {
        "n_anggota": ens.n_members == len(seeds),
        "snapshot_hash": m.get("snapshot_hash") == exp["snapshot_hash"] == data["snapshot_hash"],
        "split_hashes": m.get("split_hashes") == spl["split_hashes"] == data["split_hashes"],
        "n_rows": m.get("n_rows") == dict(total=exp["n_total"], train=exp["n_train"],
                                          val=exp["n_val"], test=exp["n_test"]),
        "train_end": m.get("train_end") == exp["train_end"],
        "trained_on": m.get("fit") == "trainval",
    }
    ok2 = all(checks.values())
    log(f"[2/4] METADATA      : {'OK' if ok2 else 'GAGAL ' + str([k for k, v in checks.items() if not v])}")
    if not ok2:
        problems.append(f"metadata bundle tidak cocok: {[k for k, v in checks.items() if not v]}")

    # 3. prediksi anggota di test = prediksi yang disimpan tahap train
    store, idx = data["store"], data["test_idx"]
    parts, ys = [], []
    for b in iterate_batches(store, idx, 256, data["device"], shuffle=False):
        parts.append(ens.predict_batch(b["image_tokens"], b["text_tokens"], b["text_mask"],
                                       b["tabular"], b["genre_idx"]))
        ys.append(b["target"].cpu().numpy())
    members = np.concatenate(parts, axis=1)
    y = np.concatenate(ys)
    diffs = []
    for k, s in enumerate(seeds):
        z = np.load(run.preds_dir / f"seed{s}.npz")
        diffs.append(float(np.max(np.abs(z["test_preds"] - members[k]))))
    ens_m = metrics(members.mean(axis=0), y)
    ok3 = max(diffs) < PRED_TOL
    log(f"[3/4] PREDIKSI      : {'OK' if ok3 else 'GAGAL'} (selisih maks {max(diffs):.2e}; "
        f"test ensemble Spearman {ens_m['spearman']:.4f})")
    if not ok3:
        problems.append(f"prediksi bundle berbeda dari yang disimpan (maks {max(diffs):.2e})")

    # 4. jalur baris mentah -> tensor = tensor training
    df, _, _ = load_snapshot(verbose=False, spec=run.spec(store_dir), load_embeddings=False)
    cont, gidx = ens.prepare_tabular(df.iloc[idx])
    tidx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=store["tab"].device)
    ok4 = bool(np.allclose(cont, store["tab"][tidx].cpu().numpy(), atol=1e-5)
               and np.array_equal(gidx, store["gi"][tidx].cpu().numpy()))
    log(f"[4/4] JALUR TABULAR : {'OK' if ok4 else 'GAGAL'}")
    if not ok4:
        problems.append("prepare_tabular tidak identik dengan tensor training")

    if problems:
        raise SystemExit("VERIFIKASI GAGAL: " + "; ".join(problems))
    log("HASIL VERIFIKASI: LULUS")

    decision = dict(
        promoted=False, versioned_path=str(bundle),
        reason="kandidat dibangun dan diverifikasi; gerbang promosi belum dijalankan (Fase 5)",
        new_spearman=float(ens_m["spearman"]), checked_at=_now(),
        stage="verify", run_id=run.run_id, snapshot_hash=exp["snapshot_hash"],
        n_rows=m["n_rows"], train_end=exp["train_end"], test_start=exp["test_start"],
        bundle_sha256=pkg["sha256"], n_members=ens.n_members,
    )
    run.mark("verify", decision)
    return decision


def stage_payload(stage, run, info):
    """JSON baris terakhir untuk tahap selain verify.

    Membawa kunci yang dibaca check_promotion (promoted/versioned_path/reason) supaya SETIAP tahap,
    kalau dijalankan sendirian sebagai task DAG, tetap menghasilkan XCom yang valid. `new_spearman`
    sengaja tidak ada (belum ada nilainya); check_promotion memakai nan sebagai bawaan.
    """
    return {"promoted": False, "versioned_path": None,
            "reason": f"tahap '{stage}' selesai; belum ada kandidat yang dinilai",
            "stage": stage, "status": "ok", "run_id": run.run_id, **info}


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------
def _common(p):
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="folder induk semua run")
    p.add_argument("--run-id", default=None, help="wajib kecuali untuk export/all (yang membuat run baru)")
    p.add_argument("--store", default=str(DEFAULT_STORE), help="direktori TokenStore")


def _args_export(p):
    p.add_argument("--from-parquet", default=None, help="pakai snapshot parquet yang sudah ada (tanpa Postgres)")
    p.add_argument("--max-rows", type=int, default=None, help="ambil N video terbaru saja (uji coba kering)")


def _args_tokens(p):
    p.add_argument("--create-empty", action="store_true")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--flush-every", type=int, default=128)


def _args_train(p):
    p.add_argument("--config", default=str(FINAL_CFG))
    p.add_argument("--seeds", type=int, nargs="+", default=PRODUCTION_SEEDS)
    p.add_argument("--refit-epochs", type=int, default=PRODUCTION_REFIT_EPOCHS)
    p.add_argument("--set", nargs="*", default=[], metavar="K=V",
                   help="override cfg (hanya uji coba kering; dicatat di run.json dan bundle)")


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m early_fusion.retrain", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, extra in (("export", [_args_export]), ("tokens", [_args_tokens]), ("split", []),
                        ("train", [_args_train]), ("package", []), ("verify", []),
                        ("all", [_args_export, _args_tokens, _args_train])):
        p = sub.add_parser(name)
        _common(p)
        for fn in extra:
            fn(p)
    return ap


def _get_run(args, create):
    if create:
        return Run.new(args.root, args.run_id)
    if not args.run_id:
        raise SystemExit("--run-id wajib untuk tahap ini")
    return Run.existing(args.root, args.run_id)


def _train_kwargs(args):
    return dict(cfg=load_production_cfg(args.config, parse_overrides(args.set)), seeds=args.seeds,
                refit_epochs=args.refit_epochs, store_dir=args.store)


def run_stage(args):
    st = args.stage
    if st == "all":
        run = _get_run(args, create=True)
        print(f"run {run.run_id}", flush=True)
        stage_export(run, from_parquet=args.from_parquet, max_rows=args.max_rows)
        stage_tokens(run, store_dir=args.store, create_empty=args.create_empty,
                     batch=args.batch, flush_every=args.flush_every)
        stage_split(run)
        stage_train(run, **_train_kwargs(args))
        stage_package(run)
        return stage_verify(run, store_dir=args.store)

    run = _get_run(args, create=(st == "export"))
    if st == "export":
        info = stage_export(run, from_parquet=args.from_parquet, max_rows=args.max_rows)
    elif st == "tokens":
        info = stage_tokens(run, store_dir=args.store, create_empty=args.create_empty,
                            batch=args.batch, flush_every=args.flush_every)
    elif st == "split":
        info = stage_split(run)
    elif st == "train":
        info = stage_train(run, **_train_kwargs(args))
    elif st == "package":
        info = stage_package(run)
    else:
        return stage_verify(run, store_dir=args.store)
    return stage_payload(st, run, info)


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = run_stage(args)
    emit(payload)            # terakhir; tidak boleh ada print setelah ini
    return 0


if __name__ == "__main__":
    sys.exit(main())
