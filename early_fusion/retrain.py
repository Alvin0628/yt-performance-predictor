"""Retrain otomatis ensemble M6 (Fase 4): enam tahap yang bisa dijalankan terpisah.

    export   Ekspor snapshot dari Postgres ke data_snapshots/retrain/<run_id>/snapshot.parquet
             (urut kronologis, kolom `split` 80/10/10, hash dihitung dari file hasil baca ulang).
    tokens   Perbarui TokenStore (cache token per video_id): hanya video baru yang diproses.
    split    Tulis split.json (format yang sama dengan early_fusion/splits/temporal_no_subs.json).
    train    Refit N seed di train+val dengan resep produksi; checkpoint + prediksi test per seed.
    package  Kemas N checkpoint jadi SATU bundle .pt (+ .json): tulis train_end, hash snapshot,
             jumlah baris, hash split ke dalam bundle.
    verify   Muat bundle, cek hash/metadata/prediksi/jalur tabular; cetak kontrak JSON.
    gate     (Fase 5) Nilai kandidat vs bundle live di 10% terbaru dengan bootstrap berpasangan
             (early_fusion/promotion.py), catat ke promotions.jsonl, cetak keputusan.
    all      export -> tokens -> split -> train -> package -> verify -> gate dalam satu perintah;
             --run-id yang sudah ada = dilanjutkan (tahap yang selesai dilewati).

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
  * verify: kontrak lengkap dengan promoted=false (kandidat terverifikasi, belum dinilai)
  * gate dan all: keputusan promosi sebenarnya, kontrak yang dibaca
    dags/train_model.py::check_promotion
        {"promoted": true|false, "versioned_path": ..., "reason": ..., "new_spearman": ..., ...}
    `gate` MEMUTUSKAN dan MENCATAT. Bundle live baru diganti HANYA bila --promote diberikan, keputusannya
    PROMOTE, dan run standar (early_fusion/live_bundle.py: latest disalin ke prev, lalu latest diganti
    atomik). Kunci `applied` pada JSON = bundle benar-benar sudah diganti. `promoted` true dengan
    `applied` false berarti "akan dipromosikan" (tanpa --promote); DAG me-restart API hanya bila applied.
    --anchor PATH menambah gerbang ke-3 (bandingkan dengan model referensi tetap), --min-new-rows N
    (hanya `all`, butuh snapshot_n_total di m6_latest.json) melewati retraining bila data baru sedikit:
    JSON-nya skipped=true, promoted=false.
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

from early_fusion import live_bundle as lb
from early_fusion import promotion as pm
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

STAGES = ("export", "tokens", "split", "train", "package", "verify", "gate")


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
                 extractor=None, max_fail_frac=0.05, log=print):
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
        stats = update_store(store, df, extractor, batch=batch, flush_every=flush_every,
                             max_fail_frac=max_fail_frac, log=log)
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

    def score(path, df, idx, store, device):
        """Prediksi ensemble (rata-rata anggota) di baris `idx`, dengan scaler/genre MILIK bundle itu.

        Token (gambar/teks) bersifat umum (per video_id, dari TokenStore); tabular disiapkan oleh
        bundle sendiri lewat prepare_tabular, jadi model live dinilai dengan scaler live, bukan scaler
        kandidat. Kandidat dan live lewat fungsi yang sama -> perbandingan simetris.
        """
        from early_fusion.models.m6_ensemble import M6Ensemble
        ens = M6Ensemble.load(path, device=device)
        cont, gidx = ens.prepare_tabular(df.iloc[idx])
        t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=store["img"].device)
        mean = ens.predict(store["img"][t], store["txt"][t], store["mask"][t], cont, gidx)
        return mean, ens.meta

    def shuffle_labels(data, seed):
        """Acak target HANYA di baris train+val (untuk simulasi kandidat rusak). Test tetap asli."""
        idx = np.concatenate([data["train_idx"], data["val_idx"]])
        perm = np.random.default_rng(seed).permutation(len(idx))
        y = data["store"]["y"]
        it = torch.as_tensor(idx, dtype=torch.long, device=y.device)
        y[it] = y[it][torch.as_tensor(perm, dtype=torch.long, device=y.device)]

    return SimpleNamespace(load_data=load_data, train_one=train_one, metrics=metrics, save=save, load=load,
                           score=score, shuffle_labels=shuffle_labels)


# ----------------------------------------------------------------------------------
# Tahap 4: train
# ----------------------------------------------------------------------------------
def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def stage_train(run, *, cfg, seeds, refit_epochs, store_dir, shuffle_labels=None, deps=None, log=print):
    """Refit tiap seed di train+val (epoch tetap). Test = 10% terbaru: tidak dilihat kandidat.

    Test dievaluasi di sini HANYA untuk dicatat dan dipakai gerbang Fase 5; tidak ada keputusan
    pelatihan (pemilihan epoch, early stop) yang memakainya, karena fit=trainval tanpa val.
    Idempoten per seed: seed yang ckpt+preds-nya sudah ada dilewati (resume setelah crash).

    shuffle_labels=SEED (hanya simulasi gerbang, Fase 5): target train+val diacak sebelum latihan,
    test tetap asli. Run seperti ini ditandai `sabotage` di run.json dan bundle, dan gate menolak
    mencatatnya ke promotions.jsonl produksi.
    """
    from early_fusion.experiments.run_m6_config import cfg_hash

    run.require("export", "tokens", "split")
    deps = deps or _torch_deps()
    data = deps.load_data(spec=run.spec(store_dir))
    sabotage = {"shuffle_labels": int(shuffle_labels)} if shuffle_labels is not None else None
    if sabotage:
        deps.shuffle_labels(data, int(shuffle_labels))
        log(f"PERINGATAN: label train+val diacak (seed {shuffle_labels}); ini kandidat SIMULASI.")
    chash = cfg_hash(cfg, "full", "trainval")
    run.ckpt_dir.mkdir(parents=True, exist_ok=True)
    run.preds_dir.mkdir(parents=True, exist_ok=True)

    per_seed, seconds = {}, 0.0
    for seed in seeds:
        ck, pr = run.ckpt_dir / f"seed{seed}.pt", run.preds_dir / f"seed{seed}.npz"
        if ck.exists() and pr.exists():
            prev = deps.load(ck)
            if (prev.get("cfg_hash") != chash or prev.get("refit_epochs") != refit_epochs
                    or prev.get("sabotage") != sabotage):
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
                       cfg_hash=chash, refit_epochs=refit_epochs, run_id=run.run_id,
                       sabotage=sabotage), ck)   # ckpt = penanda selesai

    info = dict(seeds=list(seeds), refit_epochs=refit_epochs, cfg=cfg, cfg_hash=chash, sabotage=sabotage,
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
    if trn.get("sabotage"):
        blob_out["sabotage"] = trn["sabotage"]
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


# ----------------------------------------------------------------------------------
# Tahap 7: gate (Fase 5)
# ----------------------------------------------------------------------------------
def run_deviations(run):
    """Penyimpangan run ini dari resep produksi (kosong = run standar)."""
    exp, trn = run.stage_info("export"), run.stage_info("train")
    dev = {}
    if trn["refit_epochs"] != PRODUCTION_REFIT_EPOCHS:
        dev["refit_epochs"] = trn["refit_epochs"]
    if list(trn["seeds"]) != PRODUCTION_SEEDS:
        dev["seeds"] = list(trn["seeds"])
    if trn.get("cfg") != load_production_cfg():
        dev["cfg"] = "berbeda dari resep produksi"
    if trn.get("sabotage"):
        dev["sabotage"] = trn["sabotage"]
    if exp.get("max_rows"):
        dev["max_rows"] = exp["max_rows"]
    return dev


def check_gate_args(live_path, first_promotion):
    """Tepat satu dari --live PATH atau --first-promotion; path live yang tidak ada = error.

    Sengaja eksplisit: path live yang salah ketik TIDAK boleh diam-diam menjadi "promosi pertama".
    """
    if bool(live_path) == bool(first_promotion):
        raise SystemExit("gate: pilih tepat satu: --live PATH_BUNDLE_LIVE atau --first-promotion "
                         "(tidak ada bundle live sama sekali)")
    if live_path and not Path(live_path).exists():
        raise SystemExit(f"gate: bundle live tidak ditemukan: {live_path}")


def stage_gate(run, *, store_dir, live_path=None, first_promotion=False, live_train_end=None,
               log_path=pm.PROMOTION_LOG_PATH, anchor_path=None, promote=False,
               bundles_dir=lb.DEFAULT_BUNDLES, deps=None, log=print):
    """Nilai kandidat dan bundle live di 10% terbaru snapshot ini; putuskan; catat; (opsional) promosikan.

    Test (10% terbaru) tidak pernah dilihat kandidat (fit=trainval) maupun model live (check_no_leak).
    Keduanya dinilai lewat fungsi yang sama, masing-masing dengan scaler/genre miliknya.

    anchor_path: model referensi tetap; hanya dipakai bila ada bundle live (bukan --first-promotion).
    promote: bila True dan keputusan PROMOTE, salin kandidat ke bundles_dir/m6_latest.* (live lama ke m6_prev.*).
    Hanya run standar yang boleh dipromosikan. Kegagalan menyalin dicatat (applied=false, apply_error) lalu
    task digagalkan: keputusan dan keadaan berkas tidak boleh saling berbohong.
    """
    from features.target import compute_target
    from early_fusion.splits.load_split import apply_split_to_df, load_canonical_split

    run.require("export", "split", "train", "package", "verify")
    check_gate_args(live_path, first_promotion)
    if anchor_path and not Path(anchor_path).exists():
        raise SystemExit(f"gate: bundle anchor tidak ditemukan: {anchor_path}")
    dev = run_deviations(run)
    if dev and Path(log_path).resolve() == Path(pm.PROMOTION_LOG_PATH).resolve():
        raise SystemExit(f"gate: run ini menyimpang dari resep produksi {dev}; jangan dicatat ke "
                         f"{pm.PROMOTION_LOG_PATH}. Pakai --log-path lain (mis. data_snapshots/promotions_sim.jsonl).")
    if dev and promote:
        raise SystemExit(f"gate: --promote hanya untuk run standar; run ini menyimpang dari resep produksi {dev}")

    deps = deps or _torch_deps()
    spec = run.spec(store_dir)
    data = deps.load_data(spec=spec)
    df = pd.read_parquet(run.snapshot_path)
    df = df.drop(columns=[c for c in EMBEDDING_COLS if c in df.columns]).reset_index(drop=True)
    df["target"] = compute_target(df["views"], df["trailing_avg_views"])
    tr, va, te = apply_split_to_df(df, load_canonical_split(verbose=False, spec=spec))
    targets = df["target"].values[te]
    test_start = str(df["published_at"].iloc[te].min())

    pkg = run.stage_info("package")
    cand_path = Path(pkg["bundle"])
    new_preds, cand_meta = deps.score(cand_path, df, te, data["store"], data["device"])

    old_preds, live_sha = None, None
    anchor_preds, anchor_sha, anchor_end = None, None, None
    if live_path:
        try:
            live_side = lb.verify_sidecar(live_path)
        except ValueError as e:
            raise SystemExit(f"gate: {e}")
        old_preds, live_meta = deps.score(live_path, df, te, data["store"], data["device"])
        live_sha = pm.sha256_file(live_path)
        live_end = live_meta.get("train_end") or (live_side or {}).get("train_end") or live_train_end
        if live_end is None:
            raise SystemExit("gate: bundle live tidak punya train_end (bundle lama). Berikan --live-train-end "
                             "'<waktu publikasi terbaru di train+val bundle live>' (atau m6_latest.json dari "
                             "`python -m early_fusion.live_bundle bootstrap`) supaya bisa dipastikan "
                             "live tidak pernah melihat baris test.")
        try:
            pm.check_no_leak(live_end, test_start)
        except ValueError as e:
            raise SystemExit(f"gate: {e}")
        live_train_end = str(live_end)

        if anchor_path:
            try:
                anchor_side = lb.verify_sidecar(anchor_path)
            except ValueError as e:
                raise SystemExit(f"gate (anchor): {e}")
            anchor_preds, anchor_meta = deps.score(anchor_path, df, te, data["store"], data["device"])
            anchor_sha = pm.sha256_file(anchor_path)
            anchor_end = anchor_meta.get("train_end") or (anchor_side or {}).get("train_end")
            if anchor_end is None:
                raise SystemExit("gate (anchor): bundle anchor tidak punya train_end; buat m6_anchor.json lewat "
                                 "`python -m early_fusion.live_bundle bootstrap`")
            try:
                pm.check_no_leak(anchor_end, test_start)
            except ValueError as e:
                raise SystemExit(f"gate (anchor): {e}")
            anchor_end = str(anchor_end)

    baseline = pm.linear_baseline_spearman(df.iloc[np.concatenate([tr, va])], df.iloc[te])
    extra = dict(
        model_kind="m6_ensemble", run_id=run.run_id, snapshot_hash=run.stage_info("export")["snapshot_hash"],
        n_test=int(len(te)), test_start=test_start, candidate_train_end=cand_meta.get("train_end"),
        candidate_sha256=pkg["sha256"], n_members=len(run.stage_info("train")["seeds"]),
        live_path=str(live_path) if live_path else None, live_sha256=live_sha, live_train_end=live_train_end,
        baseline_fit="train+val", bootstrap_seed=pm.BOOTSTRAP_SEED,
    )
    if anchor_preds is not None:
        extra.update(anchor_path=str(anchor_path), anchor_sha256=anchor_sha, anchor_train_end=anchor_end)
    if dev:
        extra["deviations"] = dev
    decision = pm.decide(new_preds=new_preds, targets=targets, baseline_spearman=baseline,
                         old_preds=old_preds, first_promotion=first_promotion,
                         versioned_path=cand_path, extra=extra, anchor_preds=anchor_preds)

    decision.update(promote_requested=bool(promote), applied=False)
    apply_error = None
    if promote and decision["promoted"]:
        try:
            applied = lb.promote(cand_path, bundles_dir)
            decision.update(applied=True, applied_latest=applied["latest"], applied_prev=applied["prev"])
        except Exception as e:                       # dicatat apa adanya, lalu task digagalkan di bawah
            decision.update(applied=False, apply_error=repr(e))
            apply_error = e
    pm.log_decision(decision, log_path)

    verdict = "PROMOTED" if decision["promoted"] else "REJECTED"
    if decision["promoted"]:
        verdict += " (APPLIED)" if decision["applied"] else " (tidak diterapkan: tanpa --promote)" if not promote else " (GAGAL diterapkan)"
    ci = (f" CI=[{decision['ci_lower']:.4f}, {decision['ci_upper']:.4f}] margin={decision['ci_reject_margin']}"
          if "ci_lower" in decision else "")
    anc = (f" anchor={decision['anchor_spearman']:.4f} CI=[{decision['anchor_ci_lower']:.4f}, "
           f"{decision['anchor_ci_upper']:.4f}] margin={decision['anchor_margin']}" if "anchor_ci_lower" in decision else "")
    log(f"[gate] {verdict} -- {decision['reason']} | new={decision['new_spearman']:.4f} "
        f"old={decision.get('old_spearman', float('nan')):.4f} baseline={baseline:.4f}{ci}{anc}")
    log(f"[gate] dicatat di {log_path}")
    if apply_error is not None:
        raise RuntimeError(f"gate: keputusan PROMOTE, tetapi bundle gagal diganti: {apply_error!r} "
                           f"(m6_latest tetap bundle lama; keputusan dicatat dengan applied=false)") from apply_error
    result = {**decision, "stage": "gate", "run_id": run.run_id}
    run.mark("gate", result)
    return result


def stage_payload(stage, run, info):
    """JSON baris terakhir untuk tahap selain verify.

    Membawa kunci yang dibaca check_promotion (promoted/versioned_path/reason) supaya SETIAP tahap,
    kalau dijalankan sendirian sebagai task DAG, tetap menghasilkan XCom yang valid. `new_spearman`
    sengaja tidak ada (belum ada nilainya); check_promotion memakai nan sebagai bawaan.
    """
    return {"promoted": False, "applied": False, "versioned_path": None,
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
    p.add_argument("--max-fail-frac", type=float, default=0.05,
                   help="hentikan (tanpa menulis ke store) bila lebih dari ini thumbnail gagal dimuat dalam satu kelompok")


def _args_train(p):
    p.add_argument("--config", default=str(FINAL_CFG))
    p.add_argument("--seeds", type=int, nargs="+", default=PRODUCTION_SEEDS)
    p.add_argument("--refit-epochs", type=int, default=PRODUCTION_REFIT_EPOCHS)
    p.add_argument("--threads", type=int, default=None,
                   help="jumlah thread CPU torch (di VM sisakan core untuk API)")
    p.add_argument("--set", nargs="*", default=[], metavar="K=V",
                   help="override cfg (hanya uji coba kering; dicatat di run.json dan bundle)")
    p.add_argument("--shuffle-labels", type=int, default=None, metavar="SEED",
                   help="SIMULASI gerbang: acak target train+val (kandidat rusak); run ditandai sabotage")


def _args_gate(p):
    p.add_argument("--live", default=None, help="path bundle live yang dibandingkan (wajib, atau --first-promotion)")
    p.add_argument("--first-promotion", action="store_true", help="tidak ada bundle live sama sekali")
    p.add_argument("--live-train-end", default=None,
                   help="train_end bundle live bila bundle itu tidak menyimpannya (bundle lama)")
    p.add_argument("--log-path", default=pm.PROMOTION_LOG_PATH,
                   help="berkas JSONL keputusan (run non-standar/simulasi tidak boleh ke berkas produksi)")
    p.add_argument("--anchor", default=None,
                   help="bundle referensi tetap (m6_anchor.pt): gerbang ke-3, kandidat tidak boleh yakin lebih buruk")
    p.add_argument("--promote", action="store_true",
                   help="bila keputusan PROMOTE (run standar), ganti bundles-dir/m6_latest.* dan simpan yang lama di m6_prev.*")
    p.add_argument("--bundles-dir", default=str(lb.DEFAULT_BUNDLES), help="tempat m6_latest/m6_prev (untuk --promote)")


def _args_all(p):
    p.add_argument("--min-new-rows", type=int, default=0,
                   help="lewati retraining bila video baru sejak bundle live (snapshot_n_total di m6_latest.json) "
                        "kurang dari ini; 0 = selalu jalan")


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m early_fusion.retrain", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, extra in (("export", [_args_export]), ("tokens", [_args_tokens]), ("split", []),
                        ("train", [_args_train]), ("package", []), ("verify", []), ("gate", [_args_gate]),
                        ("all", [_args_export, _args_tokens, _args_train, _args_gate, _args_all])):
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


def _get_run_all(args):
    """`all` membuat run baru, kecuali --run-id menunjuk run yang sudah ada: itu dilanjutkan."""
    if args.run_id and Run(args.root, args.run_id).state_path.exists():
        return Run.existing(args.root, args.run_id), True
    return Run.new(args.root, args.run_id), False


def _apply_threads(args):
    if getattr(args, "threads", None):
        import torch
        torch.set_num_threads(int(args.threads))


def _train_kwargs(args):
    return dict(cfg=load_production_cfg(args.config, parse_overrides(args.set)), seeds=args.seeds,
                refit_epochs=args.refit_epochs, store_dir=args.store, shuffle_labels=args.shuffle_labels)


def _gate_kwargs(args):
    return dict(store_dir=args.store, live_path=args.live, first_promotion=args.first_promotion,
                live_train_end=args.live_train_end, log_path=args.log_path, anchor_path=args.anchor,
                promote=args.promote, bundles_dir=args.bundles_dir)


def _maybe_skip(run, args, log=print):
    """Lewati retraining bila data baru terlalu sedikit. None = lanjut; dict = hasil 'skipped' (sudah ditandai)."""
    n_min = getattr(args, "min_new_rows", 0) or 0
    if not n_min or not args.live:
        return None
    live_n = lb.snapshot_n_total(lb.read_sidecar(args.live))
    n_total = run.stage_info("export")["n_total"]
    if live_n is None:
        log("[skip-check] bundle live tidak mencatat snapshot_n_total: pemeriksaan data baru dilewati")
        return None
    n_new = int(n_total) - int(live_n)
    if n_new >= n_min:
        log(f"[skip-check] {n_new} video baru sejak bundle live (minimum {n_min}): retraining dilanjutkan")
        return None
    result = {"promoted": False, "applied": False, "skipped": True, "versioned_path": None, "new_spearman": None,
              "stage": "skip", "run_id": run.run_id, "n_total": int(n_total), "live_n": int(live_n), "n_new": n_new,
              "reason": f"hanya {n_new} video baru sejak bundle live (minimum {n_min}); retraining dilewati"}
    run.mark("gate", result)
    return result


def run_stage(args):
    st = args.stage
    if st == "all":
        check_gate_args(args.live, args.first_promotion)          # gagal cepat, sebelum run dibuat / training
        run, resumed = _get_run_all(args)
        print(f"run {run.run_id}" + (" (dilanjutkan)" if resumed else ""), flush=True)
        done = run.load_state()["stages"]
        if "gate" in done:                                # run sudah tuntas: tampilkan keputusan yang tersimpan
            return {k: v for k, v in done["gate"].items() if k != "finished_at"}
        _apply_threads(args)
        if "export" not in done:
            stage_export(run, from_parquet=args.from_parquet, max_rows=args.max_rows)
        skipped = _maybe_skip(run, args)
        if skipped:
            return skipped
        if "tokens" not in done:
            stage_tokens(run, store_dir=args.store, create_empty=args.create_empty, batch=args.batch,
                         flush_every=args.flush_every, max_fail_frac=args.max_fail_frac)
        if "split" not in done:
            stage_split(run)
        if "train" not in done:
            stage_train(run, **_train_kwargs(args))
        if "package" not in done:
            stage_package(run)
        if "verify" not in done:
            stage_verify(run, store_dir=args.store)
        return stage_gate(run, **_gate_kwargs(args))

    run = _get_run(args, create=(st == "export"))
    if st == "export":
        info = stage_export(run, from_parquet=args.from_parquet, max_rows=args.max_rows)
    elif st == "tokens":
        info = stage_tokens(run, store_dir=args.store, create_empty=args.create_empty,
                            batch=args.batch, flush_every=args.flush_every, max_fail_frac=args.max_fail_frac)
    elif st == "split":
        info = stage_split(run)
    elif st == "train":
        _apply_threads(args)
        info = stage_train(run, **_train_kwargs(args))
    elif st == "package":
        info = stage_package(run)
    elif st == "gate":
        return stage_gate(run, **_gate_kwargs(args))
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
