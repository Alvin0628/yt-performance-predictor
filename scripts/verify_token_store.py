"""Fase 3 gate: the token store must reproduce the legacy cache, and incremental extraction
must reproduce the legacy tokens for videos it extracts itself.

Needs the local data (data_snapshots/snapshot.parquet and data_snapshots/token_cache), and for
the extraction check also MinIO (docker compose up -d minio) + POSTGRES_*/MINIO_* env vars + CLIP.

    python -m scripts.verify_token_store --skip-extract      # checks 1 + 2 (no CLIP, no MinIO)
    python -m scripts.verify_token_store                     # all checks

Checks
  1. COPY      store.take(snapshot ids) is bit-identical to the legacy cache.
  2. LOAD_DATA load_data(store spec) gives the same tensors as load_data(default spec).
  3. EXTRACT   a store missing its last videos is brought up to date with update_store() and the
               newly extracted rows are compared with the legacy tokens. Batches are aligned to
               the legacy batches of 16, so identical results are expected on the same device and
               library versions; tiny fp16-level differences are reported as such.
  (info) a random-batch re-extraction, to show how much batch composition matters.
"""
import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from early_fusion.data_spec import DEFAULT_SPEC
from early_fusion.datasets.token_cache import load_cache
from early_fusion.datasets.token_store import TokenStore, META_FILE

LEGACY_BATCH = 16


REL_L2_FP16 = 2e-3      # batas "selisih pembulatan fp16" untuk galat relatif L2 per token


def _compare(a, b, name):
    """Bandingkan dua array berbentuk sama. Return dict ringkas.

    Untuk float: galat relatif L2 per token (norma selisih / norma acuan), karena selisih absolut
    saja tidak berarti tanpa tahu besar nilai tokennya.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print(f"    {name}: BENTUK {a.shape} vs {b.shape}")
        return dict(exact=False, ok=False, max_abs=np.inf, equal=0.0, rel_max=np.inf, rel_mean=np.inf)
    exact = bool(np.array_equal(a, b))
    equal = float((a == b).mean())
    if a.dtype == np.bool_:
        return dict(exact=exact, ok=exact, max_abs=0.0 if exact else 1.0, equal=equal, rel_max=0.0, rel_mean=0.0)
    af, bf = a.astype(np.float32), b.astype(np.float32)
    num = np.linalg.norm(af - bf, axis=-1)
    den = np.linalg.norm(af, axis=-1) + 1e-12
    rel = num / den
    return dict(exact=exact, ok=exact or bool(rel.max() <= REL_L2_FP16), max_abs=float(np.abs(af - bf).max()),
                equal=equal, rel_max=float(rel.max()), rel_mean=float(rel.mean()))


def _show(name, r):
    verdict = "bit-identical" if r["exact"] else ("fp16-level" if r["ok"] else "TOO LARGE")
    print(f"    {name:8s} {verdict:13s} equal elements={r['equal']:.6f}  max|diff|={r['max_abs']:.3e}  "
          f"rel L2 per token: max={r['rel_max']:.2e} mean={r['rel_mean']:.2e}")


def check_copy(store, legacy_dir, df):
    img, txt, mask, thumb_ok, index_df, _meta = load_cache(legacy_dir)
    ids = df["video_id"].tolist()
    assert ids == index_df["video_id"].tolist(), "legacy index order differs from the snapshot order"
    good = True
    for start in range(0, len(ids), 2048):
        sl = slice(start, start + 2048)
        t = store.take(ids[sl])
        for name, got, ref in (("img", t["img"], img[sl]), ("txt", t["txt"], txt[sl]),
                               ("mask", t["mask"], mask[sl]), ("thumb_ok", t["thumb_ok"], thumb_ok[sl])):
            if not np.array_equal(got, np.asarray(ref)):
                good = False
                print(f"    {name} differs in rows {start}..{start + 2048}")
    return good


def check_load_data(store_dir):
    import torch
    from early_fusion.experiments.m6_core import load_data
    d1 = load_data(verbose=False)
    ref = {k: v.cpu() for k, v in d1["store"].items()}
    idx1 = (np.asarray(d1["train_idx"]), np.asarray(d1["val_idx"]), np.asarray(d1["test_idx"]))
    meta1 = (d1["n_cont"], d1["n_genres"])
    del d1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    spec = replace(DEFAULT_SPEC, cache_dir=Path(store_dir), cache_kind="store")
    d2 = load_data(verbose=False, spec=spec)
    good = True
    for k, v in d2["store"].items():
        same = torch.equal(v.cpu(), ref[k])
        print(f"    store[{k}] {'identical' if same else 'DIFFERS'}")
        good &= same
    same = (d2["n_cont"], d2["n_genres"]) == meta1
    print(f"    n_cont/n_genres {'identical' if same else 'DIFFERS'}")
    good &= same
    idx2 = (np.asarray(d2["train_idx"]), np.asarray(d2["val_idx"]), np.asarray(d2["test_idx"]))
    for a, b, nm in zip(idx1, idx2, ("train", "val", "test")):
        same = bool(np.array_equal(a, b))
        print(f"    {nm}_idx {'identical' if same else 'DIFFERS'}")
        good &= same
    return good


def check_extract(legacy_dir, df, simulate, work_dir):
    from early_fusion.datasets.token_update import make_extractor, update_store
    img, txt, mask, thumb_ok, index_df, meta = load_cache(legacy_dir)
    n = len(df)
    keep = max(0, (n - simulate) // LEGACY_BATCH * LEGACY_BATCH)     # aligned to the legacy batches
    sim = Path(work_dir) / "token_store_sim"
    if sim.exists():
        shutil.rmtree(sim)
    TokenStore.create_from_legacy(legacy_dir, sim, keep_first=keep)
    store = TokenStore(sim)
    extract = make_extractor()                                       # numerics="legacy" by default
    print(f"    extractor: {extract.info}; legacy cache built with transformers {meta.get('transformers_version')}")
    stats = update_store(store, df, extract, batch=LEGACY_BATCH, log=lambda *_: None)
    print(f"    simulated {n - keep} new videos (store had {keep}): {stats}")
    ids = df["video_id"].tolist()[keep:]
    t = store.take(ids)
    results = {}
    for name, got, ref in (("img", t["img"], img[keep:]), ("txt", t["txt"], txt[keep:]),
                           ("mask", t["mask"], mask[keep:]), ("thumb_ok", t["thumb_ok"], thumb_ok[keep:])):
        results[name] = _compare(got, np.asarray(ref), name)
        _show(name, results[name])
    shutil.rmtree(sim)

    # --- informasi: seberapa besar efek TF32 (yang diaktifkan m6_core untuk training) ---
    rng = np.random.default_rng(0)
    pick = rng.choice(n, size=min(64, n), replace=False)
    pids = [df["video_id"].iloc[i] for i in pick]
    ptitles = [df["title"].iloc[i] if isinstance(df["title"].iloc[i], str) else "" for i in pick]
    for mode in ("legacy", "tf32"):
        ex = make_extractor(numerics_mode=mode)
        parts = [ex(pids[i:i + LEGACY_BATCH], ptitles[i:i + LEGACY_BATCH]) for i in range(0, len(pids), LEGACY_BATCH)]
        r = _compare(np.concatenate([p_[0] for p_ in parts]).astype(np.float16), np.asarray(img[pick]), "img")
        _show(f"(info) img, 64 acak, numerics={mode}", r)

    exact_all = all(r["exact"] for r in results.values())
    ok_all = all(r["ok"] for r in results.values())
    return exact_all, ok_all


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="data_snapshots/snapshot.parquet")
    ap.add_argument("--legacy", default="data_snapshots/token_cache")
    ap.add_argument("--store", default="data_snapshots/token_store")
    ap.add_argument("--simulate", type=int, default=48, help="how many 'new' videos to simulate in check 3")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-load-data", action="store_true")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.snapshot, columns=["video_id", "title"])
    store_dir = Path(args.store)
    if not (store_dir / META_FILE).exists():
        print(f"bootstrapping store {store_dir} from {args.legacy}")
        TokenStore.create_from_legacy(args.legacy, store_dir)
    store = TokenStore(store_dir)
    print(f"store: {len(store)} rows, {store.meta.get('n_thumb_failed')} failed thumbnails")

    results = {}
    print("[1] COPY: store.take(snapshot ids) vs legacy cache")
    results["COPY"] = check_copy(store, args.legacy, df)
    print("    ->", "identical" if results["COPY"] else "DIFFERS")

    if not args.skip_load_data:
        print("[2] LOAD_DATA: store spec vs default spec")
        results["LOAD_DATA"] = check_load_data(store_dir)
        print("    ->", "identical" if results["LOAD_DATA"] else "DIFFERS")

    if not args.skip_extract:
        print("[3] EXTRACT: incremental extraction vs legacy tokens")
        exact, ok16 = check_extract(args.legacy, df, args.simulate, store_dir.parent)
        results["EXTRACT"] = exact or ok16
        print("    ->", "bit-identical" if exact else ("fp16-level differences only" if ok16 else "TOO LARGE"))
        results["EXTRACT_EXACT"] = exact

    print()
    for k, v in results.items():
        print(f"  {k:14s} {'OK' if v else 'FAILED'}")
    gate = all(v for k, v in results.items() if k != "EXTRACT_EXACT")
    print("FASE 3 GATE:", "LULUS" if gate else "GAGAL")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
