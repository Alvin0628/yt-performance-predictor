"""
Serving parity for the early-fusion bundle: raw JPEG -> LoadedRatfBundle.predict()
must give the same score as the training path (token cache -> M6Ensemble).

Run from the repo root, with MinIO reachable (docker compose up -d minio) and the
token cache + snapshot present (data_snapshots/):

    set POSTGRES_USER=x & set POSTGRES_PASSWORD=x & set POSTGRES_DB=x      (values are unused, import needs them)
    set MINIO_ENDPOINT=localhost:9000 & set MINIO_ROOT_USER=... & set MINIO_ROOT_PASSWORD=...
    python -m tests.test_serving_parity --model early_fusion/models/final/m6_granular_ensemble_v1.pt --n 40

Pass criteria (adjust after seeing the real numbers): max |diff| < 1e-2 and Spearman > 0.999.
"""
import argparse
import io
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from early_fusion.experiments._common import load_snapshot
from early_fusion.experiments.m6_core import iterate_batches, load_data
from early_fusion.models.m6_ensemble import M6Ensemble
from models.precompute_embeddings import MINIO_BUCKET, minio_client
from serving.ratf_bundle import LoadedRatfBundle

MAX_DIFF = 1e-2
MIN_SPEARMAN = 0.999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    data = load_data(verbose=False)
    dev = data["device"]
    df, _, _ = load_snapshot(verbose=False)

    test_idx = np.asarray(data["test_idx"])
    sel = test_idx[np.linspace(0, len(test_idx) - 1, args.n).astype(int)]

    # --- training path (token cache) ---
    ens = M6Ensemble.load(args.model, device=dev)
    parts = []
    for b in iterate_batches(data["store"], sel, 64, dev, shuffle=False):
        parts.append(ens.predict_batch(b["image_tokens"], b["text_tokens"], b["text_mask"],
                                       b["tabular"], b["genre_idx"]))
    cache_scores = np.concatenate(parts, axis=1).mean(axis=0)

    # --- serving path (raw JPEG from MinIO) ---
    serving = LoadedRatfBundle(args.model, device=torch.device("cpu"))
    serve_scores = []
    for i in sel:
        r = df.iloc[int(i)]
        body = minio_client.get_object(Bucket=MINIO_BUCKET, Key=f"{r['video_id']}.jpg")["Body"].read()
        out = serving.predict(
            thumbnail=Image.open(io.BytesIO(body)), title=r["title"],
            trailing_avg_views=r["trailing_avg_views"], duration_seconds=r["duration_seconds"],
            genre=r["genre"],
        )
        serve_scores.append(out["score"])
    serve_scores = np.asarray(serve_scores)

    diff = np.abs(cache_scores - serve_scores)
    sp = spearmanr(cache_scores, serve_scores).statistic
    print(f"n={len(sel)}  max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}  spearman={sp:.5f}")
    ok = diff.max() < MAX_DIFF and sp > MIN_SPEARMAN
    print("PARITY", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
