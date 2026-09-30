"""Build the token cache from the snapshot.

Input:  data_snapshots/snapshot.parquet  +  MinIO thumbnails
Output: data_snapshots/token_cache/      (~1 GB memmap fp16)

Prereq: L has already been decided in Step 3 (audit_title_lengths).
"""
import sys
import json
import io
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import transformers

from early_fusion.datasets.clip_tokens import load_clip, image_tokens, text_tokens
from early_fusion.datasets.token_cache import TokenCacheWriter
from models.precompute_embeddings import minio_client, MINIO_BUCKET


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
MANIFEST = Path("data_snapshots/split_manifest.json")
OUT = Path("data_snapshots/token_cache")

L = 32          # Step 3: p99=29, 0.59% truncate
BATCH = 16


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def load_thumb(video_id, transform, device):
    """Fetch thumbnail from MinIO and transform it into a tensor. Return (tensor, ok)."""
    key = f"{video_id}.jpg"
    try:
        obj = minio_client.get_object(Bucket=MINIO_BUCKET, Key=key)
        img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
        return transform(img), True
    except Exception:
        return torch.zeros(3, 224, 224), False


def main():
    df = pd.read_parquet(SNAPSHOT)
    n = len(df)
    print(f"snapshot rows: {n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    clip, encode_fn, transform, tok, cfg = load_clip(device)

    writer = TokenCacheWriter(OUT, n=n, L=L)
    n_thumb_ok = 0
    video_ids = df["video_id"].tolist()

    for start in tqdm(range(0, n, BATCH)):
        batch = df.iloc[start:start + BATCH]
        imgs, oks = [], []
        for _, row in batch.iterrows():
            img, ok = load_thumb(row["video_id"], transform, device)
            imgs.append(img)
            oks.append(ok)
        x = torch.stack(imgs).to(device)
        with torch.inference_mode():
            it = image_tokens(clip, x).cpu().numpy()               # (B, 50, 768)
            tt, tm, _ = text_tokens(clip, tok, batch["title"].tolist(), device, L)
            tt, tm = tt.cpu().numpy(), tm.cpu().numpy()             # (B, L, 512), (B, L)
        for j in range(len(batch)):
            writer.write(start + j, it[j], tt[j], tm[j], oks[j])
            if oks[j]:
                n_thumb_ok += 1

    snapshot_manifest = json.loads(MANIFEST.read_text())
    meta = {
        "n": n,
        "L": L,
        "img_tokens": 50,
        "img_dim": 768,
        "txt_dim": 512,
        "encoder": "clip_b32",
        "image_mode": "squash",
        "transformers_version": transformers.__version__,
        "git_hash": git_hash(),
        "snapshot_hash": snapshot_manifest["snapshot_hash"],
        "snapshot_n_total": snapshot_manifest["n_total"],
        "n_thumb_ok": int(n_thumb_ok),
        "n_thumb_failed": int(n - n_thumb_ok),
    }
    writer.close(meta, video_ids)

    print()
    print(f"done: {OUT}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()