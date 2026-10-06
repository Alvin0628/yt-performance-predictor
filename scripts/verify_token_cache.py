"""Verify token cache: pooled output from cache should ≈ embedding in snapshot.

Sample 200 random rows, compute pooled output from image token cache, and compare
cosine with image_embedding from snapshot parquet.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from early_fusion.datasets.clip_tokens import load_clip
from early_fusion.datasets.token_cache import load_cache


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def parse_vector(s):
    return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)


def main():
    df = pd.read_parquet("data_snapshots/snapshot.parquet")
    img, txt, mask, thumb_ok, index_df, meta = load_cache("data_snapshots/token_cache")

    # Sanity checks
    assert img.shape[0] == len(df), f"mismatch: cache={img.shape[0]} snapshot={len(df)}"
    assert (index_df["video_id"].values == df["video_id"].values).all(), \
        "index order mismatch"
    print(f"cache size: {len(df)}")
    print(f"thumb_ok frac: {thumb_ok.mean():.4f}")

    # NaN checks
    sample = np.random.default_rng(0).choice(len(df), size=min(500, len(df)), replace=False)
    assert not np.isnan(img[sample].astype(np.float32)).any(), "NaN in img tokens"
    assert not np.isnan(txt[sample].astype(np.float32)).any(), "NaN in txt tokens"
    print("NaN check: OK")

    # Image pooled parity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, encode_fn, transform, tok, cfg = load_clip(device)

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(df), size=200, replace=False)
    cosines = []
    for i in idxs:
        # Pooled from token cache
        t = torch.from_numpy(img[i].astype(np.float32)).unsqueeze(0).to(device)
        with torch.inference_mode():
            pooled = clip.visual_projection(
                clip.vision_model.post_layernorm(t[:, 0])
            ).cpu().numpy()
        # Reference from snapshot
        ref = parse_vector(df.iloc[i]["image_embedding"])
        cosines.append(cos(ref, pooled))

    cosines = np.array(cosines)
    print()
    print(f"=== Image pooled parity (n=200) ===")
    print(f"  mean cosine: {cosines.mean():.6f}")
    print(f"  min cosine:  {cosines.min():.6f}")
    print(f"  max cosine:  {cosines.max():.6f}")
    print(f"  frac > 0.999: {(cosines > 0.999).mean():.4f}")

    assert cosines.min() > 0.999, f"image parity failed: min={cosines.min()}"

    # Text pooled parity
    # NOTE: we do not store input_ids in the cache, so EOS cannot be rebuilt
    # accurately. What we can check: text token shape and mask.
    n_valid = mask[idxs].sum(axis=1)
    print()
    print(f"=== Text token stats (n=200) ===")
    print(f"  mean valid tokens: {n_valid.mean():.2f}")
    print(f"  min valid tokens:  {n_valid.min()}")
    print(f"  max valid tokens:  {n_valid.max()}")
    assert n_valid.min() >= 2, "there is text with < 2 tokens (BOS+EOS)"

    print()
    print("VERIFY OK")


if __name__ == "__main__":
    main()