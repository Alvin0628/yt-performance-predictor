"""Measure title token length + thumbnail size from the snapshot.

Output:
- Title token length distribution (to determine L)
- Thumbnail size distribution (to confirm transform assumptions)
"""
import sys
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

from models.precompute_embeddings import ENCODERS, minio_client, MINIO_BUCKET


SNAPSHOT = Path("data_snapshots/snapshot.parquet")
OUT = Path("data_snapshots/audit_metadata.json")


def audit_titles(df, tok):
    titles = df["title"].astype(str).tolist()
    lens = [len(ids) for ids in tok(titles)["input_ids"]]
    arr = np.array(lens)
    stats = {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }
    for L in [16, 24, 32, 40, 48, 56, 64, 77]:
        stats[f"frac_over_{L}"] = float((arr > L).mean())
    return stats


def audit_thumbnails(df, sample_size=500):
    """Read PIL headers only (fast, does not download full body).
    PIL needs the first few KB to parse the JPG header."""
    rng = np.random.default_rng(0)
    n = len(df)
    idxs = rng.choice(n, size=min(sample_size, n), replace=False)

    widths, heights, modes = [], [], []
    for i in idxs:
        vid = df.iloc[i]["video_id"]
        try:
            # Fetch byte range (0 - 4096) — enough for JPG header
            resp = minio_client.get_object(
                Bucket=MINIO_BUCKET, Key=f"{vid}.jpg", Range="bytes=0-4095"
            )
            img = Image.open(io.BytesIO(resp["Body"].read()))
            widths.append(img.size[0])
            heights.append(img.size[1])
            modes.append(img.mode)
        except Exception as e:
            print(f"  [warn] {vid}: {e}")

    w = np.array(widths)
    h = np.array(heights)
    stats = {
        "n_sampled": int(len(w)),
        "n_failed": int(len(idxs) - len(w)),
        "width_unique": sorted(set(w.tolist())),
        "height_unique": sorted(set(h.tolist())),
        "mode_counts": {m: modes.count(m) for m in set(modes)},
        "all_same_size": bool(len(set(w.tolist())) == 1 and len(set(h.tolist())) == 1),
    }
    return stats


def main():
    df = pd.read_parquet(SNAPSHOT)
    print(f"loaded {len(df)} rows from {SNAPSHOT}")
    print()

    # Title length
    tok = AutoTokenizer.from_pretrained(ENCODERS["clip_b32"]["hf_name"])
    title_stats = audit_titles(df, tok)
    print("=== Title token length distribution ===")
    print(json.dumps(title_stats, indent=2))
    print()

    # L recommendation
    print("L recommendation (choose L with frac_over_L <= 0.01):")
    for L in [16, 24, 32, 40, 48, 56, 64]:
        frac = title_stats[f"frac_over_{L}"]
        mark = " ← RECOMMENDED" if frac <= 0.01 else ""
        print(f"  L={L:3d}  →  {frac*100:5.2f}% truncated{mark}")
    print()

    # Thumbnail size
    print("=== Thumbnail size audit (sample 500) ===")
    thumb_stats = audit_thumbnails(df)
    print(json.dumps(thumb_stats, indent=2))

    out = {"title": title_stats, "thumbnail": thumb_stats}
    OUT.write_text(json.dumps(out, indent=2))
    print()
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()