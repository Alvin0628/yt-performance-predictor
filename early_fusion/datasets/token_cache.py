"""Read/write fp16 memmaps for image patches and text tokens.

Format:
  img_tokens.fp16   (N, 50, 768)  — image tokens (CLS + 49 patches)
  txt_tokens.fp16   (N, L, 512)   — text tokens
  txt_mask.bool     (N, L)        — 1 = valid token, 0 = padding
  thumb_ok.npy      (N,)          — flag indicating thumbnail loaded successfully
  index.parquet     (N, 3)        — video_id, row, thumb_ok
  meta.json                        — metadata (encoder, L, snapshot hash)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class TokenCacheWriter:
    def __init__(self, out_dir, n, L, img_dim=768, txt_dim=512, img_tokens=50):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = n
        self.L = L
        self.img = np.memmap(
            self.dir / "img_tokens.fp16", dtype=np.float16, mode="w+",
            shape=(n, img_tokens, img_dim),
        )
        self.txt = np.memmap(
            self.dir / "txt_tokens.fp16", dtype=np.float16, mode="w+",
            shape=(n, L, txt_dim),
        )
        self.mask = np.memmap(
            self.dir / "txt_mask.bool", dtype=np.bool_, mode="w+",
            shape=(n, L),
        )
        self.thumb_ok = np.zeros(n, dtype=np.bool_)

    def write(self, i, img_t, txt_t, txt_m, ok):
        self.img[i] = img_t.astype(np.float16)
        self.txt[i] = txt_t.astype(np.float16)
        self.mask[i] = txt_m.astype(np.bool_)
        self.thumb_ok[i] = ok

    def close(self, meta, video_ids):
        self.img.flush()
        self.txt.flush()
        self.mask.flush()
        np.save(self.dir / "thumb_ok.npy", self.thumb_ok)
        pd.DataFrame({
            "video_id": video_ids,
            "row": np.arange(self.n),
            "thumb_ok": self.thumb_ok,
        }).to_parquet(self.dir / "index.parquet", index=False)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_cache(dir_path):
    """Return (img, txt, mask, thumb_ok, index_df, meta)."""
    dir_path = Path(dir_path)
    meta = json.loads((dir_path / "meta.json").read_text())
    n, L = meta["n"], meta["L"]
    img = np.memmap(
        dir_path / "img_tokens.fp16", dtype=np.float16, mode="r",
        shape=(n, meta["img_tokens"], meta["img_dim"]),
    )
    txt = np.memmap(
        dir_path / "txt_tokens.fp16", dtype=np.float16, mode="r",
        shape=(n, L, meta["txt_dim"]),
    )
    mask = np.memmap(
        dir_path / "txt_mask.bool", dtype=np.bool_, mode="r", shape=(n, L),
    )
    thumb_ok = np.load(dir_path / "thumb_ok.npy")
    index_df = pd.read_parquet(dir_path / "index.parquet")
    return img, txt, mask, thumb_ok, index_df, meta