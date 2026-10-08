"""Persistent token store keyed by video_id.

Same on-disk layout as the legacy token cache (token_cache.py), so `load_cache()` can still
read it:

  img_tokens.fp16   (n, 50, 768)  float16   CLS + 49 patch tokens
  txt_tokens.fp16   (n, L, 512)   float16
  txt_mask.bool     (n, L)        bool      1 = valid token
  thumb_ok.npy      (n,)          bool
  index.parquet     video_id, row, thumb_ok
  meta.json         n, L, dims, encoder, image_mode, ... (n is the commit point)

Differences from the legacy cache:
  * rows are looked up by video_id (`take`), so the snapshot may be a superset or a
    reordering of what is stored;
  * new videos can be appended (`append`) without rewriting the old rows;
  * the store is not tied to one snapshot, so meta.json has no snapshot_hash.

Crash safety: `append` writes the data files first and updates meta.json (with the new n)
last, through an atomic replace. Rows beyond meta["n"] are ignored by readers and cut off
by the next append. Writers are serialised with a lock file.

Only numpy/pandas are needed (no torch).
"""
import json
import mmap
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

IMG_FILE = "img_tokens.fp16"
TXT_FILE = "txt_tokens.fp16"
MASK_FILE = "txt_mask.bool"
THUMB_FILE = "thumb_ok.npy"
INDEX_FILE = "index.parquet"
META_FILE = "meta.json"
LOCK_FILE = ".lock"


class StoreLockedError(RuntimeError):
    pass


def _atomic_write(path, write_fn):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_json(path, obj):
    _atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def _write_npy(path, arr):
    _atomic_write(path, lambda f: np.save(f, arr))


def _write_index(path, video_ids, thumb_ok):
    df = pd.DataFrame({
        "video_id": list(video_ids),
        "row": np.arange(len(video_ids), dtype=np.int64),
        "thumb_ok": np.asarray(thumb_ok, dtype=bool),
    })
    _atomic_write(path, lambda f: df.to_parquet(f, index=False))


class TokenStore:
    def __init__(self, directory):
        self.dir = Path(directory)
        if not (self.dir / META_FILE).exists():
            raise FileNotFoundError(f"no token store at {self.dir} (meta.json missing)")
        self._load()

    # ------------------------------------------------------------------ creation
    @classmethod
    def create_empty(cls, directory, L=32, img_tokens=50, img_dim=768, txt_dim=512,
                     encoder="clip_b32", image_mode="squash", extra_meta=None):
        d = Path(directory)
        if (d / META_FILE).exists():
            raise FileExistsError(f"token store already exists at {d}")
        d.mkdir(parents=True, exist_ok=True)
        for name in (IMG_FILE, TXT_FILE, MASK_FILE):
            (d / name).write_bytes(b"")
        meta = {
            "kind": "store", "n": 0, "L": L, "img_tokens": img_tokens, "img_dim": img_dim,
            "txt_dim": txt_dim, "encoder": encoder, "image_mode": image_mode,
            "n_thumb_ok": 0, "n_thumb_failed": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta.update(extra_meta or {})
        _write_npy(d / THUMB_FILE, np.zeros(0, dtype=bool))
        _write_index(d / INDEX_FILE, [], [])
        _write_json(d / META_FILE, meta)
        return cls(d)

    @classmethod
    def create_from_legacy(cls, src_dir, dst_dir, keep_first=None):
        """Copy a legacy token cache into a new store (the legacy cache is not modified).

        keep_first=k keeps only the first k rows (used by tests to simulate 'new' videos).
        """
        src, dst = Path(src_dir), Path(dst_dir)
        meta = json.loads((src / "meta.json").read_text())
        n_src = int(meta["n"])
        k = n_src if keep_first is None else int(keep_first)
        if not 0 <= k <= n_src:
            raise ValueError(f"keep_first={k} outside 0..{n_src}")
        if (dst / META_FILE).exists():
            raise FileExistsError(f"token store already exists at {dst}")
        dst.mkdir(parents=True, exist_ok=True)

        L, it, idim, tdim = int(meta["L"]), int(meta["img_tokens"]), int(meta["img_dim"]), int(meta["txt_dim"])
        for name, row_bytes in ((IMG_FILE, it * idim * 2), (TXT_FILE, L * tdim * 2), (MASK_FILE, L)):
            want = n_src * row_bytes
            have = (src / name).stat().st_size
            if have != want:
                raise ValueError(f"{src / name}: {have} bytes, expected {want} for n={n_src}")
            with open(src / name, "rb") as fin, open(dst / name, "wb") as fout:
                remaining = k * row_bytes
                while remaining > 0:
                    chunk = fin.read(min(remaining, 64 * 1024 * 1024))
                    if not chunk:
                        raise ValueError(f"unexpected end of {src / name}")
                    fout.write(chunk)
                    remaining -= len(chunk)

        index = pd.read_parquet(src / "index.parquet")
        thumb_ok = np.load(src / "thumb_ok.npy")
        ids = index["video_id"].tolist()[:k]
        ok = thumb_ok[:k]

        new_meta = {kk: vv for kk, vv in meta.items() if kk not in ("snapshot_hash", "snapshot_n_total")}
        new_meta.update({
            "kind": "store", "n": k,
            "n_thumb_ok": int(ok.sum()), "n_thumb_failed": int(k - ok.sum()),
            "provenance": {
                "bootstrapped_from": str(src),
                "legacy_snapshot_hash": meta.get("snapshot_hash"),
                "legacy_n": n_src, "kept_first": k,
                "bootstrapped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        })
        _write_npy(dst / THUMB_FILE, ok)
        _write_index(dst / INDEX_FILE, ids, ok)
        _write_json(dst / META_FILE, new_meta)    # commit point
        return cls(dst)

    # ------------------------------------------------------------------ reading
    def _row_bytes(self):
        return (self.img_tokens * self.img_dim * 2, self.L * self.txt_dim * 2, self.L)

    def _load(self):
        self.meta = json.loads((self.dir / META_FILE).read_text())
        m = self.meta
        self.n = int(m["n"])
        self.L = int(m["L"])
        self.img_tokens = int(m["img_tokens"])
        self.img_dim = int(m["img_dim"])
        self.txt_dim = int(m["txt_dim"])
        index = pd.read_parquet(self.dir / INDEX_FILE)
        thumb_ok = np.load(self.dir / THUMB_FILE)
        if len(index) != self.n or thumb_ok.shape != (self.n,):
            raise ValueError(f"inconsistent store {self.dir}: meta n={self.n}, index={len(index)}, thumb_ok={thumb_ok.shape}")
        if not (index["row"].values == np.arange(self.n)).all():
            raise ValueError(f"index rows are not 0..n-1 in {self.dir}")
        ids = index["video_id"].tolist()
        self._pos = {v: i for i, v in enumerate(ids)}
        if len(self._pos) != self.n:
            raise ValueError(f"duplicate video_id in {self.dir}")
        self.video_ids = ids
        self.thumb_ok = thumb_ok.astype(bool)
        for name, rb in zip((IMG_FILE, TXT_FILE, MASK_FILE), self._row_bytes()):
            size = (self.dir / name).stat().st_size
            if size < self.n * rb:
                raise ValueError(f"{self.dir / name} is too short: {size} < {self.n * rb} bytes")

    def __len__(self):
        return self.n

    def __contains__(self, video_id):
        return video_id in self._pos

    def missing(self, video_ids):
        """Unique video_ids not in the store, in the order given."""
        seen, out = set(), []
        for v in video_ids:
            if v not in self._pos and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _read_rows(self, name, dtype, tail, rows):
        """Copy the requested rows out of one data file.

        The file mapping is closed before returning, so no handle outlives the call (Windows
        cannot delete or truncate a file that is still mapped).
        """
        rows = np.asarray(rows, dtype=np.int64)
        if self.n == 0 or len(rows) == 0:
            return np.zeros((len(rows),) + tail, dtype=dtype)
        count = self.n * int(np.prod(tail))
        with open(self.dir / name, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                view = np.frombuffer(mm, dtype=dtype, count=count).reshape((self.n,) + tail)
                out = view[rows]            # fancy indexing = a copy, independent of the mapping
                del view                    # release the buffer before the mapping is closed
        return out

    def take(self, video_ids):
        """Arrays aligned to `video_ids` (same order, duplicates allowed)."""
        ids = list(video_ids)
        absent = [v for v in ids if v not in self._pos]
        if absent:
            raise KeyError(f"{len(absent)} video_id(s) not in the token store, e.g. {absent[:5]}")
        rows = np.fromiter((self._pos[v] for v in ids), dtype=np.int64, count=len(ids))
        img = self._read_rows(IMG_FILE, np.float16, (self.img_tokens, self.img_dim), rows)
        txt = self._read_rows(TXT_FILE, np.float16, (self.L, self.txt_dim), rows)
        mask = self._read_rows(MASK_FILE, np.bool_, (self.L,), rows)
        return {"img": img, "txt": txt, "mask": mask, "thumb_ok": self.thumb_ok[rows]}

    # ------------------------------------------------------------------ writing
    @contextmanager
    def _lock(self):
        path = self.dir / LOCK_FILE
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise StoreLockedError(
                f"{path} exists: another writer is running, or one crashed. "
                f"If you are sure none is running, delete that file.")
        try:
            os.write(fd, f"pid={os.getpid()} at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}".encode())
            os.close(fd)
            yield
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def append(self, video_ids, img, txt, mask, thumb_ok):
        """Append rows. Returns the number of rows added."""
        ids = list(video_ids)
        b = len(ids)
        if b == 0:
            return 0
        img = np.ascontiguousarray(img, dtype=np.float16)
        txt = np.ascontiguousarray(txt, dtype=np.float16)
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        ok = np.asarray(thumb_ok, dtype=bool)
        want = {"img": (b, self.img_tokens, self.img_dim), "txt": (b, self.L, self.txt_dim),
                "mask": (b, self.L), "thumb_ok": (b,)}
        got = {"img": img.shape, "txt": txt.shape, "mask": mask.shape, "thumb_ok": ok.shape}
        if got != want:
            raise ValueError(f"shape mismatch: got {got}, expected {want}")
        if len(set(ids)) != b:
            raise ValueError("duplicate video_id inside the batch")

        with self._lock():
            self._load()                                    # fresh state, now that we hold the lock
            dup = [v for v in ids if v in self._pos]
            if dup:
                raise ValueError(f"{len(dup)} video_id(s) already in the store, e.g. {dup[:5]}")

            # cut off bytes left by an interrupted earlier append
            for name, rb in zip((IMG_FILE, TXT_FILE, MASK_FILE), self._row_bytes()):
                with open(self.dir / name, "r+b") as f:
                    f.truncate(self.n * rb)
            for name, arr in ((IMG_FILE, img), (TXT_FILE, txt), (MASK_FILE, mask)):
                with open(self.dir / name, "ab") as f:
                    f.write(arr.tobytes())
                    f.flush()
                    os.fsync(f.fileno())

            all_ids = self.video_ids + ids
            all_ok = np.concatenate([self.thumb_ok, ok])
            _write_npy(self.dir / THUMB_FILE, all_ok)
            _write_index(self.dir / INDEX_FILE, all_ids, all_ok)
            meta = dict(self.meta)
            meta.update({
                "n": len(all_ids), "n_thumb_ok": int(all_ok.sum()),
                "n_thumb_failed": int(len(all_ids) - all_ok.sum()),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            _write_json(self.dir / META_FILE, meta)         # commit point
            self._load()
        return b
