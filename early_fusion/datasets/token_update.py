"""Bring a TokenStore up to date with a snapshot: extract tokens only for new videos.

`update_store` is pure orchestration and takes the extractor as an argument, so it can be
tested without CLIP. `make_extractor` builds the real one (CLIP + MinIO) and uses exactly the
same code path as scripts/build_token_cache.py (squash transform, image_tokens/text_tokens,
L=32, thumbnails from MinIO as <video_id>.jpg; a failed thumbnail gives zero image tokens and
thumb_ok=False).
"""
from contextlib import contextmanager

import numpy as np


@contextmanager
def numerics(mode="legacy"):
    """Presisi matmul/conv selama ekstraksi, terlepas dari urutan import.

    'legacy'  : sama dengan saat token cache lama dibangun (scripts/build_token_cache.py tidak
                meng-import m6_core): matmul fp32 penuh, cudnn memakai bawaan PyTorch (TF32 aktif).
    'tf32'    : matmul TF32 (yang diaktifkan m6_core untuk training). Hanya untuk perbandingan.
    'ambient' : biarkan flag apa adanya.
    Flag dikembalikan seperti semula saat keluar.

    Kenapa penting: m6_core menyalakan torch.backends.cuda.matmul.allow_tf32 saat di-import. Tanpa
    ini, token yang diekstrak di proses yang sama dengan training berbeda halus (tingkat TF32)
    dari token yang dipakai melatih model.
    """
    if mode == "ambient":
        yield
        return
    import torch
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = (mode == "tf32")
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def update_store(store, df, extract_fn, batch=16, flush_every=128, limit=None, log=print,
                 max_fail_frac=None, fail_abort_min=5):
    """Extract and append tokens for every video_id in df that the store does not have.

    df: needs 'video_id' and 'title', in snapshot order.
    extract_fn(video_ids, titles) -> (img (B,50,768), txt (B,L,512), mask (B,L), ok (B,))
    Rows are appended every `flush_every` videos, so an interrupted run resumes where it stopped.

    max_fail_frac: if set, a group of pending videos in which at least `fail_abort_min` thumbnails failed
    AND more than this fraction failed is NOT appended and a RuntimeError is raised. A failed thumbnail
    is stored as zero tokens for good (it is never retried), so a MinIO outage must stop the run
    instead of being written into the store.
    """
    titles = dict(zip(df["video_id"], df["title"]))
    todo = store.missing(df["video_id"])
    if limit is not None:
        todo = todo[:limit]
    log(f"token store: {len(store)} rows, snapshot has {df['video_id'].nunique()} videos, "
        f"{len(todo)} to extract")

    pend_ids, pend = [], {"img": [], "txt": [], "mask": [], "ok": []}
    stats = {"n_new": 0, "n_thumb_failed_new": 0}

    def flush():
        if not pend_ids:
            return
        n_fail = int((~np.concatenate(pend["ok"])).sum())
        if max_fail_frac is not None and n_fail >= fail_abort_min and n_fail / len(pend_ids) > max_fail_frac:
            raise RuntimeError(
                f"{n_fail} of {len(pend_ids)} thumbnails failed to load (more than {max_fail_frac:.0%}); "
                f"not writing them to the token store. Check MinIO, then rerun (finished rows are kept).")
        store.append(pend_ids, np.concatenate(pend["img"]), np.concatenate(pend["txt"]),
                     np.concatenate(pend["mask"]), np.concatenate(pend["ok"]))
        stats["n_new"] += len(pend_ids)
        stats["n_thumb_failed_new"] += int((~np.concatenate(pend["ok"])).sum())
        pend_ids.clear()
        for v in pend.values():
            v.clear()
        log(f"  appended, store now {len(store)} rows")

    for start in range(0, len(todo), batch):
        ids = todo[start:start + batch]
        tt = [titles[v] if isinstance(titles[v], str) else "" for v in ids]
        img, txt, mask, ok = extract_fn(ids, tt)
        pend_ids.extend(ids)
        pend["img"].append(np.asarray(img))
        pend["txt"].append(np.asarray(txt))
        pend["mask"].append(np.asarray(mask))
        pend["ok"].append(np.asarray(ok, dtype=bool))
        if len(pend_ids) >= flush_every:
            flush()
    flush()
    stats["n_total"] = len(store)
    return stats


def make_extractor(device=None, L=32, numerics_mode="legacy"):
    """The real extractor (needs torch, CLIP weights, and MinIO env vars).

    numerics_mode: see numerics(). The default reproduces how the existing cache was built.
    """
    import io

    import torch
    import transformers
    from PIL import Image

    from early_fusion.datasets.clip_tokens import load_clip, image_tokens, text_tokens
    from models.precompute_embeddings import minio_client, MINIO_BUCKET

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, _encode_fn, transform, tok, _cfg = load_clip(device)

    def fetch(video_id):
        try:
            obj = minio_client.get_object(Bucket=MINIO_BUCKET, Key=f"{video_id}.jpg")
            img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
            return transform(img), True
        except Exception:
            return torch.zeros(3, 224, 224), False

    def extract(video_ids, titles):
        got = [fetch(v) for v in video_ids]
        x = torch.stack([g[0] for g in got]).to(device)
        oks = np.array([g[1] for g in got], dtype=bool)
        with numerics(numerics_mode), torch.inference_mode():
            it = image_tokens(clip, x).cpu().numpy()
            tt, tm, _ = text_tokens(clip, tok, list(titles), device, L)
            tt, tm = tt.cpu().numpy(), tm.cpu().numpy()
        return it, tt, tm, oks

    extract.info = {"device": str(device), "transformers_version": transformers.__version__,
                    "torch_version": torch.__version__, "numerics": numerics_mode}
    return extract
