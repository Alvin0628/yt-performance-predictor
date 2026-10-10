"""Berkas bundle yang dilayani/dirujuk di models/bundles (Fase 6). Tanpa torch.

    m6_latest.pt/.json   bundle live: yang dibandingkan gerbang dan (nanti) yang dimuat API
    m6_prev.pt/.json     bundle live sebelumnya (rollback: tukar nama)
    m6_anchor.pt/.json   model referensi tetap (v1); tidak pernah ditimpa oleh promosi

Setiap .json membawa sha256 bundle-nya, train_end, dan snapshot_n_total. Gerbang membaca train_end
dan snapshot_n_total dari sana (bundle v1 lama tidak menyimpannya), dan menolak berjalan kalau sha256
di .json tidak cocok dengan berkasnya (salah satu dari keduanya setengah tertulis atau diubah).

Promosi menyalin (bukan memindahkan) live lama ke prev, lalu mengganti latest secara atomik per berkas.
Kalau proses mati di antaranya, latest tetap bundle lama yang utuh.

    python -m early_fusion.live_bundle bootstrap --src early_fusion/models/final/m6_granular_ensemble_v1.pt \\
        --train-end "2026-07-04 22:15:09+00:00" --snapshot-n-total 11285
    python -m early_fusion.live_bundle status
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_fusion.promotion import sha256_file

LATEST, PREV, ANCHOR = "m6_latest", "m6_prev", "m6_anchor"
DEFAULT_BUNDLES = Path("models/bundles")


def sidecar_path(pt):
    return Path(pt).with_suffix(".json")


def read_sidecar(pt):
    p = sidecar_path(pt)
    return json.loads(p.read_text()) if p.exists() else None


def snapshot_n_total(side):
    """Jumlah baris snapshot pelatihan dari sidecar: bundle hasil bootstrap menyimpan `snapshot_n_total`,
    bundle hasil retrain.py menyimpan `n_rows.total`. None bila tidak ada keduanya."""
    if not side:
        return None
    n = side.get("snapshot_n_total")
    if n is None:
        n = (side.get("n_rows") or {}).get("total")
    return None if n is None else int(n)


def verify_sidecar(pt):
    """ValueError bila .json mencatat sha256 yang berbeda dari berkas .pt-nya. Tidak ada .json = lolos."""
    side = read_sidecar(pt)
    if side and side.get("sha256") and side["sha256"] != sha256_file(pt):
        raise ValueError(f"{pt}: sha256 berkas tidak cocok dengan {sidecar_path(pt).name} "
                         f"(bundle atau metadata setengah tertulis/diubah)")
    return side


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_copy(src, dst):
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".new")
    shutil.copyfile(src, tmp)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    return tmp


def _write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".new")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, default=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bootstrap(src_pt, bundles_dir=DEFAULT_BUNDLES, *, train_end, snapshot_n_total, src_json=None, force=False):
    """Buat m6_latest.* dan m6_anchor.* dari bundle yang sedang live (v1). Sumber tidak diubah."""
    src_pt = Path(src_pt)
    bundles_dir = Path(bundles_dir)
    if not src_pt.exists():
        raise FileNotFoundError(src_pt)
    src_sha = sha256_file(src_pt)
    side = json.loads(Path(src_json).read_text()) if src_json else (read_sidecar(src_pt) or {})
    if side.get("sha256") and side["sha256"] != src_sha:
        raise ValueError(f"sha256 {src_pt.name} ({src_sha}) tidak cocok dengan sidecar ({side['sha256']})")
    bundles_dir.mkdir(parents=True, exist_ok=True)
    meta = {**side, "sha256": src_sha, "train_end": str(train_end), "snapshot_n_total": int(snapshot_n_total),
            "bootstrapped_from": src_pt.name, "bootstrapped_at": _now()}
    out = {}
    for name, role in ((LATEST, "live"), (ANCHOR, "anchor")):
        dst = bundles_dir / f"{name}.pt"
        if dst.exists():
            if sha256_file(dst) == src_sha:
                out[name] = "sudah ada (identik)"
                if not sidecar_path(dst).exists():
                    _write_json_atomic(sidecar_path(dst), {**meta, "role": role})
                continue
            if not force:
                raise FileExistsError(f"{dst} sudah ada dan berbeda dari sumber; --force untuk menimpa")
        tmp = _atomic_copy(src_pt, dst)
        if sha256_file(tmp) != src_sha:
            tmp.unlink()
            raise IOError(f"salinan {dst.name} rusak (sha256 berbeda)")
        os.replace(tmp, dst)
        _write_json_atomic(sidecar_path(dst), {**meta, "role": role})
        out[name] = "dibuat"
    return out


def promote(candidate_pt, bundles_dir=DEFAULT_BUNDLES):
    """Jadikan kandidat bundle live: latest lama disalin ke prev, latest diganti kandidat."""
    candidate_pt = Path(candidate_pt)
    bundles_dir = Path(bundles_dir)
    cand_json = sidecar_path(candidate_pt)
    if not candidate_pt.exists() or not cand_json.exists():
        raise FileNotFoundError(f"kandidat/sidecar tidak ada: {candidate_pt}")
    side = json.loads(cand_json.read_text())
    cand_sha = sha256_file(candidate_pt)
    if side.get("sha256") != cand_sha:
        raise ValueError(f"sha256 kandidat tidak cocok dengan sidecar-nya: {candidate_pt}")
    bundles_dir.mkdir(parents=True, exist_ok=True)
    latest, prev = bundles_dir / f"{LATEST}.pt", bundles_dir / f"{PREV}.pt"

    staged = _atomic_copy(candidate_pt, latest)                   # 1. siapkan salinan baru, jangan sentuh yang lama
    try:
        if sha256_file(staged) != cand_sha:
            raise IOError("salinan kandidat rusak (sha256 berbeda)")
        had_prev = latest.exists()
        if had_prev:                                              # 2. latest lama -> prev (disalin, latest tetap utuh)
            verify_sidecar(latest)
            tmp_prev = _atomic_copy(latest, prev)
            os.replace(tmp_prev, prev)
            if sidecar_path(latest).exists():
                _write_json_atomic(sidecar_path(prev), {**json.loads(sidecar_path(latest).read_text()), "role": "prev"})
        os.replace(staged, latest)                                # 3. ganti latest (atomik)
        live_side = {**side, "sha256": cand_sha, "role": "live", "promoted_at": _now()}
        if snapshot_n_total(side) is not None:                    # supaya --min-new-rows tetap bekerja setelah promosi
            live_side["snapshot_n_total"] = snapshot_n_total(side)
        _write_json_atomic(sidecar_path(latest), live_side)
    except Exception:
        if staged.exists():
            staged.unlink()
        raise
    return {"latest": str(latest), "prev": str(prev) if had_prev else None, "sha256": cand_sha}


def status(bundles_dir=DEFAULT_BUNDLES):
    out = {}
    for name in (LATEST, PREV, ANCHOR):
        pt = Path(bundles_dir) / f"{name}.pt"
        if not pt.exists():
            out[name] = None
            continue
        side = read_sidecar(pt) or {}
        sha = sha256_file(pt)
        out[name] = {"sha256": sha, "sidecar_ok": (not side.get("sha256")) or side["sha256"] == sha,
                     "train_end": side.get("train_end"), "snapshot_n_total": snapshot_n_total(side),
                     "role": side.get("role")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m early_fusion.live_bundle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--src", required=True, help="bundle live saat ini (v1), hanya dibaca")
    b.add_argument("--src-json", default=None)
    b.add_argument("--train-end", required=True, help="waktu publikasi terbaru di train+val bundle itu")
    b.add_argument("--snapshot-n-total", type=int, required=True, help="jumlah baris snapshot pelatihannya")
    b.add_argument("--bundles-dir", default=str(DEFAULT_BUNDLES))
    b.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--bundles-dir", default=str(DEFAULT_BUNDLES))
    args = ap.parse_args(argv)
    if args.cmd == "bootstrap":
        res = bootstrap(args.src, args.bundles_dir, train_end=args.train_end,
                        snapshot_n_total=args.snapshot_n_total, src_json=args.src_json, force=args.force)
    else:
        res = status(args.bundles_dir)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
