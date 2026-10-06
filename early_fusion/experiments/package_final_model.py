"""Gabungkan N checkpoint refit menjadi SATU file model akhir (.pt) + file .json pendamping.

Lokasi yang disarankan: early_fusion/experiments/package_final_model.py

Contoh:
  python early_fusion/experiments/package_final_model.py --tag refit_m6_c24_gateoff --seeds 100 101 102 103 104 \
      --out early_fusion/models/final/m6_granular_ensemble_v1.pt

Masukan : early_fusion/models/checkpoints/m6_granular_<tag>_seed<N>.pt   (dibuat run_m6_config.py --save-ckpt)
          early_fusion/results/summary_<tag>.json                         (metrik test refit, untuk dokumentasi)
Keluaran: <out>.pt     satu file: config, scaler, daftar genre, N state_dict, lineage, metrik
          <out>.json   ringkasan terbaca-manusia + SHA-256 file (bahan model card / dokumentasi)

Skrip ini hanya MENGEMAS; tidak melatih dan tidak menyentuh test.
"""
import json
import time
import hashlib
import argparse
from pathlib import Path

import torch

FORMAT_VERSION = 1
MUST_MATCH = ("cfg", "variant", "fit", "n_cont", "n_genres", "genres_train", "split_hashes")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--ckpt-dir", default="early_fusion/models/checkpoints")
    ap.add_argument("--summary", default=None, help="default: early_fusion/results/summary_<tag>.json")
    ap.add_argument("--out", default="early_fusion/models/final/m6_granular_ensemble_v1.pt")
    ap.add_argument("--name", default="m6_granular_ensemble_v1")
    ap.add_argument("--snapshot-hash", default="c14dba895034fc4c")
    args = ap.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir)
    ref, ref_key, members, n_params = None, None, [], []
    for s in args.seeds:
        f = ckpt_dir / f"m6_granular_{args.tag}_seed{s}.pt"
        if not f.exists():
            raise SystemExit(f"checkpoint tidak ditemukan: {f}")
        blob = torch.load(f, map_location="cpu", weights_only=False)
        key = json.dumps({k: blob[k] for k in MUST_MATCH}, sort_keys=True, default=str)
        if ref is None:
            ref, ref_key = blob, key
        elif key != ref_key:
            raise SystemExit(f"checkpoint seed {s} tidak konsisten dengan seed {args.seeds[0]} "
                             f"(config/variant/genre/split berbeda). Menolak mengemas.")
        members.append(dict(seed=int(s), state_dict=blob["state_dict"]))
        n_params.append(int(sum(v.numel() for v in blob["state_dict"].values())))

    summary_path = Path(args.summary) if args.summary else Path(f"early_fusion/results/summary_{args.tag}.json")
    metrics = json.load(open(summary_path)) if summary_path.exists() else {}
    if not metrics:
        print(f"[peringatan] summary tidak ditemukan: {summary_path} (metrik tidak ikut tertanam)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob_out = dict(
        format_version=FORMAT_VERSION,
        name=args.name,
        model_class="RATF_M6_Granular_V2",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        cfg=ref["cfg"], variant=ref["variant"], fit=ref["fit"],
        n_cont=ref["n_cont"], n_genres=ref["n_genres"], genres_train=ref["genres_train"],
        scaler=ref["scaler"],
        split_hashes=ref["split_hashes"], snapshot_hash=args.snapshot_hash, git_sha=ref.get("git_sha"),
        seeds=[m["seed"] for m in members], tag=args.tag,
        refit_metrics=metrics,
        members=members,
    )
    torch.save(blob_out, out)

    digest = sha256_file(out)
    sidecar = dict(
        name=args.name, file=out.name, sha256=digest, size_mb=round(out.stat().st_size / 1e6, 2),
        created_at=blob_out["created_at"], model_class=blob_out["model_class"], format_version=FORMAT_VERSION,
        n_members=len(members), member_seeds=blob_out["seeds"], params_per_member=n_params[0],
        variant=ref["variant"], trained_on=("train+val" if ref["fit"] == "trainval" else "train"),
        cfg=ref["cfg"], split_hashes=ref["split_hashes"], snapshot_hash=args.snapshot_hash,
        git_sha=ref.get("git_sha"), source_tag=args.tag,
        refit_metrics={k: v for k, v in metrics.items() if k.startswith("test_") or k in ("seeds", "best_epochs")},
    )
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"model akhir: {out}   ({sidecar['size_mb']} MB, {len(members)} anggota, "
          f"{n_params[0]:,} parameter/anggota)")
    print(f"sha256     : {digest}")
    print(f"pendamping : {out.with_suffix('.json')}")
    return sidecar


if __name__ == "__main__":
    main()
