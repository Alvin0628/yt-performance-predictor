"""Bring the token store up to date with a snapshot (extract tokens only for new videos).

First use, from the legacy cache built for snapshot c14dba895034fc4c:
    python -m scripts.update_token_store --bootstrap-from data_snapshots/token_cache

Later runs (the snapshot has new videos):
    python -m scripts.update_token_store --snapshot data_snapshots/snapshot.parquet

Needs MINIO_* and POSTGRES_* env vars (the extractor imports models.precompute_embeddings)
unless --dry-run is used.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from early_fusion.datasets.token_store import TokenStore, META_FILE


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="data_snapshots/snapshot.parquet")
    ap.add_argument("--store", default="data_snapshots/token_store")
    ap.add_argument("--bootstrap-from", default=None,
                    help="legacy token cache directory used to create the store if it does not exist")
    ap.add_argument("--create-empty", action="store_true",
                    help="create an empty store if it does not exist (everything gets extracted)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="extract at most this many videos")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be extracted")
    args = ap.parse_args(argv)

    store_dir = Path(args.store)
    if not (store_dir / META_FILE).exists():
        if args.bootstrap_from:
            print(f"creating store {store_dir} from legacy cache {args.bootstrap_from}")
            TokenStore.create_from_legacy(args.bootstrap_from, store_dir)
        elif args.create_empty:
            print(f"creating empty store {store_dir}")
            TokenStore.create_empty(store_dir)
        else:
            sys.exit(f"no token store at {store_dir}: pass --bootstrap-from <legacy cache> or --create-empty")
    store = TokenStore(store_dir)

    df = pd.read_parquet(args.snapshot, columns=["video_id", "title"])
    todo = store.missing(df["video_id"])
    print(f"store: {len(store)} rows | snapshot: {df['video_id'].nunique()} videos | to extract: {len(todo)}")
    if args.dry_run or not todo:
        return 0

    from early_fusion.datasets.token_update import make_extractor, update_store
    extract = make_extractor()
    print(f"extractor: {extract.info}")
    stats = update_store(store, df, extract, batch=args.batch, flush_every=args.flush_every, limit=args.limit)
    print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
