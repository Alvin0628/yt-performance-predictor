"""Buat bucket MinIO kalau belum ada."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.precompute_embeddings import minio_client, MINIO_BUCKET


def main():
    try:
        minio_client.head_bucket(Bucket=MINIO_BUCKET)
        print(f"bucket '{MINIO_BUCKET}' already exists")
    except Exception:
        minio_client.create_bucket(Bucket=MINIO_BUCKET)
        print(f"bucket '{MINIO_BUCKET}' created")

    resp = minio_client.list_objects_v2(Bucket=MINIO_BUCKET, MaxKeys=1)
    print("verify: OK,", resp.get("KeyCount", 0), "objects")


if __name__ == "__main__":
    main()