"""
Orchestrates the full ingestion run: fetch -> download -> derive features -> upsert to Postgres.

Usage:
    export YOUTUBE_API_KEY="your_key_here"
    python -m pipeline.run_ingestion
"""

import os
import json
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ingestion.youtube_client import get_channel_info, get_video_ids, get_video_details, get_category_name
from ingestion.thumbnail_downloader import download_thumbnail
from ingestion.tubecensus_client import get_subscriber_count_at
from features.title_features import parse_duration_iso8601_to_seconds, title_features
from features.trailing_views import compute_trailing_views

import boto3
from botocore.config import Config

from ingestion.thumbnail_downloader import download_thumbnail, ensure_bucket_exists

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "channels.json")
PUBLISHED_AFTER = "2025-01-01T00:00:00Z"
LABEL_MATURITY_DAYS = 28
SHORTS_MAX_DURATION_SECONDS = 180
OUTPUT_DIR = "data"

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@{POSTGRES_HOST}:{POSTGRES_PORT}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

minio_client = boto3.client(
    "s3",
    endpoint_url=f"http://{os.environ['MINIO_ENDPOINT']}",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

# Deliberately excludes image_embedding/text_embedding -- those are populated
# later by precompute_embeddings.py, not by ingestion.
VIDEO_COLUMNS = [
    "video_id", "channel_id", "channel_ref", "title", "published_at",
    "duration_seconds", "views", "label_finalized", "subscriber_count_at_upload",
    "genre", "thumbnail_path", "title_length_chars", "title_word_count",
    "title_capitalized_word_count", "title_capitalized_letter_count",
    "title_capitalized_letter_ratio", "title_symbol_count",
    "title_has_question_mark", "title_has_number", "trailing_avg_views",
    "is_first_video",
]


def load_channels(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_dataframe(df, engine, table_name="videos", pk_col="video_id", batch_size=500):
    """Insert new rows / update existing ones (matched by pk_col) in one go.
    Only touches columns present in df -- image_embedding/text_embedding aren't
    in VIDEO_COLUMNS, so a re-ingested row never has its embeddings wiped back
    to NULL by this step.
    """
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    update_cols = [c for c in df.columns if c != pk_col]

    # pandas represents a SQL NULL as NaN for numeric columns; psycopg2 needs
    # an actual None there, not float('nan'), or the insert fails.
    clean_df = df.astype(object).where(pd.notnull(df), None)
    records = clean_df.to_dict(orient="records")

    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            stmt = pg_insert(table).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=[pk_col],
                set_={c: stmt.excluded[c] for c in update_cols},
            )
            conn.execute(stmt)


def process_channel(channel_ref, rows, skipped_shorts, finalized_ids, mature_before):
    print(f"Processing {channel_ref}...")
    info = get_channel_info(channel_ref)
    if info is None:
        return
    channel_id, uploads_playlist_id, subscriber_count = info

    video_ids = get_video_ids(uploads_playlist_id, PUBLISHED_AFTER, mature_before)
    already_finalized = [v for v in video_ids if v in finalized_ids]
    video_ids = [v for v in video_ids if v not in finalized_ids]
    print(f"  found {len(video_ids) + len(already_finalized)} videos since {PUBLISHED_AFTER} "
          f"({len(already_finalized)} already finalized, skipping their videos.list fetch)")

    if not video_ids:
        return

    details = get_video_details(video_ids)
    now = datetime.now(timezone.utc)

    for item in details:
        video_id = item["id"]
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        content = item["contentDetails"]

        published_at = snippet["publishedAt"]
        pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        age_days = (now - pub_dt).days

        duration_seconds = parse_duration_iso8601_to_seconds(content["duration"])

        if duration_seconds <= SHORTS_MAX_DURATION_SECONDS:
            skipped_shorts[0] += 1
            continue  # skip Shorts entirely -- no thumbnail download, no row

        views = int(stats.get("viewCount", 0))
        genre = get_category_name(snippet.get("categoryId"))

        thumb_path = download_thumbnail(video_id, snippet["thumbnails"], minio_client)

        sub_count_at_upload = get_subscriber_count_at(
            channel_id, published_at, fallback_count=subscriber_count
        )

        rows.append({
            "video_id": video_id,
            "channel_id": channel_id,
            "channel_ref": channel_ref,
            "title": snippet["title"],
            "published_at": published_at,
            "duration_seconds": duration_seconds,
            "views": views,
            "label_finalized": age_days >= LABEL_MATURITY_DAYS,
            "subscriber_count_at_upload": sub_count_at_upload,
            "genre": genre,
            "thumbnail_path": thumb_path,
            **title_features(snippet["title"]),
        })


def main():
    channels = load_channels()

    existing_df = pd.read_sql(f"SELECT {', '.join(VIDEO_COLUMNS)} FROM videos", engine)
    finalized_ids = set(existing_df.loc[existing_df["label_finalized"] == True, "video_id"])

    mature_before = (datetime.now(timezone.utc) - timedelta(days=LABEL_MATURITY_DAYS)).isoformat()
    rows = []
    skipped_shorts = [0]
    for channel_ref in channels:
        try:
            process_channel(channel_ref, rows, skipped_shorts, finalized_ids, mature_before)
        except Exception as e:
            print(f"  [error] {channel_ref}: {e}")

    df = pd.DataFrame(rows)

    if df.empty:
        print("\nNo new videos ingested this run.")
    else:
        existing = existing_df[~existing_df["video_id"].isin(df["video_id"])]
        df = pd.concat([existing, df], ignore_index=True)
        # Recomputed over the FULL history (existing + new), not just this
        # run's new rows -- fixes the "truncated history on incremental
        # re-runs" limitation the CSV version had (each channel's trailing
        # average now always sees its complete stored history).
        df = compute_trailing_views(df)
        upsert_dataframe(df, engine)
        print(f"\nDone. {len(df)} rows in videos table ({len(rows)} new/updated this run)")

    print(f"Skipped {skipped_shorts[0]} Shorts (<= {SHORTS_MAX_DURATION_SECONDS}s)")


if __name__ == "__main__":
    main()