"""Check the 16 channels missing from Postgres: determine whether they have videos
that pass the filters (>= 2025-01-01, non-Shorts, >= 28 days old)."""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from models.precompute_embeddings import engine
from ingestion.youtube_client import (
    get_channel_info,
    get_video_ids,
    get_video_details,
)
from features.title_features import parse_duration_iso8601_to_seconds


MISSING = [
    "@AdamRagusea",
    "@AlexG",
    "@CompanyMan",
    "@Empleman",
    "@Garage54",
    "@Hoog",
    "@HowToMakeEverything",
    "@Kraut",
    "@LessonsFromTheScreenplay",
    "@MaxMiller",
    "@NikoOmilana",
    "@PracticalEngineering",
    "@ProHomeCooks",
    "@TheEngineeringMindset",
    "@TomSka",
    "@jasontheween",
]

PUBLISHED_AFTER = "2025-01-01T00:00:00Z"
LABEL_MATURITY_DAYS = 28
SHORTS_MAX_DURATION_SECONDS = 180


def main():
    mature_before = (
        datetime.now(timezone.utc) - timedelta(days=LABEL_MATURITY_DAYS)
    ).isoformat()

    results = []

    for handle in MISSING:
        try:
            info = get_channel_info(handle)

            if info is None:
                print(f"{handle:<30} -> channel not found / API error")
                results.append({
                    "handle": handle,
                    "status": "channel_not_found",
                    "n_non_shorts": 0,
                })
                continue

            channel_id, uploads_playlist_id, subs = info

            video_ids = get_video_ids(
                uploads_playlist_id,
                PUBLISHED_AFTER,
                mature_before,
            )

            n_total = len(video_ids)

            if n_total == 0:
                print(f"{handle:<30} -> 0 videos since {PUBLISHED_AFTER}")
                results.append({
                    "handle": handle,
                    "status": "no_videos_in_window",
                    "n_total": 0,
                    "n_non_shorts": 0,
                })
                continue

            # Check how many videos are non-Shorts
            details = get_video_details(video_ids)

            n_non_shorts = 0

            for item in details:
                dur = parse_duration_iso8601_to_seconds(
                    item["contentDetails"]["duration"]
                )

                if dur > SHORTS_MAX_DURATION_SECONDS:
                    n_non_shorts += 1

            status = (
                "has_content"
                if n_non_shorts > 0
                else "all_shorts"
            )

            print(
                f"{handle:<30} -> "
                f"total={n_total}, "
                f"non-Shorts={n_non_shorts} "
                f"[{status}]"
            )

            results.append({
                "handle": handle,
                "status": status,
                "n_total": n_total,
                "n_non_shorts": n_non_shorts,
                "channel_id": channel_id,
            })

        except Exception as e:
            print(f"{handle:<30} -> [error] {e}")

            results.append({
                "handle": handle,
                "status": "error",
                "error": str(e),
            })

    # Summary
    print()
    print("=" * 60)

    df = pd.DataFrame(results)

    print(df.to_string(index=False))

    print()

    df.to_csv(
        "data_snapshots/missing_channels_audit.csv",
        index=False,
    )

    print("Saved: data_snapshots/missing_channels_audit.csv")

    # Identify channels that need to be re-run
    need_rerun = [
        r["handle"]
        for r in results
        if r["status"] == "has_content"
    ]

    print()

    if need_rerun:
        print(f"CHANNELS THAT NEED TO BE RE-RUN ({len(need_rerun)}):")

        for handle in need_rerun:
            print(f"  {handle}")
    else:
        print("No channels need to be re-run.")


if __name__ == "__main__":
    main()