"""
monitor_drift -- reports (never gates) whether recently-matured videos'
tabular feature distributions have shifted from the rest of the finalized
history. Read-only: writes one line to experiments/drift_reports.jsonl.

"Recent" = published_at within --recent-days (no ingested-at timestamp
exists in the schema, so this is a time-based proxy, not "rows this
specific ingest_new run touched"). Deliberately only touches columns
run_ingestion.py itself fills in -- embed_new runs in parallel with this
DAG, not before it, so image_embedding may still be NULL for the newest
rows when this runs.

Usage:
    python -m pipeline.monitor_drift
    python -m pipeline.monitor_drift --recent-days 60
"""

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ks_2samp
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

DRIFT_LOG_PATH = "experiments/drift_reports.jsonl"
MIN_SAMPLE_SIZE = 20
ALPHA = 0.05
NUMERIC_COLS = [
    "duration_seconds", "subscriber_count_at_upload", "title_length_chars",
    "title_word_count", "trailing_avg_views",
]


def load_finalized():
    return pd.read_sql(
        "SELECT channel_ref, published_at, duration_seconds, subscriber_count_at_upload, "
        "title_length_chars, title_word_count, trailing_avg_views "
        "FROM videos WHERE label_finalized = true",
        engine,
    )


def channel_mix_shift(recent, reference):
    recent_mix = recent["channel_ref"].value_counts(normalize=True)
    ref_mix = reference["channel_ref"].value_counts(normalize=True)
    rows = []
    for ch in sorted(set(recent_mix.index) | set(ref_mix.index)):
        r, b = recent_mix.get(ch, 0.0), ref_mix.get(ch, 0.0)
        rows.append({"channel_ref": ch, "recent_share": round(r, 4),
                     "baseline_share": round(b, 4), "abs_diff": round(abs(r - b), 4)})
    return sorted(rows, key=lambda x: -x["abs_diff"])


def _log(report):
    os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
    with open(DRIFT_LOG_PATH, "a") as f:
        f.write(json.dumps(report) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-days", type=int, default=60,
                         help="videos published within this many days count as 'recent'")
    args = parser.parse_args()

    df = load_finalized()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.recent_days)
    recent = df[df["published_at"] >= cutoff]
    reference = df[df["published_at"] < cutoff]
    print(f"{len(df)} finalized rows total -- recent: {len(recent)}, reference: {len(reference)}")

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "recent_days": args.recent_days,
        "n_recent": int(len(recent)),
        "n_reference": int(len(reference)),
    }

    if len(recent) < MIN_SAMPLE_SIZE or len(reference) < MIN_SAMPLE_SIZE:
        report["skipped"] = True
        report["reason"] = f"too few rows for a meaningful test (need >= {MIN_SAMPLE_SIZE} per group)"
        print(f"[drift] SKIPPED -- {report['reason']}")
        _log(report)
        return

    numeric_results, flagged_numeric = {}, []
    for col in NUMERIC_COLS:
        r, b = recent[col].dropna(), reference[col].dropna()
        if len(r) < MIN_SAMPLE_SIZE or len(b) < MIN_SAMPLE_SIZE:
            continue
        stat, p = ks_2samp(r, b)
        numeric_results[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p), 6)}
        if p < ALPHA:
            flagged_numeric.append(col)

    channel_mix = channel_mix_shift(recent, reference)
    flagged_channels = [row["channel_ref"] for row in channel_mix if row["abs_diff"] > 0.15]

    report.update(numeric_results=numeric_results, flagged_numeric_columns=flagged_numeric,
                  channel_mix=channel_mix, flagged_channels=flagged_channels)
    _log(report)

    if flagged_numeric or flagged_channels:
        print(f"[drift] FLAGGED -- numeric: {flagged_numeric or 'none'}, "
              f"channel share shifts >15pp: {flagged_channels or 'none'}")
    else:
        print("[drift] no shift detected past thresholds")


if __name__ == "__main__":
    main()