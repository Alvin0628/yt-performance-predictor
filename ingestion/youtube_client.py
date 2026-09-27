"""Thin wrapper around the YouTube Data API v3 endpoints this project needs."""

import os
import time
from datetime import datetime

import requests

API_KEY = os.environ.get("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the YOUTUBE_API_KEY environment variable first.")

BASE_URL = "https://www.googleapis.com/youtube/v3"


def api_get(endpoint, params):
    params["key"] = API_KEY
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params)
    resp.raise_for_status()
    return resp.json()


def get_channel_info(channel_ref):
    """Resolve a handle (@name) or channel ID to (channel_id, uploads_playlist_id, subscriber_count)."""
    if channel_ref.startswith("@"):
        params = {"part": "contentDetails,statistics", "forHandle": channel_ref}
    else:
        params = {"part": "contentDetails,statistics", "id": channel_ref}

    data = api_get("channels", params)
    if not data.get("items"):
        print(f"  [warn] channel not found: {channel_ref}")
        return None

    item = data["items"][0]
    channel_id = item["id"]
    uploads_playlist_id = item["contentDetails"]["relatedPlaylists"]["uploads"]
    subscriber_count = int(item["statistics"].get("subscriberCount", 0))
    return channel_id, uploads_playlist_id, subscriber_count


def get_video_ids(playlist_id, published_after, mature_before):
    """Page through a channel's uploads playlist, returning only videos
    published in [published_after, mature_before] -- i.e. within the
    overall ingestion window AND already at least LABEL_MATURITY_DAYS old.

    Stops paging entirely once a video predates published_after (everything
    further back in this reverse-chronological playlist is out of range).
    A video newer than mature_before is skipped but does NOT stop paging --
    older, already-mature videos may still follow it in the list.
    """
    video_ids = []
    page_token = None
    after_cutoff = datetime.fromisoformat(published_after.replace("Z", "+00:00"))
    mature_cutoff = datetime.fromisoformat(mature_before.replace("Z", "+00:00"))  # <-- added

    while True:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token

        data = api_get("playlistItems", params)
        items = data.get("items", [])
        if not items:
            break

        stop = False
        for item in items:
            published_at = item["contentDetails"].get("videoPublishedAt")
            if published_at is None:
                continue
            pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            if pub_dt < after_cutoff:
                stop = True
                continue
            if pub_dt > mature_cutoff:   # <-- was "mature_before"
                continue
            video_ids.append(item["contentDetails"]["videoId"])

        page_token = data.get("nextPageToken")
        if not page_token or stop:
            break

    return video_ids

def get_video_details(video_ids):
    """Batch-fetch full details for up to 50 video IDs at a time."""
    all_details = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        params = {"part": "snippet,contentDetails,statistics", "id": ",".join(batch)}
        data = api_get("videos", params)
        all_details.extend(data.get("items", []))
        time.sleep(0.1)
    return all_details


_category_cache = {}

def get_category_name(category_id, region_code="US"):
    if category_id in _category_cache:
        return _category_cache[category_id]
    data = api_get("videoCategories", {"part": "snippet", "regionCode": region_code})
    for item in data.get("items", []):
        _category_cache[item["id"]] = item["snippet"]["title"]
    return _category_cache.get(category_id, "Unknown")