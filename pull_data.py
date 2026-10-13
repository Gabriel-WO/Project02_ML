"""
pull_data.py - Step 1 of Project02_ML (K-pop video popularity)

Downloads every upload from a list of YouTube channels (YouTube Data API v3)
and saves them to kpop_videos_raw.csv.

    python pull_data.py --check   # look up each channel: prints its real name and
                                  # video count so you can confirm the handle is right
                                  # (costs about 1 quota unit per channel)
    python pull_data.py           # pull every channel that is not in the CSV yet

It is safe to re-run. Each channel is saved as soon as it finishes, and channels
already in the CSV are skipped, so you can add channels to the lists below and
run again, or just try again tomorrow if you hit the daily quota.

Quota: the API allows 10,000 units/day. Pulling 50 videos costs about 2 units,
so even a channel with 10,000 videos costs roughly 400.

AI assistance: drafted with Claude (document this in the README).
"""
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE = "https://www.googleapis.com/youtube/v3"
OUT = "kpop_videos_raw.csv"

# A channel can be written as an @handle OR as a channel ID (starts with "UC").
# To find one: open the channel on YouTube; the @handle is under its name.

# Channels that belong to ONE artist (the 26 you already pulled). Add more here.
ARTIST_CHANNELS = [
    "@pledis17", "@TXT_bighit", "@LESSERAFIM_official", "@BTS", "@ENHYPENOFFICIAL",
    "@official_i_dle", "@ITZY", "@IVEstarship", "@ATEEZofficial", "@TWICE", "@StrayKids",
    "@aespa", "@NMIXXOfficial", "@ILLIT_official", "@hearts2hearts.official",
    "@MAMAMOO_OFFICIAL", "@MEOVV_OFFICIAL", "@BABYMONSTER", "@cortis_bighit",
    "@KiiiKiii_starship", "@weareoneEXO", "@redvelvet", "@BIGBANG", "@NewJeans_official",
    "@BLACKPINK", "@2NE1",
]

# Label / multi-artist channels. Run `python pull_data.py --check` first to confirm
# each one is the channel you expect (these came from third-party stats sites).
LABEL_CHANNELS = [
    "@HYBELABELS",  # HYBE LABELS
    "@SMTOWN",  # SMTOWN
    "@JYPEntertainment",  # JYP Entertainment
    "@YGEntertainment"         # YG ENTERTAINMENT
]

# Optional cap on how many of a channel's newest videos to pull (saves time on huge channels).
MAX_VIDEOS = {
    # "@1theK": 8000,
}

quota_used = 0


class QuotaExceeded(Exception):
    pass


def api_get(endpoint, **params):
    """One API call with retries. Stops with a clear message on errors."""
    global quota_used
    if not API_KEY:
        sys.exit("No API key found. Put YOUTUBE_API_KEY=... in your .env file.")
    params["key"] = API_KEY
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 403 and "quotaExceeded" in r.text:
            raise QuotaExceeded()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        if not r.ok:
            sys.exit(f"YouTube API error {r.status_code} on '{endpoint}': {r.text[:300]}")
        quota_used += 1
        return r.json()
    sys.exit(f"Giving up on '{endpoint}' after repeated network errors. Check your connection.")


def resolve_channel(ref):
    """Look up a channel by @handle or ID. Returns None if it does not exist."""
    key = "id" if (ref.startswith("UC") and len(ref) == 24) else "forHandle"
    items = api_get("channels", part="snippet,contentDetails,statistics", **{key: ref}).get("items", [])
    if not items:
        return None
    it = items[0]
    return {
        "title": it["snippet"]["title"],
        "uploads": it["contentDetails"]["relatedPlaylists"]["uploads"],
        "video_count": int(it["statistics"].get("videoCount", 0)),
    }


def get_video_ids(playlist_id, limit=None):
    """Page through the uploads playlist (newest first) and collect video IDs."""
    ids, token = [], None
    while True:
        params = dict(part="contentDetails", playlistId=playlist_id, maxResults=50)
        if token:
            params["pageToken"] = token
        data = api_get("playlistItems", **params)
        ids += [i["contentDetails"]["videoId"] for i in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token or (limit and len(ids) >= limit):
            return ids[:limit] if limit else ids


def get_video_details(video_ids, channel_ref):
    """Fetch title, date, duration and stats for videos, 50 at a time."""
    rows = []
    for i in range(0, len(video_ids), 50):
        data = api_get("videos", part="snippet,contentDetails,statistics",
                       id=",".join(video_ids[i:i + 50]))
        for v in data.get("items", []):
            stats = v.get("statistics", {})
            rows.append({
                "channel": channel_ref,
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "published_at": v["snippet"]["publishedAt"],
                "duration": v["contentDetails"]["duration"],
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            })
    return rows


def main():
    channels = ARTIST_CHANNELS + LABEL_CHANNELS

    if "--check" in sys.argv:
        for ref in channels:
            info = resolve_channel(ref)
            if info is None:
                print(f"{ref:28s}  NOT FOUND - check the spelling or use the channel ID")
            else:
                print(f"{ref:28s}  {info['title']:32s} {info['video_count']:>7,} videos")
        print(f"(about {quota_used} quota units used)")
        return

    done = set(pd.read_csv(OUT, usecols=["channel"])["channel"]) if os.path.exists(OUT) else set()
    todo = [c for c in channels if c not in done]
    print(f"{len(done)} channels already saved, {len(todo)} to pull.")

    try:
        for ref in todo:
            info = resolve_channel(ref)
            if info is None:
                print(f"{ref}: NOT FOUND, skipping")
                continue
            ids = get_video_ids(info["uploads"], MAX_VIDEOS.get(ref))
            rows = get_video_details(ids, ref)
            if rows:
                pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{ref:28s} {info['title']:30s} {len(rows):>6,} videos saved "
                  f"(~{quota_used} quota units used so far)")
    except QuotaExceeded:
        print("\nDaily YouTube quota used up. Everything finished so far is saved.\n"
              "The quota resets at midnight Pacific time. Run this script again then; "
              "it will skip the channels already saved and continue.")
        return
    print("\nDone. Pull date:", pd.Timestamp.now().date(), "<- record this in your README.")


if __name__ == "__main__":
    main()