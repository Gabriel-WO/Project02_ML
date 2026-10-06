import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE = "https://www.googleapis.com/youtube/v3"

def get_uploads_playlist_id(handle):
    """Look up a channel by its @handle and return its uploads playlist ID."""
    r = requests.get(f"{BASE}/channels", params={
        "part": "contentDetails",
        "forHandle": handle,
        "key": API_KEY,
    })
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise ValueError(f"No channel found for {handle}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_video_ids(playlist_id):
    """Page through the uploads playlist and collect every video ID."""
    ids, token = [], None
    while True:
        r = requests.get(f"{BASE}/playlistItems", params={
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
            "pageToken": token,
            "key": API_KEY,
        })
        r.raise_for_status()
        data = r.json()
        ids += [i["contentDetails"]["videoId"] for i in data["items"]]
        token = data.get("nextPageToken")
        if not token:
            return ids


def get_video_details(video_ids, channel_name):
    """Fetch stats for videos in batches of 50."""
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        r = requests.get(f"{BASE}/videos", params={
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(batch),
            "key": API_KEY,
        })
        r.raise_for_status()
        for v in r.json()["items"]:
            rows.append({
                "channel": channel_name,
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "published_at": v["snippet"]["publishedAt"],
                "duration": v["contentDetails"]["duration"],
                "views": int(v["statistics"].get("viewCount", 0)),
                "likes": int(v["statistics"].get("likeCount", 0)),
                "comments": int(v["statistics"].get("commentCount", 0)),
            })
    return rows


channels = ["@BTS", "@BLACKPINK", "BABYMONSTER", "@official_i_dle",  "@IVEstarship",
            "@ILLIT_official", "@aespa", "@hearts2hearts.official", "@ITZY", "@TWICE",
            "@LESSERAFIM_official", "@KiiiKiii_starship", "@MEOVV_OFFICIAL", "@NMIXXOfficial",
            "@pledis17", "@ENHYPENOFFICIAL", "@StrayKids", "@NewJeans_official", "@TXT_bighit",
            "@ATEEZofficial", "@MAMAMOO_OFFICIAL", "@BIGBANG", "@weareoneEXO", "@redvelvet",
            "@2NE1", "@cortis_bighit"]

all_rows = []
for handle in channels:
    playlist = get_uploads_playlist_id(handle)
    ids = get_video_ids(playlist)
    all_rows += get_video_details(ids, handle)
    print(f"{handle}: {len(ids)} videos")

df = pd.DataFrame(all_rows)
df.to_csv("kpop_videos_raw.csv", index=False)
print(df.head())
