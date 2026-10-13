"""
clean_data.py - Step 2 of Project02_ML (K-pop video popularity)

Reads kpop_videos_raw.csv (pulled from the YouTube Data API) and writes
kpop_videos_clean.csv with parsed durations/dates, a video_type label,
engineered features, and the target column.

Design decision: filtering to "official MVs" by title keyword kept only ~390
of ~49,600 videos and was biased toward older groups, because many labels do not
put "MV" in titles (e.g. Stray Kids use '"CHEESE" Video'). So we keep all
videos and classify the type of each one from its title instead.

AI assistance: drafted with Claude (document this in the README).
"""
import re
import numpy as np
import pandas as pd

RAW = "kpop_videos_merged.csv"
OUT = "kpop_videos_clean.csv"
PULL_DATE = pd.Timestamp("2026-10-06", tz="UTC")  # day the API data was pulled

df = pd.read_csv(RAW)
print("raw rows:", len(df))

# 1. Remove duplicate videos.
df = df.drop_duplicates("video_id").copy()

# 2. Parse ISO 8601 duration (PT3M45S, PT1H30M38S) into seconds.
def duration_to_seconds(iso):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return np.nan
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s

df["duration_sec"] = df["duration"].apply(duration_to_seconds)
df = df.dropna(subset=["duration_sec"])

# 3. Dates and video age. Older videos have had longer to collect views.
df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
df["days_since_release"] = (PULL_DATE - df["published_at"]).dt.days.clip(lower=1)

# 4. Drop videos with 0 views (private, blocked, or brand new).
df = df[df["views"] > 0].copy()

# 5. Classify video type from the title (checked in this order; first match wins).
def has(pattern):
    return df["title"].str.contains(pattern, flags=re.IGNORECASE, regex=True)

rules = [
    ("short",            df["duration_sec"] <= 60),
    ("teaser",           has(r"teaser|trailer|spoiler|preview|reveal|highlight|concept\s*(film|photo)|poster")),
    ("lyric_audio",      has(r"lyric|visualizer|official audio|\(audio\)")),
    ("dance_practice",   has(r"dance\s*(practice|video)|choreograph|camerawork|안무")),
    ("behind_the_scenes", has(r"behind|making|sketch|shoot|monitoring|vlog|비하인드|메이킹|\bEP\.?\s*\d|episode|\[\s*EPISODE|selfie|-note\]|T:TIME|vcr")),
    ("performance",      has(r"performance|stage|live|showcase|concert|fancam|special clip|@ ")),
    ("cover",            has(r"\bcover\b|원곡")),
    ("reaction_review",  has(r"reaction|review|react to")),
    ("mv",               has(r"\bM/V\b|\bMV\b|official music video|official mv|[\"“”']\s*video\s*$")),
]
df["video_type"] = np.select([c for _, c in rules], [n for n, _ in rules], default="other")

# 6. Time features (all known at release).
df["release_year"] = df["published_at"].dt.year
df["release_month"] = df["published_at"].dt.month
df["release_dow"] = df["published_at"].dt.dayofweek
df["release_hour_utc"] = df["published_at"].dt.hour

# 7. Title features.
df["title_has_korean"] = df["title"].str.contains(r"[가-힣]").astype(int)
df["title_has_english"] = df["title"].str.contains(r"[A-Za-z]").astype(int)
df["title_length"] = df["title"].str.len()
df["n_hashtags"] = df["title"].str.count("#")

# 8. Targets. Views are extremely skewed, so use log10.
df["log_views"] = np.log10(df["views"])
df["views_per_day"] = df["views"] / df["days_since_release"]
df["log_views_per_day"] = np.log10(df["views_per_day"])

# 9. Channel history using ONLY earlier videos (prevents data leakage).
#    The first video from a channel gets NaN: handle it when modeling.
df = df.sort_values(["channel", "published_at"])
g = df.groupby("channel")["log_views_per_day"]
df["prev_video_count"] = df.groupby("channel").cumcount()
df["prev_avg_log_vpd"] = g.transform(lambda s: s.shift().expanding().mean())

# 10. Likes and comments are results of popularity, not causes: drop them.
df = df.drop(columns=["likes", "comments", "duration"])

df.to_csv(OUT, index=False)
print("clean rows:", len(df))
print(df["video_type"].value_counts())