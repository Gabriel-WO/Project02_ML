"""
add_artists.py - Step 1b of Project02_ML (K-pop video popularity)

Label channels (HYBE LABELS, SMTOWN, JYP, ...) hold videos for many artists, and
the same music video is often published on BOTH the label channel and the
artist's own channel. This script turns the raw pull into one table where:

  * the `channel` column now means THE ARTIST (so every artist's videos are
    grouped together no matter which YouTube channel they came from),
  * `source_channel` keeps the real YouTube channel,
  * a video published on two channels is kept once (the copy with more views).

Reads kpop_videos_raw.csv, writes kpop_videos_merged.csv, artist_report.csv and
artist_candidates.csv (names that were NOT added, for you to review).
Then point clean_data.py at the merged file (RAW = "kpop_videos_merged.csv").

    python add_artists.py

Artist names on label channels are read from video titles, and label channels also
host shows, competitions and documentaries, so ONLY artists on the approved list
(KNOWN_ARTISTS below, plus your dedicated channels) are kept. Look through
artist_candidates.csv, add any real artists to KNOWN_ARTISTS, and re-run. Fix split
or misspelled names (e.g. "NCT 127" vs "NCT127") with a line in ALIASES.

AI assistance: drafted with Claude.
"""
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

RAW, OUT, REPORT = "kpop_videos_raw.csv", "kpop_videos_merged.csv", "artist_report.csv"
CANDIDATES = "artist_candidates.csv"

# 1. Channels that belong to one artist: every video is that artist's.
ARTIST_CHANNEL_NAMES = {
    "@pledis17": "SEVENTEEN", "@TXT_bighit": "TXT", "@LESSERAFIM_official": "LE SSERAFIM",
    "@BTS": "BTS", "@ENHYPENOFFICIAL": "ENHYPEN", "@official_i_dle": "(G)I-DLE",
    "@ITZY": "ITZY", "@IVEstarship": "IVE", "@ATEEZofficial": "ATEEZ", "@TWICE": "TWICE",
    "@StrayKids": "Stray Kids", "@aespa": "aespa", "NMIXXOfficial": "NMIXX",
    "@ILLIT_official": "ILLIT", "@hearts2hearts.official": "Hearts2Hearts",
    "@MAMAMOO_OFFICIAL": "MAMAMOO", "@MEOVV_OFFICIAL": "MEOVV", "BABYMONSTER": "BABYMONSTER",
    "@cortis_bighit": "CORTIS", "@KiiiKiii_starship": "KiiiKiii", "@weareoneEXO": "EXO",
    "@redvelvet": "Red Velvet", "@BIGBANG": "BIGBANG", "@NewJeans_official": "NewJeans",
    "@BLACKPINK": "BLACKPINK", "@2NE1": "2NE1",
}

# 2. Label channels: the artist is read from each video's title.
LABEL_CHANNELS = {
    "@HYBELABELS": "HYBE LABELS",
    "@SMTOWN": "SMTOWN",
    "@JYPEntertainment": "JYP Entertainment",
    "@YGEntertainment": "YG ENTERTAINMENT"
}

# Artists to keep when they are found on a LABEL channel. Anything not on this list
# (plus the artists above) is NOT added to your data: label channels also host
# competitions, variety shows, documentaries and vlogs whose titles look just like
# "NAME 'Song'", so a name is only trusted if you approve it. Names that are not
# approved are written to artist_candidates.csv with example titles so you can
# review them; to add one, put its name here and re-run. Spelling does not need to
# be exact (case, spaces and punctuation are ignored). This starter list is from
# memory of the labels' rosters, so add or delete freely.
KNOWN_ARTISTS = [
    # HYBE
    "BTS", "SEVENTEEN", "TXT", "ENHYPEN", "LE SSERAFIM", "NewJeans", "fromis_9", "ILLIT",
    "BOYNEXTDOOR", "&TEAM", "KATSEYE", "CORTIS", "tripleS", "Jung Kook", "Jimin", "V", "RM",
    "Jin", "SUGA", "Agust D", "j-hope", "TWS",
    # SM
    "EXO", "NCT 127", "NCT DREAM", "NCT WISH", "NCT U", "WayV", "aespa", "Red Velvet",
    "Girls' Generation", "SHINee", "Super Junior", "TVXQ", "f(x)", "RIIZE", "Hearts2Hearts",
    "BoA", "TAEYEON", "KAI", "BAEKHYUN", "XIUMIN", "CHEN", "SEULGI", "IRENE", "WENDY", "JOY",
    # JYP
    "TWICE", "Stray Kids", "ITZY", "NMIXX", "DAY6", "2PM", "GOT7", "Wonder Girls",
    "Xdinary Heroes", "VCHA", "NEXZ", "KickFlip", "Baby DONT Cry",
    # YG
    "BLACKPINK", "BIGBANG", "2NE1", "WINNER", "iKON", "TREASURE", "BABYMONSTER", "AKMU",
    "PSY", "JENNIE", "ROSE", "LISA", "JISOO", "G-DRAGON", "TAEYANG", "CL",
    # Starship / others
    "IVE", "MONSTA X", "CRAVITY", "WJSN", "KiiiKiii", "MAMAMOO", "(G)I-DLE", "ATEEZ",
    "ZEROBASEONE", "Dreamcatcher", "Kep1er", "LOONA", "ONEUS", "PENTAGON", "BTOB",
]

# Unapproved names with at least this many videos are listed for review.
MIN_CANDIDATE_VIDEOS = 3

# 3. Spelling fixes. Left side: the name with only letters/digits, lower case.
#    Right side: the name you want to see. Add lines as you spot problems in the report.
ALIASES = {
    "tomorrowxtogether": "TXT", "txt": "TXT", "idle": "(G)I-DLE", "gidle": "(G)I-DLE",
    "lesserafim": "LE SSERAFIM", "straykids": "Stray Kids", "skz": "Stray Kids",
    "snsd": "Girls' Generation", "girlsgeneration": "Girls' Generation",
    "nct127": "NCT 127", "nct127127": "NCT 127", "nctdream": "NCT DREAM",
    "nctwish": "NCT WISH", "nctu": "NCT U", "wayv": "WayV", "shinee": "SHINee",
    "superjunior": "Super Junior", "tvxq": "TVXQ", "riize": "RIIZE", "exo": "EXO",
    "boynextdoor": "BOYNEXTDOOR", "fromis9": "fromis_9", "team": "&TEAM", "day6": "DAY6",
    "got7": "GOT7", "2pm": "2PM", "ikon": "iKON", "akmu": "AKMU", "winner": "WINNER",
    "zerobaseone": "ZEROBASEONE", "zb1": "ZEROBASEONE", "xdinaryheroes": "Xdinary Heroes",
    "newjeans": "NewJeans", "hearts2hearts": "Hearts2Hearts",
}
for _name in ARTIST_CHANNEL_NAMES.values():
    ALIASES.setdefault(re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", _name).lower()), _name)


def norm_key(name):
    """'(G)I-DLE' -> 'gidle', 'ROSE' -> 'rose': letters and digits only, lower case."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------- artist from title
HANGUL = re.compile(r"[ᄀ-ᇿㄱ-ㆎ가-힣]")
LEAD_TAGS = re.compile(r"^\s*(\[[^\]]*\]|【[^】]*】)\s*")
QUOTES = "'‘’“”\"「『"
NOT_A_NAME = re.compile(r"\b(ep\.?\s*\d*|episode|official|teaser|making|behind|vlog|reaction|"
                        r"cover|challenge|shorts?|mv|m/v|ver\.?|live|stage|special)\b", re.I)


def extract_artist(title):
    """Best guess at the artist name at the start of a title, or None.
    Handles  ARTIST 'Song'  /  ARTIST (hangul) 'Song'  /  [MV] hangul (ARTIST) - Song."""
    t = title.strip()
    while LEAD_TAGS.match(t):
        t = LEAD_TAGS.sub("", t, count=1)
    cuts = [i for i in (t.find(q) for q in QUOTES) if i > 0]
    if cuts:
        cut = min(cuts)
    else:
        m = re.search(r"\s[-–—|:]\s", t)
        if not m:
            return None
        cut = m.start()
    name = HANGUL.sub("", t[:cut])
    while re.search(r"\(\s*\)", name):
        name = re.sub(r"\(\s*\)", "", name)
    name = re.sub(r"\s+", " ", name).strip(" -–—|:·,/")
    if name.startswith("(") and name.endswith(")") and name.count("(") == name.count(")"):
        name = name[1:-1].strip()
    if not (2 <= len(name) <= 30) or NOT_A_NAME.search(name) or not re.search(r"[A-Za-z0-9]", name):
        return None
    return name


def canonical(name):
    """Map an extracted name to its preferred spelling."""
    return ALIASES.get(norm_key(name), name)


def guess_artist(title):
    """Artist name for a title in its preferred spelling, or None."""
    name = extract_artist(title)
    return canonical(name) if name else None


MV_MARKER = re.compile(r"\b(m/v|mv|official music video)\b", re.I)
SERIES_MARKER = re.compile(r"\b(ep\.?\s*\d+|episode|season|trailer|audition|documentary|part\s*\d+|vlog|"
                           r"behind|making|reaction|interview|variety|show|talk|radio)\b", re.I)


def approved_names():
    """{letters-and-digits key: preferred spelling} for every artist we keep."""
    names = list(ARTIST_CHANNEL_NAMES.values()) + KNOWN_ARTISTS
    return {norm_key(canonical(n)): canonical(n) for n in names}


def candidate_table(lab):
    """Names read from label-channel titles that are NOT approved, with a guess at whether each
    is an artist (many MV-style titles, few episode/trailer-style titles) or a show."""
    rows = []
    for name, g in lab[lab["guess"].notna() & lab["artist"].isna()].groupby("guess"):
        if len(g) < MIN_CANDIDATE_VIDEOS:
            continue
        mv_like = int(g["title"].str.contains(MV_MARKER).sum())
        series_share = float(g["title"].str.contains(SERIES_MARKER).mean())
        rows.append({
            "name": name, "videos": len(g), "mv_style_titles": mv_like,
            "series_style_share": round(series_share, 2),
            "looks_like": "artist?" if mv_like >= 3 and series_share < 0.3 else "show/series?",
            "example_titles": " || ".join(g["title"].head(3)),
        })
    cols = ["name", "videos", "mv_style_titles", "series_style_share", "looks_like", "example_titles"]
    return pd.DataFrame(rows, columns=cols).sort_values(["looks_like", "mv_style_titles"], ascending=[True, False])


# ---------------------------------------------------------------- duplicates
def duration_seconds(iso):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(iso))
    if not m:
        return np.nan
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def title_key(t, keep_hangul=True):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"\b(m/v|mv|official|music video|video)\b", " ", t)
    return re.sub(r"[^0-9a-z" + ("\u3131-\u318e\uac00-\ud7a3" if keep_hangul else "") + "]+", "", t)


def find_duplicates(df):
    """Find copies of the same video published on two different YouTube channels:
    published within 3 days, durations within 3 s, near-identical titles. This is
    checked across ALL channels, so a member's video on a label channel still matches
    the group's own copy. At least one side must be a label channel. Returns ({row label of the copy to drop}, {kept label: artist}),
    where the second dict hands the kept copy the artist of the dropped one when the kept
    copy has none (e.g. the label channel's copy had more views but its title was unreadable)."""
    df = df.sort_values("published_at")
    secs = (df["published_at"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    dur = df["duration"].map(duration_seconds).to_numpy()
    full = df["title"].map(title_key).to_numpy()
    latin = df["title"].map(lambda t: title_key(t, keep_hangul=False)).to_numpy()
    views = df["views"].to_numpy()
    src = df["source_channel"].to_numpy()
    labels = df.index.to_numpy()
    artist = df["artist"].to_numpy(dtype=object)
    own = df["source_channel"].isin(ARTIST_CHANNEL_NAMES).to_numpy()
    nums = df["title"].map(lambda t: re.findall(r"\d+", t)).to_numpy()
    # 1) every plausible pair with a similarity score
    pairs = []
    for i in range(len(df)):
        hi = np.searchsorted(secs, secs[i] + 3 * 86400, side="right")
        if hi <= i + 1:
            continue
        cand = np.arange(i + 1, hi)
        cand = cand[(src[cand] != src[i]) & (np.abs(dur[cand] - dur[i]) <= 3)]
        if own[i]:   # two artists' own channels are never copies of each other: one side must be a label channel
            cand = cand[~own[cand]]
        for j in cand:
            if nums[i] != nums[j]:      # "Teaser 1" vs "Teaser 2", "EP.3" vs "EP.4": different videos
                continue
            score = SequenceMatcher(None, full[i], full[j]).ratio()
            if latin[i] and latin[j]:
                score = max(score, SequenceMatcher(None, latin[i], latin[j]).ratio())
            if score >= 0.8:
                pairs.append((score, secs[j] - secs[i], i, j))
    # 2) best matches first, each video used in at most one pair (so a label copy can't
    #    swallow a similar-looking sibling such as "Choreography" vs "Choreography (Moving ver.)")
    pairs.sort(key=lambda p: (-p[0], p[1]))
    used, drop, give = set(), set(), {}
    for _, _, i, j in pairs:
        if i in used or j in used:
            continue
        used.update((i, j))
        win, lose = (i, j) if views[i] >= views[j] else (j, i)
        drop.add(labels[lose])
        # the artist's own channel is the more trustworthy label
        if pd.isna(artist[win]) or (own[lose] and not own[win]):
            if pd.notna(artist[lose]):
                give[labels[win]] = artist[lose]
    return drop, give


# ---------------------------------------------------------------- main
def main(raw=RAW, out=OUT, report=REPORT):
    df = pd.read_csv(raw)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["source_channel"] = df["channel"]
    print(f"{len(df):,} videos from {df['source_channel'].nunique()} YouTube channels")

    # --- dedicated artist channels
    is_label = df["source_channel"].isin(LABEL_CHANNELS)
    df["artist"] = df["source_channel"].map(ARTIST_CHANNEL_NAMES)
    unknown = sorted(set(df.loc[~is_label & df["artist"].isna(), "source_channel"]))
    for ch in unknown:  # a channel in the CSV that is in neither list: name it after the channel
        df.loc[df["source_channel"] == ch, "artist"] = ch.lstrip("@")
        print(f"  note: '{ch}' is in neither list in add_artists.py; using '{ch.lstrip('@')}' as its artist name")

    # --- label channels: read the artist from the title
    if is_label.any():
        lab = df[is_label].copy()
        lab["guess"] = lab["title"].map(guess_artist)
        approved = approved_names()
        lab["artist"] = lab["guess"].map(lambda n: approved.get(norm_key(n)) if isinstance(n, str) else None)
        df.loc[lab.index, "artist"] = lab["artist"]

        print("\nLabel channels: which approved artists were found")
        for src, g in lab.groupby("source_channel"):
            found = g["artist"].notna().sum()
            print(f"  {LABEL_CHANNELS.get(src, src):20s} {len(g):>6,} videos -> {found:>6,} assigned to an approved artist, "
                  f"{len(g) - found:>6,} left out")
        top = lab["artist"].value_counts()
        print(f"  {len(top)} approved artists found: " + ", ".join(f"{n} ({c})" for n, c in top.head(40).items()))

        cand = candidate_table(lab)
        cand.to_csv(CANDIDATES, index=False)
        print(f"\n  {len(cand)} other names were read from titles but NOT added (shows, series, unlisted artists).")
        print(f"  Review {CANDIDATES}. To add an artist, put its name in KNOWN_ARTISTS and re-run. Top candidates:")
        for _, r in cand.head(25).iterrows():
            print(f"    {r['name'][:28]:28s} {r['videos']:>4} videos, {r['mv_style_titles']:>3} MV-style  -> {r['looks_like']}")

    # --- the same video published on two channels (done before dropping unlabeled rows, so a
    #     label-channel copy with more views and an unreadable title can still win)
    dup, give = find_duplicates(df)
    for lbl, art in give.items():
        df.loc[lbl, "artist"] = art
    df = df.drop(index=list(dup))
    print(f"\nRemoved {len(dup):,} duplicate videos (same video on two channels; kept the copy with more views)")

    df = df[df["artist"].notna()].copy()
    df["channel"] = df["artist"]
    df = df.drop(columns=["artist"])

    df = df.sort_values(["channel", "published_at"]).reset_index(drop=True)
    df.to_csv(out, index=False)

    rep = (df.groupby("channel")
             .agg(videos=("video_id", "size"), total_views=("views", "sum"),
                  source_channels=("source_channel", lambda s: " + ".join(sorted(set(s)))))
             .sort_values("videos", ascending=False))
    rep.to_csv(report)
    print(f"\nWrote {out}: {len(df):,} videos from {df['channel'].nunique()} artists (see {report})")
    return df


if __name__ == "__main__":
    main(*sys.argv[1:4])