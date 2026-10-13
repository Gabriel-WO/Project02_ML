"""
app.py - Step 7 of Project02_ML (K-pop video popularity)

A small Streamlit interface: describe a K-pop YouTube video and see its
predicted view count, plus how the prediction grows as the video ages.

How to open it:
  * In PyCharm: open this file and press the green Run button (or right-click
    app.py -> Run 'app'). It starts the server and opens your browser.
    Press the red Stop button to shut it down.
  * Or from a terminal:   streamlit run app.py

If model.joblib does not exist yet, the first run trains it automatically
by running build_model.py (takes about a minute).

AI assistance: drafted with Claude (document this in the README).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _running_in_streamlit():
    """True when Streamlit is executing this file (False for a plain `python app.py`)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        return False


def _model_is_current():
    """True if model.joblib and a model_info.json from the current build_model.py exist."""
    import json
    try:
        with open(os.path.join(HERE, "model_info.json")) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return False
    return (os.path.exists(os.path.join(HERE, "model.joblib"))
            and "range_bins" in info and "history" in info)


def _launch():
    """Start the Streamlit server for this file in-process and open the browser."""
    import runpy
    import socket
    import threading
    import time
    import urllib.request
    import webbrowser

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit("Streamlit is not installed for this interpreter. "
                 "In the PyCharm Terminal run:  pip install streamlit")

    os.chdir(HERE)  # so model.joblib / CSV files are found whatever the run config says

    if not _model_is_current():
        print("Training the model (about a minute)...", flush=True)
        runpy.run_path(os.path.join(HERE, "build_model.py"), run_name="__main__")

    # Prefer Streamlit's usual port; fall back to any free one if it is busy.
    port = 8501
    with socket.socket() as s:
        if os.name != "nt":
            # Without this, a server that was just stopped can leave the port
            # "in use" for a minute and we would pick a different one each time.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("localhost", port))
        except OSError:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
    url = f"http://localhost:{port}"

    def open_browser_when_ready():
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    print(f"Starting the app at {url}  (press Stop in PyCharm to shut it down)", flush=True)

    # headless=True skips Streamlit's first-run email prompt, which would hang
    # PyCharm's Run window; we open the browser ourselves above.
    sys.argv = ["streamlit", "run", os.path.abspath(__file__),
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"]
    sys.exit(stcli.main())


if __name__ == "__main__" and not _running_in_streamlit():
    _launch()


# ---------------------------------------------------------------------------
# The app itself (only reached when Streamlit is running this file)
# ---------------------------------------------------------------------------
import bisect
import datetime as dt
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(HERE, "model.joblib")
INFO_PATH = os.path.join(HERE, "model_info.json")

st.set_page_config(page_title="K-pop View Predictor", page_icon="🎤", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(MODEL_PATH)
    with open(INFO_PATH) as f:
        info = json.load(f)
    return model, info


if not (os.path.exists(MODEL_PATH) and os.path.exists(INFO_PATH)):
    st.error("model.joblib not found. Run `python build_model.py` first, then reload this page.")
    st.stop()

model, info = load()
if "range_bins" not in info or "history" not in info:
    st.error("model_info.json is from an older version of build_model.py. Delete model.joblib and "
             "model_info.json, then run `python build_model.py` and reload this page.")
    st.stop()
channels = info["channels"]
history = info.get("history", {})
range_bins = info["range_bins"]
MAE = info["test_mae"]  # typical error on the log10 scale
EPOCH = dt.datetime(1970, 1, 1)


def fmt(n):
    """Human-readable view count."""
    for div, s in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if n >= div:
            return f"{n / div:.3g}{s}"
    return f"{n:,.0f}"


def channel_record(channel, release_dt):
    """The channel's upload count and average views/day for videos published BEFORE
    release_dt. This matches how the training features were built."""
    h = history[channel]
    hours = (release_dt - EPOCH).total_seconds() / 3600
    n = bisect.bisect_left(h["t_hours"], hours)
    if n == 0:  # nothing earlier in the data: fall back to the whole-channel average
        return channels[channel]["video_count"], channels[channel]["avg_log_vpd"], False
    return n, h["cum_lvpd"][n - 1] / n, True


def make_row(channel, video_type, release, hour, duration, title, age_days, prev_count, prev_avg):
    """Build one model input row from what the user typed."""
    return pd.DataFrame([{
        "video_type": video_type,
        "channel": channel,
        "duration_sec": duration,
        "release_year": release.year,
        "release_month": release.month,
        "release_dow": release.weekday(),
        "release_hour_utc": hour,
        "title_has_korean": int(any("가" <= c <= "힣" for c in title)),
        "title_has_english": int(any(c.isascii() and c.isalpha() for c in title)),
        "title_length": len(title),
        "n_hashtags": title.count("#"),
        "prev_video_count": prev_count,
        "prev_avg_log_vpd": prev_avg,
        "log_days_since_release": np.log10(max(age_days, 1)),
    }])


def prediction_range(pred_log):
    """80% range, widened where the model is known to be less accurate."""
    b = int(np.digitize(pred_log, range_bins["edges"]))
    return 10 ** (pred_log + range_bins["q10"][b]), 10 ** (pred_log + range_bins["q90"][b])


st.title("🎤 K-pop YouTube View Predictor")
st.caption(
    f"Random forest trained on {sum(c['video_count'] for c in channels.values()):,} videos "
    f"from {len(channels)} official channels. Held-out test score: R² {info['test_r2']}, "
    f"typical error about {10 ** MAE:.1f}x."
)

left, right = st.columns([1, 1.4], gap="large")
today = dt.date.today()

with left:
    st.subheader("Describe the video")
    channel = st.selectbox("Artist channel", sorted(channels),
                           format_func=lambda c: c.lstrip("@"))
    default_type = "mv" if "mv" in info["video_types"] else info["video_types"][0]
    video_type = st.selectbox(
        "Video type", info["video_types"],
        index=info["video_types"].index(default_type),
        format_func=lambda t: t.replace("_", " "),
    )
    title = st.text_input("Video title", "ARTIST 'Song Name' Official MV #KPOP",
                          help="Length, hashtags, and Korean/English text are used as features.")
    c1, c2 = st.columns(2)
    release = c1.date_input("Release date", today,
                            min_value=dt.date(2009, 1, 1), max_value=dt.date(2030, 12, 31))
    hour = c2.slider("Release hour (UTC)", 0, 23, 9)
    duration = st.slider("Video length (seconds)", 10, 600,
                         int(min(max(channels[channel]["median_duration"], 10), 600)))

    mode = st.radio("Predict views", ["As of today", "After a set number of days"],
                    index=0 if release < today else 1, horizontal=True,
                    help="'As of today' counts the days from the release date to today, "
                         "so use it for videos that are already out.")
    if mode == "As of today":
        age = max((today - release).days, 1)
        st.caption(f"Video age: {age:,} days")
    else:
        age = st.slider("Days after release", 1, 3650, 30)

    release_dt = dt.datetime.combine(release, dt.time(hour))
    prev_count, prev_avg, found = channel_record(channel, release_dt)
    note = (f"Before this release date, {channel.lstrip('@')} had {prev_count:,} videos in the data "
            f"averaging about {fmt(10 ** prev_avg)} views per day."
            if found else
            f"{channel.lstrip('@')} has no earlier videos in the data before this date, so its "
            f"overall average ({fmt(10 ** prev_avg)} views per day) is used.")
    st.info(note + " The model fills these in automatically.")

row = make_row(channel, video_type, release, hour, duration, title, age, prev_count, prev_avg)
pred_log = float(model.predict(row)[0])
pred = 10 ** pred_log
low, high = prediction_range(pred_log)

with right:
    st.subheader("Prediction")
    m1, m2 = st.columns(2)
    m1.metric(f"Total views after {age:,} day{'s' if age != 1 else ''}", fmt(pred))
    m2.metric("Likely range (80%)", f"{fmt(low)} to {fmt(high)}",
              help="Built from the model's actual errors at this prediction level in testing. "
                   "The range is lopsided because the model tends to underpredict big hits. "
                   "Real blockbusters can still land above it.")

    # How predicted views grow with age
    ages = np.unique(np.round(np.logspace(0, np.log10(max(3650, age)), 40)).astype(int))
    curve = pd.concat([make_row(channel, video_type, release, hour, duration, title, a,
                                prev_count, prev_avg) for a in ages])
    curve_df = pd.DataFrame({"days": ages, "views": 10 ** model.predict(curve)})
    line = alt.Chart(curve_df).mark_line(strokeWidth=2.5, color="#2a78d6").encode(
        x=alt.X("days:Q", title="Days since release", scale=alt.Scale(type="log")),
        y=alt.Y("views:Q", title="Predicted total views", scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("days:Q", title="Day"),
                 alt.Tooltip("views:Q", title="Predicted views", format=",.0f")],
    )
    dot = alt.Chart(pd.DataFrame({"days": [age], "views": [pred]})).mark_point(
        size=120, filled=True, color="#eb6834", stroke="white", strokeWidth=2
    ).encode(x="days:Q", y="views:Q")
    st.altair_chart((line + dot).properties(height=320), use_container_width=True)

    st.subheader("Compare video types for this artist")
    types = pd.concat([make_row(channel, t, release, hour, duration, title, age, prev_count, prev_avg)
                       for t in info["video_types"]])
    cmp_df = pd.DataFrame({
        "type": [t.replace("_", " ") for t in info["video_types"]],
        "views": 10 ** model.predict(types),
    }).sort_values("views", ascending=False)
    bars = alt.Chart(cmp_df).mark_bar(color="#2a78d6", cornerRadiusEnd=4).encode(
        x=alt.X("views:Q", title=f"Predicted views after {age:,} days", axis=alt.Axis(format="~s")),
        y=alt.Y("type:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("type:N"), alt.Tooltip("views:Q", format=",.0f")],
    )
    st.altair_chart(bars.properties(height=300), use_container_width=True)

with st.expander("Limitations (read before trusting a number)"):
    st.markdown(
        f"- The model learned from **already released videos from {len(channels)} official channels**, so it "
        "works best for artists in that list.\n"
        "- **Mega-hits are badly underpredicted.** In testing, videos with over 100M views came out about "
        "10x too low on average. Only a few dozen videos in the data are that big, and the model cannot "
        "see hype, fan-base size, or marketing. For a major group's music video, treat the top of the "
        "range as the realistic number.\n"
        "- A brand-new group has no channel history for the model to use.\n"
        "- Predictions for the first few days are the least reliable, because few training "
        "videos were that young when the data was pulled.\n"
        "- The score comes from a random train/test split, which is lenient. Expect worse "
        "accuracy on a truly new comeback."
    )