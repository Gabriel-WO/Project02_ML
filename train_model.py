"""
train_model.py - Step 3 of Project02_ML (K-pop video popularity)

Trains and compares models that predict how popular a YouTube video is from
information known at release. Runs two experiments so we can see how the
video-age problem affects the results:

  A) target = log10(views per day), TIME split (train on old, test on new)
  B) target = log10(total views) with log10(video age) as a feature, RANDOM split

AI assistance: drafted with Claude (document this in the README).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("kpop_videos_clean.csv", parse_dates=["published_at"])
df = df.dropna(subset=["prev_avg_log_vpd"])  # first video of each channel
df["log_days_since_release"] = np.log10(df["days_since_release"])

CATEGORICAL = ["video_type", "channel"]
BASE_NUMERIC = [
    "duration_sec", "release_year", "release_month", "release_dow",
    "release_hour_utc", "title_has_korean", "title_has_english",
    "title_length", "n_hashtags", "prev_video_count", "prev_avg_log_vpd",
]


def make_models(numeric):
    def pipe(model, scale):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler() if scale else "passthrough", numeric),
        ])
        return Pipeline([("pre", pre), ("model", model)])
    return {
        "Baseline (predict the mean)": pipe(DummyRegressor(), False),
        "Linear regression": pipe(LinearRegression(), True),
        "Random forest": pipe(RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, max_features=0.5,
            n_jobs=-1, random_state=42), False),
    }


def run(title, train, test, target, numeric):
    print(f"\n=== {title} ===")
    print(f"train: {len(train)} rows | test: {len(test)} rows | target: {target}")
    cols = CATEGORICAL + numeric
    models = make_models(numeric)
    print(f"{'model':30s} {'MAE':>7s} {'R2':>7s}")
    for name, m in models.items():
        m.fit(train[cols], train[target])
        pred = m.predict(test[cols])
        print(f"{name:30s} {mean_absolute_error(test[target], pred):7.3f} "
              f"{r2_score(test[target], pred):7.3f}")
    return models["Random forest"]


def show_importance(rf):
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
    grouped = imp.groupby(lambda n: "channel" if n.startswith("cat__channel")
                          else "video_type" if n.startswith("cat__video_type")
                          else n.split("__")[1]).sum()
    print("\nRandom forest feature importance (grouped):")
    print(grouped.sort_values(ascending=False).round(3).to_string())


# Experiment A: views per day, time-based split
cutoff = df["published_at"].quantile(0.8)
run(f"A: views/day, time split (test = videos after {cutoff.date()})",
    df[df["published_at"] <= cutoff], df[df["published_at"] > cutoff],
    "log_views_per_day", BASE_NUMERIC)

# Experiment B: total views, age as a feature, random split
train, test = train_test_split(df, test_size=0.2, random_state=42)
rf = run("B: total views, age as feature, random split", train, test,
         "log_views", BASE_NUMERIC + ["log_days_since_release"])
show_importance(rf)