"""
build_model.py - Step 5 of Project02_ML (K-pop video popularity)

1. Scores the app's random forest on the same held-out test set as before.
2. Saves test-set predictions (used by make_plots.py).
3. Refits on ALL data and saves model.joblib + model_info.json for app.py.

The app model is a smaller forest than the 355 MB one from tune_models.py
(min_samples_leaf=5, 100 trees, max_features=0.33), so the file is ~23 MB and
fits on GitHub. model.joblib is git-ignored: anyone can recreate it by
running this script.

AI assistance: drafted with Claude (document this in the README).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("kpop_videos_clean.csv", parse_dates=["published_at"])
df = df.dropna(subset=["prev_avg_log_vpd"])
df["log_days_since_release"] = np.log10(df["days_since_release"])

CATEGORICAL = ["video_type", "channel"]
NUMERIC = [
    "duration_sec", "release_year", "release_month", "release_dow",
    "release_hour_utc", "title_has_korean", "title_has_english",
    "title_length", "n_hashtags", "prev_video_count", "prev_avg_log_vpd",
    "log_days_since_release",
]
COLS = CATEGORICAL + NUMERIC
TARGET = "log_views"


def make_model():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                               max_features=0.33, n_jobs=-1, random_state=42)
    return Pipeline([("pre", pre), ("model", rf)])


# 1. Honest score on the same split as train_model.py / tune_models.py
train, test = train_test_split(df, test_size=0.2, random_state=42)
model = make_model().fit(train[COLS], train[TARGET])
pred = model.predict(test[COLS])
mae, r2 = mean_absolute_error(test[TARGET], pred), r2_score(test[TARGET], pred)
print(f"App model on held-out test set: MAE {mae:.3f}  R2 {r2:.3f}")

# 2. Save test predictions + grouped feature importance for the plots
out = test[["channel", "video_type", "title", "log_views"]].copy()
out["predicted_log_views"] = pred
out.to_csv("test_predictions.csv", index=False)

names = model.named_steps["pre"].get_feature_names_out()
imp = pd.Series(model.named_steps["model"].feature_importances_, index=names)
grouped = imp.groupby(lambda n: "channel" if n.startswith("cat__channel")
                      else "video_type" if n.startswith("cat__video_type")
                      else n.split("__")[1]).sum().sort_values(ascending=False)
grouped.to_csv("feature_importance.csv", header=["importance"])

# 3. Refit on all data for the app
final = make_model().fit(df[COLS], df[TARGET])
joblib.dump(final, "model.joblib", compress=3)

# Per-channel facts the app uses to fill in "history" features automatically
channels = {}
for ch, g in df.sort_values("published_at").groupby("channel"):
    channels[ch] = {
        "video_count": int(len(g)),
        "avg_log_vpd": float(g["log_views_per_day"].mean()),
        "median_duration": float(g["duration_sec"].median()),
    }
info = {
    "channels": channels,
    "video_types": sorted(df["video_type"].unique().tolist()),
    "test_mae": round(mae, 3),
    "test_r2": round(r2, 3),
    "min_age_days": 1,
    "max_age_days": int(df["days_since_release"].max()),
}
with open("model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print("saved model.joblib, model_info.json, test_predictions.csv, feature_importance.csv")