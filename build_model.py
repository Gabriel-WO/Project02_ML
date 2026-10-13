"""
build_model.py - Step 5 of Project02_ML (K-pop video popularity)

1. Scores the app's random forest on the same held-out test set as before.
2. Saves test-set predictions (used by make_plots.py).
3. Measures how wrong the model tends to be at each prediction level, using
   5-fold out-of-fold predictions, so the app can show an honest 80% range.
4. Refits on ALL data and saves model.joblib + model_info.json for app.py,
   including each channel's upload history (so the app can look up a
   channel's record AS OF any release date).

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
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

full = pd.read_csv("kpop_videos_clean.csv", parse_dates=["published_at"])
df = full.dropna(subset=["prev_avg_log_vpd"]).copy()
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

# 3. Honest prediction ranges.
#    A random forest pulls big hits toward the middle, so errors are NOT the
#    same size everywhere: videos predicted to be large are usually
#    underpredicted. We record the 10th/90th percentile of the error
#    (actual minus predicted, in log10) for each prediction level, using
#    out-of-fold predictions so every video is scored by a model that never
#    saw it. The app turns these into an 80% range.
oof = cross_val_predict(make_model(), df[COLS], df[TARGET],
                        cv=KFold(5, shuffle=True, random_state=0))
resid = df[TARGET].values - oof
edges = np.quantile(oof, [0.2, 0.4, 0.6, 0.8, 0.95, 0.99])
bin_id = np.digitize(oof, edges)
range_bins = {"edges": [round(float(e), 4) for e in edges], "q10": [], "q90": []}
print("\nHow far off is the model, by predicted level? (80% of actual views land in this multiple of the prediction)")
for b in range(len(edges) + 1):
    lo, hi = np.quantile(resid[bin_id == b], [0.1, 0.9])
    range_bins["q10"].append(round(float(lo), 4))
    range_bins["q90"].append(round(float(hi), 4))
    typical = 10 ** np.median(oof[bin_id == b])
    print(f"  predicted about {typical:>13,.0f} views (n={int((bin_id == b).sum()):>5}): "
          f"x{10 ** lo:.2f} to x{10 ** hi:.2f}")

# 4. Refit on all data for the app
final = make_model().fit(df[COLS], df[TARGET])
joblib.dump(final, "model.joblib", compress=3)

# Per-channel facts for the app. "history" lets the app compute, for any
# release date, how many videos the channel had uploaded before it and their
# average views per day, matching how the training features were built.
epoch = pd.Timestamp("1970-01-01", tz="UTC")
channels, history = {}, {}
for ch, g in full.sort_values("published_at").groupby("channel"):
    channels[ch] = {
        "video_count": int(len(g)),
        "avg_log_vpd": float(g["log_views_per_day"].mean()),
        "median_duration": float(g["duration_sec"].median()),
    }
    history[ch] = {
        "t_hours": ((g["published_at"] - epoch) // pd.Timedelta(hours=1)).astype(int).tolist(),
        "cum_lvpd": np.round(np.cumsum(g["log_views_per_day"].values), 4).tolist(),
    }
info = {
    "channels": channels,
    "history": history,
    "video_types": sorted(df["video_type"].unique().tolist()),
    "range_bins": range_bins,
    "test_mae": round(mae, 3),
    "test_r2": round(r2, 3),
    "min_age_days": 1,
    "max_age_days": int(df["days_since_release"].max()),
}
with open("model_info.json", "w") as f:
    json.dump(info, f)
print("\nsaved model.joblib, model_info.json, test_predictions.csv, feature_importance.csv")