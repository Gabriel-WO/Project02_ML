"""
tune_models.py - Step 4 of Project02_ML (K-pop video popularity)

Compares a default random forest, a GridSearchCV-tuned random forest, and
HistGradientBoostingRegressor (default and tuned) on the same train/test split
as train_model.py experiment B (target = log10 total views, video age as a
feature).

To keep the grid search fast, hyperparameters are searched on a 15,000-row
sample of the training set with 3-fold cross-validation; the best settings
are then refit on the FULL training set and scored once on the test set.

AI assistance: drafted with Claude (document this in the README).
"""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
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
TARGET = "log_views"
COLS = CATEGORICAL + NUMERIC

# Same split as train_model.py so results are comparable.
train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train = train[COLS], train[TARGET]
X_test, y_test = test[COLS], test[TARGET]

# Smaller sample just for the hyperparameter search.
search = train.sample(15000, random_state=0)
X_search, y_search = search[COLS], search[TARGET]


def pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, pipe, fit_seconds):
    pred = pipe.predict(X_test)
    mae, r2 = mean_absolute_error(y_test, pred), r2_score(y_test, pred)
    print(f"{name:34s} MAE {mae:.3f}   R2 {r2:.3f}   ({fit_seconds:.0f}s)")
    return mae, r2


results = {}

# 1. Default random forest (same settings as train_model.py)
t = time.time()
rf_default = pipeline(RandomForestRegressor(
    n_estimators=100, min_samples_leaf=10, max_features=0.5,
    n_jobs=-1, random_state=42))
rf_default.fit(X_train, y_train)
results["Random forest (default)"] = evaluate("Random forest (default)", rf_default, time.time() - t)

# 2. Tuned random forest with GridSearchCV
t = time.time()
rf_grid = GridSearchCV(
    pipeline(RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)),
    param_grid={
        "model__min_samples_leaf": [1, 3, 10],
        "model__max_features": [0.33, 0.5, 1.0],
        "model__max_depth": [None, 20],
    },
    cv=3, scoring="neg_mean_absolute_error", n_jobs=1, verbose=0,
)
rf_grid.fit(X_search, y_search)
print("\nBest forest params:", rf_grid.best_params_)
print(f"Best forest CV MAE (on sample): {-rf_grid.best_score_:.3f}")
cv = pd.DataFrame(rf_grid.cv_results_)[
    ["param_model__min_samples_leaf", "param_model__max_features",
     "param_model__max_depth", "mean_test_score"]
].assign(cv_mae=lambda d: -d["mean_test_score"]).drop(columns="mean_test_score")
print(cv.sort_values("cv_mae").head(6).to_string(index=False))

best_rf = pipeline(RandomForestRegressor(
    n_estimators=300, n_jobs=-1, random_state=42,
    **{k.replace("model__", ""): v for k, v in rf_grid.best_params_.items()}))
best_rf.fit(X_train, y_train)
print()
results["Random forest (tuned)"] = evaluate("Random forest (tuned, 300 trees)", best_rf, time.time() - t)

# 3. HistGradientBoosting, default
t = time.time()
hgb_default = pipeline(HistGradientBoostingRegressor(random_state=42))
hgb_default.fit(X_train, y_train)
results["HistGradientBoosting (default)"] = evaluate("HistGradientBoosting (default)", hgb_default, time.time() - t)

# 4. HistGradientBoosting, tuned with GridSearchCV
t = time.time()
hgb_grid = GridSearchCV(
    pipeline(HistGradientBoostingRegressor(random_state=42)),
    param_grid={
        "model__learning_rate": [0.05, 0.1],
        "model__max_iter": [200, 500],
        "model__max_leaf_nodes": [31, 63],
        "model__min_samples_leaf": [20, 50],
    },
    cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0,
)
hgb_grid.fit(X_search, y_search)
print("\nBest boosting params:", hgb_grid.best_params_)
print(f"Best boosting CV MAE (on sample): {-hgb_grid.best_score_:.3f}")
best_hgb = pipeline(HistGradientBoostingRegressor(
    random_state=42,
    **{k.replace("model__", ""): v for k, v in hgb_grid.best_params_.items()}))
best_hgb.fit(X_train, y_train)
print()
results["HistGradientBoosting (tuned)"] = evaluate("HistGradientBoosting (tuned)", best_hgb, time.time() - t)

print("\nSummary (test set, log10 total views):")
summary = pd.DataFrame(results, index=["MAE", "R2"]).T.round(3)
print(summary.to_string())
summary.to_csv("model_comparison.csv")