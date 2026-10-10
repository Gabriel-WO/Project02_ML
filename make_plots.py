"""
make_plots.py - Step 6 of Project02_ML (K-pop video popularity)

Makes the presentation graphs into the plots/ folder. Run build_model.py first
(it creates test_predictions.csv and feature_importance.csv).

AI assistance: drafted with Claude (document this in the README).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

os.makedirs("plots", exist_ok=True)

# One-hue palette, light surface (single series per chart -> no legend needed).
BLUE, GRAY, INK, MUTED, GRID = "#2a78d6", "#a8a7a1", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight",
})


def human(x, _=None):
    """10**x as 1K / 1M / 1B for log10 axes."""
    v = 10 ** x
    for div, s in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if v >= div:
            return f"{v / div:g}{s}"
    return f"{v:g}"


df = pd.read_csv("kpop_videos_clean.csv", parse_dates=["published_at"])
preds = pd.read_csv("test_predictions.csv")
imp = pd.read_csv("feature_importance.csv", index_col=0)["importance"]
comp = pd.read_csv("model_comparison.csv", index_col=0) if os.path.exists("model_comparison.csv") else None

# 1. Distribution of total views (log scale)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df["log_views"], bins=60, color=BLUE, edgecolor="#fcfcfb", linewidth=0.6)
ax.xaxis.set_major_formatter(mt.FuncFormatter(human))
ax.set_xlabel("Total views (log scale)")
ax.set_ylabel("Number of videos")
ax.set_title("Views span five orders of magnitude, so we model log10(views)")
ax.axvline(df["log_views"].median(), color=INK, lw=1.2, ls="--")
ax.text(df["log_views"].median() + 0.05, ax.get_ylim()[1] * 0.92,
        f"median {human(df['log_views'].median())}", color=INK, fontsize=10)
fig.savefig("plots/1_views_distribution.png"); plt.close(fig)

# 2. Median views by video type
t = df.groupby("video_type")["log_views"].agg(["median", "size"]).sort_values("median")
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.barh(t.index.str.replace("_", " "), t["median"], color=BLUE, height=0.65)
ax.xaxis.set_major_formatter(mt.FuncFormatter(human))
ax.set_xlim(3, t["median"].max() + 0.25)
ax.set_xlabel("Median total views (log scale)")
ax.set_title("What kind of video it is matters a lot")
for b, (m, n) in zip(bars, zip(t["median"], t["size"])):
    ax.text(m + 0.03, b.get_y() + b.get_height() / 2, f"{human(m)}  (n={n:,})",
            va="center", fontsize=9.5, color=MUTED)
ax.grid(axis="y", visible=False)
fig.savefig("plots/2_views_by_video_type.png"); plt.close(fig)

# 3. Median views per day by channel (top 15 by video count)
top = df["channel"].value_counts().head(15).index
c = df[df["channel"].isin(top)].groupby("channel")["log_views_per_day"].median().sort_values()
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(c.index.str.lstrip("@"), c.values, color=BLUE, height=0.65)
ax.xaxis.set_major_formatter(mt.FuncFormatter(human))
ax.set_xlabel("Median views per day (log scale)")
ax.set_title("Channel size is the biggest driver of views")
ax.grid(axis="y", visible=False)
fig.savefig("plots/3_views_by_channel.png"); plt.close(fig)

# 4. Era check: views per day by release year (the original "eras" idea)
y = df[df["release_year"] >= 2012].groupby("release_year")["log_views_per_day"].median()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(y.index, y.values, color=BLUE, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2)
ax.yaxis.set_major_formatter(mt.FuncFormatter(human))
ax.set_xlabel("Release year")
ax.set_ylabel("Median views per day (log scale)")
ax.set_title("Newer videos have higher views per day (age effect)")
fig.savefig("plots/4_views_per_day_by_year.png"); plt.close(fig)

# 5. Predicted vs actual on the held-out test set
fig, ax = plt.subplots(figsize=(6.2, 6.2))
hb = ax.hexbin(preds["log_views"], preds["predicted_log_views"], gridsize=45,
               cmap="Blues", mincnt=1, bins="log", linewidths=0.2)
lo, hi = preds["log_views"].min(), preds["log_views"].max()
ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--")
ax.xaxis.set_major_formatter(mt.FuncFormatter(human))
ax.yaxis.set_major_formatter(mt.FuncFormatter(human))
ax.set_xlabel("Actual total views")
ax.set_ylabel("Predicted total views")
ax.set_title("Predictions hug the diagonal")
cb = fig.colorbar(hb, ax=ax, shrink=0.75); cb.set_label("videos (log count)", color=MUTED)
fig.savefig("plots/5_predicted_vs_actual.png"); plt.close(fig)

# 6. Feature importance
imp_s = imp.sort_values()
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.barh(imp_s.index, imp_s.values, color=BLUE, height=0.65)
ax.set_xlabel("Share of random forest importance")
ax.set_title("A channel's earlier performance predicts the next video best")
ax.grid(axis="y", visible=False)
fig.savefig("plots/6_feature_importance.png"); plt.close(fig)

# 7. Model comparison (from tune_models.py output), if present
if comp is not None:
    comp = comp.sort_values("R2")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(comp.index, comp["R2"], color=BLUE, height=0.6)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("R² on held-out test set (higher is better)")
    ax.set_title("Tuning the random forest gave the best score")
    for b, v in zip(bars, comp["R2"]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
                fontsize=10, color=MUTED)
    ax.grid(axis="y", visible=False)
    fig.savefig("plots/7_model_comparison.png"); plt.close(fig)

print("saved:", sorted(os.listdir("plots")))