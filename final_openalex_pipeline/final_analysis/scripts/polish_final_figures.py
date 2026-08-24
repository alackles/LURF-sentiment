from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Paths
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
OUT = PIPELINE_ROOT / "final_analysis" / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Color tones
GROUP_COLORS = {
    "older dialogue-system terms": "#708C72",
    "agent/assistant terms": "#7089A8",
    "chatbot/conversational-AI terms": "#C58D63",
}

SENTIMENT_COLORS = {
    "negative": "#B96A5E",
    "neutral": "#B8B2AA",
    "positive": "#708C72",
}

EMOTION_COLORS = {
    "trust": "#708C72",
    "anticipation": "#C58D63",
    "joy": "#D5B65A",
    "fear": "#7089A8",
    "sadness": "#8E7AA8",
    "anger": "#B96A5E",
}

# Period order
PERIOD_ORDER = [
    "1976-1999 early dialogue / NLI",
    "2000-2015 dialogue systems / HCI",
    "2016-2022 chatbot + assistant growth",
    "2023-2025 generative AI / chatbot boom",
]

PERIOD_LABELS = {
    "1976-1999 early dialogue / NLI": "1976–99\nEarly dialogue",
    "2000-2015 dialogue systems / HCI": "2000–15\nDialogue / HCI",
    "2016-2022 chatbot + assistant growth": "2016–22\nChatbot growth",
    "2023-2025 generative AI / chatbot boom": "2023–25\nGenAI boom",
}

GROUP_ORDER = [
    "older dialogue-system terms",
    "agent/assistant terms",
    "chatbot/conversational-AI terms",
]

GROUP_SHORT = {
    "older dialogue-system terms": "Older dialogue terms",
    "agent/assistant terms": "Agent/assistant terms",
    "chatbot/conversational-AI terms": "Chatbot / conv. AI terms",
}

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
EMOTIONS = ["trust", "anticipation", "joy", "fear", "sadness", "anger"]
SMALL_EMOTIONS = ["joy", "fear", "sadness", "anger"]

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})

def style_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linewidth=0.6, alpha=0.18)
    ax.set_axisbelow(True)

def save_fig(fig, filename):
    fig.savefig(FIGURES / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def order_periods(df):
    df = df.copy()
    df["analysis_period"] = pd.Categorical(
        df["analysis_period"],
        categories=PERIOD_ORDER,
        ordered=True,
    )
    return df.sort_values("analysis_period")

def add_bar_labels(ax, bars, threshold=7):
    for bar in bars:
        width = bar.get_width()
        if width >= threshold:
            ax.text(
                bar.get_x() + width / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="#222222",
            )

# Load tables
keyword_counts = pd.read_csv(TABLES / "keyword_group_counts_by_period.csv")
sentiment_labels = pd.read_csv(TABLES / "sentiment_label_by_period.csv")
emotion_by_year = pd.read_csv(TABLES / "emotion_by_year.csv")
emotion_pct = emotion_by_year.copy()
for emotion in EMOTIONS:
    emotion_pct[emotion] = emotion_pct[emotion] * 100

emotion_pct.to_csv(TABLES / "emotion_by_year_percent.csv", index=False)
sentiment_group = pd.read_csv(TABLES / "sentiment_by_keyword_group_and_period.csv")

# Save small-n check table
small_n = sentiment_group[sentiment_group["n"] < 30].copy()
small_n.to_csv(TABLES / "small_n_sentiment_group_check.csv", index=False)

# --------------------------------------------------
# Figure 1: Keyword group share by period
# --------------------------------------------------
kg = order_periods(keyword_counts)
kg["period_label"] = kg["analysis_period"].map(PERIOD_LABELS)

pivot_kg = (
    kg.pivot(index="period_label", columns="keyword_group", values="percent_within_period")
    .reindex(columns=GROUP_ORDER)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
left = np.zeros(len(pivot_kg))

for group in GROUP_ORDER:
    bars = ax.barh(
        pivot_kg.index,
        pivot_kg[group].values,
        left=left,
        color=GROUP_COLORS[group],
        edgecolor="white",
        linewidth=0.8,
        height=0.65,
        label=GROUP_SHORT[group],
    )
    add_bar_labels(ax, bars)
    left += pivot_kg[group].values

ax.set_title("Keyword Group Share by Time Period", pad=12)
ax.set_xlabel("Percent of keyword matches within period")
ax.set_ylabel("")
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
style_axes(ax, grid_axis="x")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

save_fig(fig, "01_keyword_group_percent_by_period.png")

# --------------------------------------------------
# Figure 2: Sentiment label distribution by period
# --------------------------------------------------
sl = order_periods(sentiment_labels)
sl["period_label"] = sl["analysis_period"].map(PERIOD_LABELS)

pivot_sl = (
    sl.pivot(index="period_label", columns="sentiment_label", values="percent_within_period")
    .reindex(columns=SENTIMENT_ORDER)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
left = np.zeros(len(pivot_sl))

for label in SENTIMENT_ORDER:
    bars = ax.barh(
        pivot_sl.index,
        pivot_sl[label].values,
        left=left,
        color=SENTIMENT_COLORS[label],
        edgecolor="white",
        linewidth=0.8,
        height=0.65,
        label=label.capitalize(),
    )
    add_bar_labels(ax, bars, threshold=6)
    left += pivot_sl[label].values

ax.set_title("Sentiment Label Distribution by Time Period", pad=12)
ax.set_xlabel("Percent of abstracts within period")
ax.set_ylabel("")
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
style_axes(ax, grid_axis="x")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

save_fig(fig, "02_sentiment_label_percent_by_period.png")

# --------------------------------------------------
# Figure 3A: Emotion score over time
# This keeps the original line-graph idea using NRCLex scores, not percentages.
# --------------------------------------------------
ey = emotion_by_year.copy().sort_values("publication_year")

for emotion in EMOTIONS:
    ey[f"{emotion}_smooth"] = ey[emotion].rolling(window=3, min_periods=1).mean()

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10.5, 8.2),
    sharex=True,
    gridspec_kw={"height_ratios": [2.1, 1.4]},
    constrained_layout=True,
)

for emotion in EMOTIONS:
    ax1.plot(
        ey["publication_year"],
        ey[f"{emotion}_smooth"],
        color=EMOTION_COLORS[emotion],
        linewidth=2.1,
        label=emotion.capitalize(),
    )

ax1.set_title("Emotion Scores Over Time in Conversational AI Abstracts", pad=10)
ax1.set_ylabel("Mean NRCLex emotion score")
ax1.set_ylim(0, ey[[f"{e}_smooth" for e in EMOTIONS]].max().max() * 1.08)
style_axes(ax1, grid_axis="y")
ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))

for emotion in SMALL_EMOTIONS:
    ax2.plot(
        ey["publication_year"],
        ey[f"{emotion}_smooth"],
        color=EMOTION_COLORS[emotion],
        linewidth=2.0,
        label=emotion.capitalize(),
    )

ax2.set_title("Zoomed View of Smaller Emotion Scores", fontsize=12, pad=8)
ax2.set_xlabel("Publication year")
ax2.set_ylabel("Mean NRCLex emotion score")
ax2.set_ylim(0, ey[[f"{e}_smooth" for e in SMALL_EMOTIONS]].max().max() * 1.18)
style_axes(ax2, grid_axis="y")
ax2.legend(frameon=False, ncol=4, loc="upper left")

save_fig(fig, "03a_emotion_score_over_time.png")


# --------------------------------------------------
# Figure 3B: Emotion percentage share by time period
# This separates the percentage idea from the raw score trend.
# --------------------------------------------------
def year_to_period(year):
    if year <= 1999:
        return "1976-1999 early dialogue / NLI"
    elif year <= 2015:
        return "2000-2015 dialogue systems / HCI"
    elif year <= 2022:
        return "2016-2022 chatbot + assistant growth"
    else:
        return "2023-2025 generative AI / chatbot boom"


emotion_period = ey.copy()
emotion_period["analysis_period"] = emotion_period["publication_year"].apply(year_to_period)
emotion_period["analysis_period"] = pd.Categorical(
    emotion_period["analysis_period"],
    categories=PERIOD_ORDER,
    ordered=True,
)

emotion_share = (
    emotion_period
    .groupby("analysis_period")[EMOTIONS]
    .mean()
    .reset_index()
)

emotion_share["total_emotion_score"] = emotion_share[EMOTIONS].sum(axis=1)

for emotion in EMOTIONS:
    emotion_share[emotion] = emotion_share[emotion] / emotion_share["total_emotion_score"] * 100

emotion_share.to_csv(TABLES / "emotion_share_by_period_percent.csv", index=False)

emotion_share["period_label"] = emotion_share["analysis_period"].map(PERIOD_LABELS)

fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)

left = np.zeros(len(emotion_share))

for emotion in EMOTIONS:
    vals = emotion_share[emotion].values

    bars = ax.barh(
        emotion_share["period_label"],
        vals,
        left=left,
        color=EMOTION_COLORS[emotion],
        edgecolor="white",
        linewidth=0.8,
        height=0.65,
        label=emotion.capitalize(),
    )

    for bar, val in zip(bars, vals):
        if val >= 7:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#222222",
            )

    left += vals

ax.set_title("Emotion Share by Time Period", pad=12)
ax.set_xlabel("Percent of total emotion framing within period")
ax.set_ylabel("")
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
style_axes(ax, grid_axis="x")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6)

save_fig(fig, "03b_emotion_share_by_period.png")
# --------------------------------------------------
# Figure 4: Mean sentiment by keyword group and period
# --------------------------------------------------
sg = order_periods(sentiment_group)
sg["period_label"] = sg["analysis_period"].map(PERIOD_LABELS)

pivot_mean = (
    sg.pivot(index="period_label", columns="keyword_group", values="mean_sentiment")
    .reindex(columns=GROUP_ORDER)
)

pivot_n = (
    sg.pivot(index="period_label", columns="keyword_group", values="n")
    .reindex(columns=GROUP_ORDER)
)

fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
x = np.arange(len(pivot_mean.index))
x_offsets = {
    "older dialogue-system terms": -0.08,
    "agent/assistant terms": 0.00,
    "chatbot/conversational-AI terms": 0.08,
}

for group in GROUP_ORDER:
    y = pivot_mean[group].values
    nvals = pivot_n[group].values

    ax.plot(
        x,
        y,
        marker="o",
        markersize=7,
        linewidth=2.5,
        color=GROUP_COLORS[group],
        label=GROUP_SHORT[group],
    )

    for xi, yi, nval in zip(x, y, nvals):
        if pd.notna(yi):
            ax.text(
                xi + x_offsets[group],
                yi + 0.018,
                f"{yi:.2f}\n(n={int(nval)})",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

ax.set_xticks(x)
ax.set_xticklabels(pivot_mean.index)
ax.set_ylabel("Mean VADER compound score")
ax.set_xlabel("Time period")
ax.set_ylim(0.52, 1.05)
ax.set_title("Mean Sentiment by Keyword Group and Time Period", pad=12)
style_axes(ax, grid_axis="y")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)

save_fig(fig, "04_mean_sentiment_by_keyword_group_period.png")

print("Done. Polished figures saved to:", FIGURES)
print("Small-n check saved to:", TABLES / "small_n_sentiment_group_check.csv")
