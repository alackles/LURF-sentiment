from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
INPUT = PIPELINE_ROOT / "data/processed/openalex_clean_corpus.parquet"
OUT = PIPELINE_ROOT / "final_analysis" / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEARCH_TERMS = [
    "chatbot",
    "dialogue system",
    "conversational agent",
    "virtual assistant",
    "voice assistant",
    "conversational AI",
    "dialog system",
    "natural language interface",
    "spoken dialogue system",
    "embodied conversational agent",
    "chatterbot",
]

OLDER_TERMS = {
    "dialogue system",
    "dialog system",
    "spoken dialogue system",
    "natural language interface",
}

ASSISTANT_TERMS = {
    "conversational agent",
    "embodied conversational agent",
    "virtual assistant",
    "voice assistant",
}

CHATBOT_TERMS = {
    "chatbot",
    "chatterbot",
    "conversational AI",
}

GROUP_COLORS = {
    "older": "#708C72",
    "assistant": "#7089A8",
    "chatbot": "#C58D63",
}

def term_color(term):
    if term in OLDER_TERMS:
        return GROUP_COLORS["older"]
    if term in ASSISTANT_TERMS:
        return GROUP_COLORS["assistant"]
    return GROUP_COLORS["chatbot"]

def parse_terms(value):
    if pd.isna(value):
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]

def compact_number(x, pos):
    if x >= 1000:
        return f"{x/1000:.0f}k"
    return f"{int(x)}"

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})

print("Loading full clean corpus...")
df = pd.read_parquet(INPUT)

df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df = df.dropna(subset=["publication_year"]).copy()
df["publication_year"] = df["publication_year"].astype(int)
df["detected_terms_list"] = df["detected_terms_in_title_abstract"].apply(parse_terms)

term_df = df.explode("detected_terms_list").dropna(subset=["detected_terms_list"]).copy()
term_df = term_df[term_df["detected_terms_list"].isin(SEARCH_TERMS)].copy()

term_year = (
    term_df.groupby(["publication_year", "detected_terms_list"])
    .size()
    .reset_index(name="count")
    .rename(columns={"detected_terms_list": "keyword"})
)

term_year.to_csv(TABLES / "keyword_frequency_by_year.csv", index=False)

term_totals = (
    term_year.groupby("keyword")["count"]
    .sum()
    .sort_values(ascending=False)
)

terms = list(term_totals.index)

min_year = int(df["publication_year"].min())
max_year = int(df["publication_year"].max())
year_index = pd.Index(range(min_year, max_year + 1), name="publication_year")

ncols = 3
nrows = math.ceil(len(terms) / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(13.5, 10.8),
    sharex=True,
    constrained_layout=True,
)

axes = axes.flatten()

for ax, term in zip(axes, terms):
    # Fix: reindex only the numeric count series, not the whole dataframe.
    one = (
        term_year[term_year["keyword"] == term]
        .set_index("publication_year")["count"]
        .reindex(year_index, fill_value=0)
        .rename("count")
        .reset_index()
    )

    one["smooth_count"] = one["count"].rolling(window=3, min_periods=1).mean()

    color = term_color(term)

    ax.plot(
        one["publication_year"],
        one["smooth_count"],
        color=color,
        linewidth=2.2,
    )

    ax.fill_between(
        one["publication_year"],
        one["smooth_count"],
        color=color,
        alpha=0.14,
    )

    ax.set_title(
        f"{term}\nTotal: {term_totals[term]:,}",
        pad=6,
        color="#222222",
    )

    ax.set_xlim(min_year, max_year)
    ax.set_xticks([1980, 1990, 2000, 2010, 2020])
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
    style_axes(ax)

for ax in axes[len(terms):]:
    ax.axis("off")

fig.suptitle(
    "Individual Keyword Frequency Over Time",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)

fig.text(
    0.5,
    -0.01,
    "Each panel uses its own y-axis so smaller keyword patterns remain visible. Lines show 3-year smoothed keyword-match counts.",
    ha="center",
    fontsize=9.5,
    color="#444444",
)

fig.savefig(
    FIGURES / "05_keyword_frequency_small_multiples.png",
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("Saved:")
print(TABLES / "keyword_frequency_by_year.csv")
print(FIGURES / "05_keyword_frequency_small_multiples.png")
