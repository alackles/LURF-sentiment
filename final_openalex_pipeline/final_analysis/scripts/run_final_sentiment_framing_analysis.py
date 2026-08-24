from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[2]

INPUT = PIPELINE_ROOT / "data/processed/df_en_scored.parquet"
OUTPUT = PIPELINE_ROOT / "final_analysis/outputs"

TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
VALIDATION = OUTPUT / "validation_samples"
REPORTS = OUTPUT / "reports"

for folder in [TABLES, FIGURES, VALIDATION, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["fear", "anticipation", "trust", "joy", "sadness", "anger"]

OLDER_TERMS = [
    "dialogue system",
    "dialog system",
    "spoken dialogue system",
    "natural language interface",
]

AGENT_ASSISTANT_TERMS = [
    "conversational agent",
    "embodied conversational agent",
    "virtual assistant",
    "voice assistant",
]

CHATBOT_AI_TERMS = [
    "chatbot",
    "chatterbot",
    "conversational AI",
]

TERM_TO_GROUP = {}
for term in OLDER_TERMS:
    TERM_TO_GROUP[term] = "older dialogue-system terms"
for term in AGENT_ASSISTANT_TERMS:
    TERM_TO_GROUP[term] = "agent/assistant terms"
for term in CHATBOT_AI_TERMS:
    TERM_TO_GROUP[term] = "chatbot/conversational-AI terms"


# -----------------------------
# Helper functions
# -----------------------------
def assign_period(year):
    if year <= 1999:
        return "1976-1999 early dialogue / NLI"
    elif year <= 2015:
        return "2000-2015 dialogue systems / HCI"
    elif year <= 2022:
        return "2016-2022 chatbot + assistant growth"
    else:
        return "2023-2025 generative AI / chatbot boom"


def parse_terms(value):
    if pd.isna(value):
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]


# -----------------------------
# Load scored data
# -----------------------------
print("Loading scored English abstract dataset...")
df = pd.read_parquet(INPUT)

df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df = df.dropna(subset=["publication_year"]).copy()
df["publication_year"] = df["publication_year"].astype(int)

df["analysis_period"] = df["publication_year"].apply(assign_period)
df["detected_terms_list"] = df["detected_terms_in_title_abstract"].apply(parse_terms)

print(f"Loaded {len(df):,} English abstract records")


# -----------------------------
# Explode keyword terms
# -----------------------------
term_df = df.explode("detected_terms_list").dropna(subset=["detected_terms_list"]).copy()
term_df = term_df[term_df["detected_terms_list"] != ""].copy()
term_df["keyword_group"] = term_df["detected_terms_list"].map(TERM_TO_GROUP)

term_df = term_df.dropna(subset=["keyword_group"]).copy()


# -----------------------------
# Summary tables
# -----------------------------
corpus_summary = pd.DataFrame({
    "metric": [
        "English abstract records scored",
        "Minimum publication year",
        "Maximum publication year",
        "Mean VADER compound score",
        "Positive records",
        "Neutral records",
        "Negative records",
    ],
    "value": [
        len(df),
        df["publication_year"].min(),
        df["publication_year"].max(),
        df["sentiment_compound"].mean(),
        (df["sentiment_label"] == "positive").sum(),
        (df["sentiment_label"] == "neutral").sum(),
        (df["sentiment_label"] == "negative").sum(),
    ],
})
corpus_summary.to_csv(TABLES / "corpus_summary.csv", index=False)

sentiment_by_period = (
    df.groupby(["analysis_period", "sentiment_label"])
    .size()
    .reset_index(name="n")
)
sentiment_by_period["percent_within_period"] = (
    sentiment_by_period["n"]
    / sentiment_by_period.groupby("analysis_period")["n"].transform("sum")
    * 100
)
sentiment_by_period.to_csv(TABLES / "sentiment_label_by_period.csv", index=False)

sentiment_score_by_period = (
    df.groupby("analysis_period")
    .agg(
        n=("sentiment_compound", "size"),
        mean_sentiment=("sentiment_compound", "mean"),
        median_sentiment=("sentiment_compound", "median"),
    )
    .reset_index()
)
sentiment_score_by_period.to_csv(TABLES / "sentiment_score_by_period.csv", index=False)

sentiment_by_group_period = (
    term_df.groupby(["analysis_period", "keyword_group"])
    .agg(
        n=("sentiment_compound", "size"),
        mean_sentiment=("sentiment_compound", "mean"),
        median_sentiment=("sentiment_compound", "median"),
    )
    .reset_index()
)
sentiment_by_group_period.to_csv(TABLES / "sentiment_by_keyword_group_and_period.csv", index=False)

emotion_by_period = (
    df.groupby("analysis_period")[EMOTIONS]
    .mean()
    .reset_index()
)
emotion_by_period.to_csv(TABLES / "emotion_by_period.csv", index=False)

emotion_by_year = (
    df.groupby("publication_year")[EMOTIONS]
    .mean()
    .reset_index()
    .sort_values("publication_year")
)
emotion_by_year.to_csv(TABLES / "emotion_by_year.csv", index=False)

keyword_group_counts = (
    term_df.groupby(["analysis_period", "keyword_group"])
    .size()
    .reset_index(name="n")
)
keyword_group_counts["percent_within_period"] = (
    keyword_group_counts["n"]
    / keyword_group_counts.groupby("analysis_period")["n"].transform("sum")
    * 100
)
keyword_group_counts.to_csv(TABLES / "keyword_group_counts_by_period.csv", index=False)


# -----------------------------
# Manual validation samples
# -----------------------------
sample_cols = [c for c in [
    "openalex_id",
    "doi",
    "publication_year",
    "title",
    "abstract",
    "detected_terms_in_title_abstract",
    "sentiment_compound",
    "sentiment_label",
] + EMOTIONS if c in df.columns]

high_positive = df.sort_values("sentiment_compound", ascending=False).head(30)
high_negative = df.sort_values("sentiment_compound", ascending=True).head(30)
neutral = df.assign(abs_sentiment=df["sentiment_compound"].abs()).sort_values("abs_sentiment").head(30)

high_positive[sample_cols].to_csv(VALIDATION / "high_positive_abstracts.csv", index=False)
neutral[sample_cols].to_csv(VALIDATION / "neutral_abstracts.csv", index=False)
high_negative[sample_cols].to_csv(VALIDATION / "high_negative_abstracts.csv", index=False)


# -----------------------------
# Figure 1: Keyword group percent by period
# -----------------------------
pivot_keyword = keyword_group_counts.pivot(
    index="analysis_period",
    columns="keyword_group",
    values="percent_within_period",
).fillna(0)

pivot_keyword.plot(kind="bar", figsize=(12, 6))
plt.title("Keyword Group Share by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Percent of Keyword Matches Within Period")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "01_keyword_group_percent_by_period.png", dpi=200)
plt.close()


# -----------------------------
# Figure 2: Sentiment label percent by period
# -----------------------------
pivot_sentiment = sentiment_by_period.pivot(
    index="analysis_period",
    columns="sentiment_label",
    values="percent_within_period",
).fillna(0)

pivot_sentiment.plot(kind="bar", figsize=(12, 6))
plt.title("Sentiment Label Distribution by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Percent of Records Within Period")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "02_sentiment_label_percent_by_period.png", dpi=200)
plt.close()


# -----------------------------
# Figure 3: Emotion framing over time
# -----------------------------
emotion_by_year_smooth = emotion_by_year.copy()
emotion_by_year_smooth[EMOTIONS] = emotion_by_year_smooth[EMOTIONS].rolling(
    window=3,
    min_periods=1,
).mean()

plt.figure(figsize=(12, 6))
for emotion in EMOTIONS:
    plt.plot(
        emotion_by_year_smooth["publication_year"],
        emotion_by_year_smooth[emotion],
        marker="o",
        linewidth=1.5,
        label=emotion,
    )

plt.title("Emotion Framing Over Time in Conversational AI Abstracts")
plt.xlabel("Publication Year")
plt.ylabel("Mean NRCLex Emotion Score")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "03_emotion_framing_over_time.png", dpi=200)
plt.close()


# -----------------------------
# Figure 4: Mean sentiment by keyword group and period
# -----------------------------
pivot_group_sentiment = sentiment_by_group_period.pivot(
    index="analysis_period",
    columns="keyword_group",
    values="mean_sentiment",
).fillna(0)

pivot_group_sentiment.plot(kind="bar", figsize=(12, 6))
plt.title("Mean Sentiment by Keyword Group and Time Period")
plt.xlabel("Time Period")
plt.ylabel("Mean VADER Compound Score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "04_mean_sentiment_by_keyword_group_period.png", dpi=200)
plt.close()


# -----------------------------
# Write short report
# -----------------------------
report = f"""
Final Sentiment/Framing Analysis Summary

Input file:
{INPUT}

Records analyzed:
{len(df):,} English records with abstracts

Publication year range:
{df['publication_year'].min()}-{df['publication_year'].max()}

Sentiment label counts:
{df['sentiment_label'].value_counts().to_string()}

Mean emotion scores:
{df[EMOTIONS].mean().sort_values(ascending=False).to_string()}

Final interpretation:
This analysis treats VADER and NRCLex results as sentiment/framing indicators,
not as perfect measures of author attitude. The goal is to compare how emotional
and sentiment framing changes across time periods and keyword groups as the field
moves from older dialogue-system terminology toward newer chatbot and
conversational-AI terminology.
"""

(REPORTS / "sentiment_framing_summary.txt").write_text(report)

print("Done.")
print(f"Final outputs saved to: {OUTPUT}")
