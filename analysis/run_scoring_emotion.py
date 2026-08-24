import pandas as pd
import nltk
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

EMOTIONS = ["fear", "anticipation", "trust", "joy", "sadness", "anger"]

def score_emotions(text):
    if not isinstance(text, str) or not text.strip():
        return {e: 0.0 for e in EMOTIONS}
    nrc = NRCLex()
    nrc.load_raw_text(text)
    freqs = nrc.affect_frequencies
    return {e: freqs.get(e, 0.0) for e in EMOTIONS}

analyzer = SentimentIntensityAnalyzer()

def score_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None
    return analyzer.polarity_scores(text)["compound"]

def label_sentiment(score):
    if score is None:
        return None
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"

print("Loading data...")
df = pd.read_parquet("final_openalex_pipeline/data/processed/openalex_clean_corpus.parquet")
df_en = df[(df["language"] == "en") & (df["has_abstract"] == True)].copy()
print(f"{len(df_en)} rows to score")

tqdm.pandas(desc="VADER sentiment")
df_en["sentiment_compound"] = df_en["abstract"].progress_apply(score_sentiment)
df_en["sentiment_label"] = df_en["sentiment_compound"].apply(label_sentiment)

tqdm.pandas(desc="NRCLex emotions")
emotion_scores = df_en["abstract"].progress_apply(score_emotions).apply(pd.Series)
df_en = pd.concat([df_en, emotion_scores], axis=1)

print()
print("--- Sentiment label counts ---")
print(df_en["sentiment_label"].value_counts())
print()
print("--- Mean emotion scores ---")
print(df_en[EMOTIONS].mean().sort_values(ascending=False))

output_path = "final_openalex_pipeline/data/processed/df_en_scored.parquet"
df_en.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")