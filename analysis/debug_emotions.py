import os
import pandas as pd

_original_stat = os.stat

def _safe_stat(path, *args, **kwargs):
    try:
        return _original_stat(path, *args, **kwargs)
    except OSError as e:
        raise FileNotFoundError(e.errno, "No such file or directory", path) from None

os.stat = _safe_stat

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nrclex import NRCLex

EMOTIONS = ["fear", "anticipation", "trust", "joy", "sadness", "anger"]

def score_emotions(text):
    if not isinstance(text, str) or not text.strip():
        return {e: 0.0 for e in EMOTIONS}
    freqs = NRCLex(text).affect_frequencies
    return {e: freqs.get(e, 0.0) for e in EMOTIONS}

print("Loading data...")
df = pd.read_parquet("final_openalex_pipeline/data/processed/openalex_clean_corpus.parquet")
df_en = df[(df["language"] == "en") & (df["has_abstract"] == True)].copy()

print(f"Testing on a small sample of 20 rows first...")
sample = df_en["abstract"].head(20)

for i, text in enumerate(sample):
    try:
        scores = score_emotions(text)
        print(f"Row {i}: OK -> {scores}")
    except Exception as e:
        print(f"Row {i}: FAILED -> {type(e).__name__}: {str(e)[:200]}")

print("Done with sample test.")