# Final Sentiment/Framing Analysis

This folder contains the final cleaned analysis for the OpenAlex conversational AI corpus.

## Goal

The goal is to study how sentiment and emotional framing change over time as conversational AI research shifts from older dialogue-system terminology to newer chatbot and conversational-AI terminology.

## Input

The analysis uses:

- `final_openalex_pipeline/data/processed/df_en_scored.parquet`

This file contains English-language records with abstracts scored using VADER sentiment and NRCLex emotion categories.

## Main scripts

- `scripts/run_final_sentiment_framing_analysis.py`
- `scripts/polish_final_figures.py`

The first script creates the main tables, validation samples, and figures.  
The second script polishes the final figures for poster use.

## Outputs

Final outputs are saved in:

- `outputs/tables/`
- `outputs/figures/`
- `outputs/validation_samples/`
- `outputs/reports/`

## Main final figures

1. Keyword group share by time period
2. Sentiment label distribution by time period
3. Emotion framing over time
4. Mean sentiment by keyword group and time period

## Interpretation note

VADER and NRCLex scores are treated as sentiment/framing indicators, not perfect measures of author attitude. Because the analysis uses academic abstracts, positive language may reflect technical claims such as improvement, effectiveness, or support.
