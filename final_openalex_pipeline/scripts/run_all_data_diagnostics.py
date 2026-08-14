from __future__ import annotations

"""
All-in-one diagnostic test for the OpenAlex conversational AI corpus.

This script does NOT download data.

It reads the processed OpenAlex files and creates:
1. quality-control checks
2. grouped CSV outputs
3. visual charts
4. manual review samples
5. a plain-English diagnostic report

The goal is to understand what the dataset contains before doing the
actual final analysis.
"""

from pathlib import Path
import re
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. Paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_openalex"
PROCESSED_DIR = ROOT / "data" / "processed"

OUT_DIR = PROCESSED_DIR / "current_data_tests"

GROUP_DIRS = {
    "progress": OUT_DIR / "01_download_progress",
    "quality": OUT_DIR / "02_quality_checks",
    "overview": OUT_DIR / "03_corpus_overview",
    "terms": OUT_DIR / "04_term_patterns",
    "metadata": OUT_DIR / "05_metadata_patterns",
    "samples": OUT_DIR / "06_manual_review_samples",
    "figures": OUT_DIR / "figures",
}

for folder in GROUP_DIRS.values():
    folder.mkdir(parents=True, exist_ok=True)

START_YEAR = 1975
END_YEAR = 2025

SEARCH_TERMS = [
    "chatbot",
    "chatterbot",
    "conversational agent",
    "embodied conversational agent",
    "dialogue system",
    "dialog system",
    "spoken dialogue system",
    "virtual assistant",
    "voice assistant",
    "conversational AI",
    "natural language interface",
]

EXPECTED_TASKS = (END_YEAR - START_YEAR + 1) * len(SEARCH_TERMS)


# ============================================================
# 2. Helper functions
# ============================================================

def safe_filename(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["\']', "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_title(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def add_result(results: list[dict], test: str, status: str, detail: str) -> None:
    results.append(
        {
            "test": test,
            "status": status,
            "detail": detail,
        }
    )


def split_semicolon_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    rows = []

    if column_name not in df.columns:
        return pd.DataFrame(columns=["openalex_id", "publication_year", "item"])

    for _, row in df.iterrows():
        value = row.get(column_name)

        if pd.isna(value):
            continue

        items = [x.strip() for x in str(value).split(";") if x.strip()]

        for item in items:
            rows.append(
                {
                    "openalex_id": row.get("openalex_id"),
                    "publication_year": row.get("publication_year"),
                    "item": item,
                }
            )

    return pd.DataFrame(rows)


def create_decade(year) -> str:
    if pd.isna(year):
        return "missing"

    year = int(year)
    return f"{year // 10 * 10}s"

def create_analysis_period(year) -> str:
    """Group years into historical periods for trend analysis."""

    if pd.isna(year):
        return "missing"

    year = int(year)

    if year <= 1999:
        return "1976-1999 early dialogue / natural language interface"
    elif year <= 2015:
        return "2000-2015 dialogue systems / HCI / virtual agents"
    elif year <= 2022:
        return "2016-2022 chatbot and assistant growth"
    else:
        return "2023-2025 generative AI / chatbot explosion"

def save_line_chart(df, x_col, y_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(df[x_col], df[y_col], marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_bar_chart(df, x_col, y_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(12, 6))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_horizontal_bar_chart(df, label_col, value_col, title, xlabel, output_path, top_n=25):
    plot_df = df.head(top_n).copy()
    plot_df = plot_df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(plot_df[label_col].astype(str), plot_df[value_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# 3. Main diagnostics
# ============================================================

def main() -> None:
    results = []
    report_lines = []

    clean_path = PROCESSED_DIR / "openalex_clean_corpus.csv"
    abstract_path = PROCESSED_DIR / "openalex_abstract_subset.csv"
    candidates_path = PROCESSED_DIR / "openalex_candidates_all.csv"
    yearly_path = PROCESSED_DIR / "openalex_yearly_summary.csv"
    term_path = PROCESSED_DIR / "openalex_term_summary.csv"
    type_path = PROCESSED_DIR / "openalex_type_summary.csv"

    clean = read_csv(clean_path)
    abstracts = read_csv(abstract_path)
    candidates = read_csv(candidates_path)
    yearly = read_csv(yearly_path)
    term_summary = read_csv(term_path)
    type_summary = read_csv(type_path)

    if clean is None:
        raise FileNotFoundError(
            "Missing openalex_clean_corpus.csv. Run:\n"
            "python scripts/run_full_openalex_search.py --process --validate"
        )

    clean["publication_year"] = pd.to_numeric(clean["publication_year"], errors="coerce")
    clean["has_abstract_bool"] = as_bool(clean["has_abstract"])
    clean["decade"] = clean["publication_year"].apply(create_decade)
    clean["analysis_period"] = clean["publication_year"].apply(create_analysis_period)
    if "title_norm" not in clean.columns:
        clean["title_norm"] = clean["title"].apply(normalize_title)

    # ------------------------------------------------------------
    # A. Download progress tests
    # ------------------------------------------------------------

    progress_rows = []
    raw_term_year_rows = []

    done_count = 0
    jsonl_count = 0

    tmp_count = len(list(RAW_DIR.rglob("*.tmp")))

    for year in range(START_YEAR, END_YEAR + 1):
        year_done = 0
        year_jsonl = 0
        year_raw_rows = 0

        for term in SEARCH_TERMS:
            file_stem = safe_filename(term)
            jsonl_path = RAW_DIR / str(year) / f"{file_stem}.jsonl"
            done_path = RAW_DIR / str(year) / f"{file_stem}.done"

            has_jsonl = jsonl_path.exists()
            has_done = done_path.exists()
            raw_rows = count_lines(jsonl_path) if has_jsonl else 0

            if has_jsonl:
                jsonl_count += 1
                year_jsonl += 1

            if has_done:
                done_count += 1
                year_done += 1

            year_raw_rows += raw_rows

            raw_term_year_rows.append(
                {
                    "year": year,
                    "term": term,
                    "has_jsonl": has_jsonl,
                    "has_done": has_done,
                    "raw_rows": raw_rows,
                    "complete": has_jsonl and has_done,
                }
            )

        progress_rows.append(
            {
                "year": year,
                "done_files": year_done,
                "jsonl_files": year_jsonl,
                "raw_rows": year_raw_rows,
                "complete_all_terms": year_done == len(SEARCH_TERMS),
            }
        )

    raw_term_year = pd.DataFrame(raw_term_year_rows)
    raw_progress = pd.DataFrame(progress_rows)

    raw_term_year.to_csv(GROUP_DIRS["progress"] / "raw_rows_by_term_year.csv", index=False)
    raw_progress.to_csv(GROUP_DIRS["progress"] / "raw_progress_by_year.csv", index=False)

    incomplete_years = raw_progress[~raw_progress["complete_all_terms"]]
    incomplete_years.to_csv(GROUP_DIRS["progress"] / "incomplete_years.csv", index=False)

    progress_pct = done_count / EXPECTED_TASKS * 100

    add_result(
        results,
        "Raw download progress",
        "PASS" if done_count == EXPECTED_TASKS else "INFO",
        f"{done_count}/{EXPECTED_TASKS} term-year tasks complete ({progress_pct:.2f}%).",
    )

    add_result(
        results,
        "Temporary files",
        "PASS" if tmp_count == 0 else "WARN",
        f"{tmp_count} unfinished .tmp files found.",
    )

    add_result(
        results,
        "JSONL/DONE consistency",
        "PASS" if jsonl_count == done_count else "WARN",
        f"{jsonl_count} JSONL files and {done_count} DONE markers.",
    )

    # ------------------------------------------------------------
    # B. Basic corpus overview
    # ------------------------------------------------------------

    total_clean = len(clean)
    records_with_abstracts = int(clean["has_abstract_bool"].sum())
    abstract_coverage = records_with_abstracts / total_clean * 100 if total_clean else 0

    min_year = int(clean["publication_year"].min())
    max_year = int(clean["publication_year"].max())

    raw_candidate_rows = len(candidates) if candidates is not None else None

    corpus_summary = pd.DataFrame(
        [
            {
                "final_clean_records": total_clean,
                "records_with_abstracts": records_with_abstracts,
                "abstract_coverage_percent": round(abstract_coverage, 2),
                "year_min": min_year,
                "year_max": max_year,
                "raw_candidate_rows": raw_candidate_rows,
                "candidate_keep_percent": round(total_clean / raw_candidate_rows * 100, 2)
                if raw_candidate_rows
                else None,
            }
        ]
    )

    corpus_summary.to_csv(GROUP_DIRS["overview"] / "corpus_summary.csv", index=False)

    add_result(
        results,
        "Clean corpus file",
        "PASS",
        f"Loaded {total_clean:,} clean records.",
    )

    add_result(
        results,
        "Year range",
        "PASS" if min_year >= START_YEAR and max_year <= END_YEAR else "WARN",
        f"Clean corpus year range is {min_year}–{max_year}.",
    )

    add_result(
        results,
        "Abstract coverage",
        "INFO",
        f"{records_with_abstracts:,}/{total_clean:,} records have abstracts ({abstract_coverage:.2f}%).",
    )

    # ------------------------------------------------------------
    # C. Quality checks
    # ------------------------------------------------------------

    required_columns = [
        "openalex_id",
        "title",
        "abstract",
        "publication_year",
        "has_abstract",
        "matches_title_or_abstract",
        "detected_terms_in_title_abstract",
    ]

    missing_cols = [col for col in required_columns if col not in clean.columns]

    add_result(
        results,
        "Required columns",
        "PASS" if not missing_cols else "FAIL",
        "All required columns are present." if not missing_cols else f"Missing columns: {missing_cols}",
    )

    missing_titles = clean["title"].fillna("").str.strip().eq("").sum()

    add_result(
        results,
        "Missing titles",
        "PASS" if missing_titles == 0 else "FAIL",
        f"{missing_titles:,} rows have missing titles.",
    )

    match_values = as_bool(clean["matches_title_or_abstract"])
    nonmatching = int((~match_values).sum())

    add_result(
        results,
        "Title/abstract boundary",
        "PASS" if nonmatching == 0 else "FAIL",
        f"{nonmatching:,} clean rows do not match any Search B term in title or abstract.",
    )

    duplicate_ids = clean["openalex_id"].duplicated().sum()

    add_result(
        results,
        "Duplicate OpenAlex IDs",
        "PASS" if duplicate_ids == 0 else "FAIL",
        f"{duplicate_ids:,} duplicate OpenAlex IDs found.",
    )

    doi_norm = clean["doi"].fillna("").astype(str).str.lower().str.strip()
    duplicate_dois = doi_norm[doi_norm.ne("")].duplicated().sum()

    add_result(
        results,
        "Duplicate DOIs",
        "PASS" if duplicate_dois == 0 else "FAIL",
        f"{duplicate_dois:,} duplicate non-empty DOIs found.",
    )

    duplicate_title_year = clean.duplicated(subset=["title_norm", "publication_year"]).sum()

    add_result(
        results,
        "Duplicate title/year",
        "PASS" if duplicate_title_year == 0 else "WARN",
        f"{duplicate_title_year:,} duplicate normalized title/year pairs found.",
    )

    duplicate_review = clean[
        clean.duplicated(subset=["title_norm", "publication_year"], keep=False)
    ].sort_values(["publication_year", "title_norm", "cited_by_count"], ascending=[True, True, False])

    duplicate_review.to_csv(GROUP_DIRS["quality"] / "duplicate_title_year_review.csv", index=False)

    if "abstract_word_count" in clean.columns:
        clean["abstract_word_count"] = pd.to_numeric(clean["abstract_word_count"], errors="coerce").fillna(0)
        short_abs = clean[(clean["abstract_word_count"] > 0) & (clean["abstract_word_count"] < 20)].copy()
        short_abs.to_csv(GROUP_DIRS["quality"] / "short_abstracts_under_20_words.csv", index=False)

        add_result(
            results,
            "Short abstracts",
            "PASS" if len(short_abs) == 0 else "WARN",
            f"{len(short_abs):,} abstracts are under 20 words.",
        )

    if yearly is not None:
        yearly_total = int(yearly["total_records"].sum())

        add_result(
            results,
            "Yearly summary total",
            "PASS" if yearly_total == total_clean else "FAIL",
            f"Yearly summary total is {yearly_total:,}; clean corpus total is {total_clean:,}.",
        )

    # ------------------------------------------------------------
    # D. Corpus overview tables and charts
    # ------------------------------------------------------------

    records_by_year = (
        clean.groupby("publication_year")
        .size()
        .reset_index(name="records")
        .sort_values("publication_year")
    )

    records_by_year["cumulative_records"] = records_by_year["records"].cumsum()
    records_by_year.to_csv(GROUP_DIRS["overview"] / "records_by_year.csv", index=False)

    save_line_chart(
        records_by_year,
        "publication_year",
        "records",
        "Clean Records by Publication Year",
        "Publication year",
        "Records",
        GROUP_DIRS["figures"] / "records_by_year.png",
    )

    save_line_chart(
        records_by_year,
        "publication_year",
        "cumulative_records",
        "Cumulative Clean Records Over Time",
        "Publication year",
        "Cumulative records",
        GROUP_DIRS["figures"] / "cumulative_records_by_year.png",
    )

    records_by_decade = (
        clean.groupby("decade")
        .size()
        .reset_index(name="records")
        .sort_values("decade")
    )

    records_by_decade.to_csv(GROUP_DIRS["overview"] / "records_by_decade.csv", index=False)

    save_bar_chart(
        records_by_decade,
        "decade",
        "records",
        "Clean Records by Decade",
        "Decade",
        "Records",
        GROUP_DIRS["figures"] / "records_by_decade.png",
    )

    abstract_by_year = (
        clean.groupby("publication_year")
        .agg(
            total_records=("openalex_id", "count"),
            records_with_abstract=("has_abstract_bool", "sum"),
        )
        .reset_index()
        .sort_values("publication_year")
    )

    abstract_by_year["abstract_coverage_percent"] = (
        abstract_by_year["records_with_abstract"] / abstract_by_year["total_records"] * 100
    ).round(2)

    abstract_by_year.to_csv(GROUP_DIRS["overview"] / "abstract_coverage_by_year.csv", index=False)

    save_line_chart(
        abstract_by_year,
        "publication_year",
        "abstract_coverage_percent",
        "Abstract Coverage by Publication Year",
        "Publication year",
        "Percent with abstract",
        GROUP_DIRS["figures"] / "abstract_coverage_by_year.png",
    )

    # ------------------------------------------------------------
    # D2. Basic corpus trend analysis
    # ------------------------------------------------------------

    trend_dir = GROUP_DIRS["overview"] / "basic_corpus_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)

    # Records by year with raw counts and percentages
    records_by_year_trend = (
        clean.groupby("publication_year")
        .size()
        .reset_index(name="raw_records")
        .sort_values("publication_year")
    )

    records_by_year_trend["percent_of_corpus"] = (
        records_by_year_trend["raw_records"] / total_clean * 100
    ).round(3)

    records_by_year_trend.to_csv(
        trend_dir / "records_by_year_raw_and_percent.csv",
        index=False,
    )

    # Records by decade with raw counts and percentages
    records_by_decade_trend = (
        clean.groupby("decade")
        .size()
        .reset_index(name="raw_records")
        .sort_values("decade")
    )

    records_by_decade_trend["percent_of_corpus"] = (
        records_by_decade_trend["raw_records"] / total_clean * 100
    ).round(3)

    records_by_decade_trend.to_csv(
        trend_dir / "records_by_decade_raw_and_percent.csv",
        index=False,
    )

    # Records by analysis period with abstract coverage
    period_summary = (
        clean.groupby("analysis_period")
        .agg(
            raw_records=("openalex_id", "count"),
            records_with_abstract=("has_abstract_bool", "sum"),
        )
        .reset_index()
    )

    period_summary["percent_of_corpus"] = (
        period_summary["raw_records"] / total_clean * 100
    ).round(3)

    period_summary["abstract_coverage_percent"] = (
        period_summary["records_with_abstract"] / period_summary["raw_records"] * 100
    ).round(2)

    period_summary.to_csv(
        trend_dir / "records_by_analysis_period_with_abstract_coverage.csv",
        index=False,
    )

    # Work type by period with raw counts and percentages within each period
    work_type_by_period = (
        clean.groupby(["analysis_period", "type"])
        .size()
        .reset_index(name="raw_records")
        .sort_values(["analysis_period", "raw_records"], ascending=[True, False])
    )

    period_totals = (
        clean.groupby("analysis_period")
        .size()
        .reset_index(name="period_total")
    )

    work_type_by_period = work_type_by_period.merge(
        period_totals,
        on="analysis_period",
        how="left",
    )

    work_type_by_period["percent_within_period"] = (
        work_type_by_period["raw_records"] / work_type_by_period["period_total"] * 100
    ).round(2)

    work_type_by_period.to_csv(
        trend_dir / "work_type_by_period_raw_and_percent.csv",
        index=False,
    )

    # Abstract coverage by year is already saved, but save a trend-focused version too
    abstract_by_year.to_csv(
        trend_dir / "abstract_coverage_by_year.csv",
        index=False,
    )

    # Search terms by year and period with raw counts and percentages
    trend_term_rows = split_semicolon_column(clean, "detected_terms_in_title_abstract")

    if not trend_term_rows.empty:
        trend_term_rows = trend_term_rows.merge(
            clean[["openalex_id", "analysis_period"]],
            on="openalex_id",
            how="left",
        )

        year_totals = (
            clean.groupby("publication_year")
            .size()
            .reset_index(name="year_total")
        )

        term_by_year_trend = (
            trend_term_rows.groupby(["publication_year", "item"])
            .size()
            .reset_index(name="raw_records")
            .rename(columns={"item": "search_term"})
        )

        term_by_year_trend = term_by_year_trend.merge(
            year_totals,
            on="publication_year",
            how="left",
        )

        term_by_year_trend["percent_within_year"] = (
            term_by_year_trend["raw_records"] / term_by_year_trend["year_total"] * 100
        ).round(2)

        term_by_year_trend.to_csv(
            trend_dir / "top_search_terms_by_year_raw_and_percent.csv",
            index=False,
        )

        period_totals = (
            clean.groupby("analysis_period")
            .size()
            .reset_index(name="period_total")
        )

        term_by_period_trend = (
            trend_term_rows.groupby(["analysis_period", "item"])
            .size()
            .reset_index(name="raw_records")
            .rename(columns={"item": "search_term"})
        )

        term_by_period_trend = term_by_period_trend.merge(
            period_totals,
            on="analysis_period",
            how="left",
        )

        term_by_period_trend["percent_within_period"] = (
            term_by_period_trend["raw_records"] / term_by_period_trend["period_total"] * 100
        ).round(2)

        term_by_period_trend = term_by_period_trend.sort_values(
            ["analysis_period", "raw_records"],
            ascending=[True, False],
        )

        term_by_period_trend.to_csv(
            trend_dir / "top_search_terms_by_period_raw_and_percent.csv",
            index=False,
        )

        # Save only top 5 terms per period for easier review
        top_terms_by_period = (
            term_by_period_trend.groupby("analysis_period")
            .head(5)
            .reset_index(drop=True)
        )

        top_terms_by_period.to_csv(
            trend_dir / "top_5_search_terms_by_period.csv",
            index=False,
        )

    add_result(
        results,
        "Basic corpus trend analysis",
        "PASS",
        "Saved raw counts and percentages for year, decade, work type, abstract coverage, and search terms by year/period.",
    )

    # ------------------------------------------------------------
    # E. Term-pattern tables and charts
    # ------------------------------------------------------------

    term_rows = split_semicolon_column(clean, "detected_terms_in_title_abstract")

    if not term_rows.empty:
        term_counts = (
            term_rows["item"]
            .value_counts()
            .rename_axis("term")
            .reset_index(name="records")
        )

        term_counts.to_csv(GROUP_DIRS["terms"] / "detected_term_counts.csv", index=False)

        save_horizontal_bar_chart(
            term_counts,
            "term",
            "records",
            "Detected Search Terms in Clean Corpus",
            "Records",
            GROUP_DIRS["figures"] / "detected_term_counts.png",
            top_n=20,
        )

        term_by_year = (
            term_rows.groupby(["publication_year", "item"])
            .size()
            .reset_index(name="records")
            .rename(columns={"item": "term"})
            .sort_values(["publication_year", "term"])
        )

        term_by_year.to_csv(GROUP_DIRS["terms"] / "term_counts_by_year.csv", index=False)

        term_by_decade = term_by_year.copy()
        term_by_decade["decade"] = term_by_decade["publication_year"].apply(create_decade)

        term_by_decade = (
            term_by_decade.groupby(["decade", "term"])["records"]
            .sum()
            .reset_index()
            .sort_values(["decade", "records"], ascending=[True, False])
        )

        term_by_decade.to_csv(GROUP_DIRS["terms"] / "term_counts_by_decade.csv", index=False)

        top_terms = term_counts.head(8)["term"].tolist()

        plt.figure(figsize=(14, 7))
        for term in top_terms:
            sub = term_by_year[term_by_year["term"] == term].sort_values("publication_year")
            plt.plot(sub["publication_year"], sub["records"], marker="o", linewidth=1, label=term)

        plt.title("Top Detected Search Terms Over Time")
        plt.xlabel("Publication year")
        plt.ylabel("Records")
        plt.legend()
        plt.tight_layout()
        plt.savefig(GROUP_DIRS["figures"] / "top_terms_over_time.png", dpi=200)
        plt.close()

    # ------------------------------------------------------------
    # F. Metadata-pattern tables and charts
    # ------------------------------------------------------------

    type_counts = (
        clean["type"]
        .fillna("missing")
        .value_counts()
        .rename_axis("type")
        .reset_index(name="records")
    )

    type_counts.to_csv(GROUP_DIRS["metadata"] / "work_type_counts.csv", index=False)

    save_horizontal_bar_chart(
        type_counts,
        "type",
        "records",
        "OpenAlex Work Types",
        "Records",
        GROUP_DIRS["figures"] / "work_type_counts.png",
        top_n=20,
    )

    if "venue" in clean.columns:
        venue_counts = (
            clean["venue"]
            .fillna("missing")
            .value_counts()
            .rename_axis("venue")
            .reset_index(name="records")
        )

        venue_counts.to_csv(GROUP_DIRS["metadata"] / "top_venues.csv", index=False)

        save_horizontal_bar_chart(
            venue_counts,
            "venue",
            "records",
            "Top Venues",
            "Records",
            GROUP_DIRS["figures"] / "top_venues.png",
            top_n=25,
        )

    topic_rows = split_semicolon_column(clean, "topics")

    if not topic_rows.empty:
        topic_counts = (
            topic_rows["item"]
            .value_counts()
            .rename_axis("topic")
            .reset_index(name="records")
        )

        topic_counts.to_csv(GROUP_DIRS["metadata"] / "top_topics.csv", index=False)

        save_horizontal_bar_chart(
            topic_counts,
            "topic",
            "records",
            "Top OpenAlex Topics",
            "Records",
            GROUP_DIRS["figures"] / "top_topics.png",
            top_n=25,
        )

    keyword_rows = split_semicolon_column(clean, "keywords")

    if not keyword_rows.empty:
        keyword_counts = (
            keyword_rows["item"]
            .value_counts()
            .rename_axis("keyword")
            .reset_index(name="records")
        )

        keyword_counts.to_csv(GROUP_DIRS["metadata"] / "top_keywords.csv", index=False)

        save_horizontal_bar_chart(
            keyword_counts,
            "keyword",
            "records",
            "Top OpenAlex Keywords",
            "Records",
            GROUP_DIRS["figures"] / "top_keywords.png",
            top_n=25,
        )

    # ------------------------------------------------------------
    # G. Candidate kept/rejected summary
    # ------------------------------------------------------------

    if candidates is not None and "matches_title_or_abstract" in candidates.columns:
        candidate_match = as_bool(candidates["matches_title_or_abstract"])

        candidate_summary = pd.DataFrame(
            [
                {
                    "category": "kept_by_title_abstract_boundary",
                    "records": int(candidate_match.sum()),
                },
                {
                    "category": "rejected_by_title_abstract_boundary",
                    "records": int((~candidate_match).sum()),
                },
            ]
        )

        candidate_summary.to_csv(GROUP_DIRS["quality"] / "candidate_boundary_summary.csv", index=False)

        save_bar_chart(
            candidate_summary,
            "category",
            "records",
            "Candidate Records Kept vs Rejected",
            "Boundary result",
            "Records",
            GROUP_DIRS["figures"] / "candidate_kept_vs_rejected.png",
        )

        rejected_sample = candidates[~candidate_match].sample(
            n=min(1000, int((~candidate_match).sum())),
            random_state=42,
        )

        rejected_sample.to_csv(GROUP_DIRS["quality"] / "candidate_rejected_boundary_sample_1000.csv", index=False)

    # ------------------------------------------------------------
    # H. Manual review samples
    # ------------------------------------------------------------

    sample_cols = [
        col
        for col in [
            "openalex_id",
            "title",
            "publication_year",
            "type",
            "venue",
            "retrieved_by_terms",
            "detected_terms_in_title_abstract",
            "abstract",
        ]
        if col in clean.columns
    ]

    clean.sample(n=min(50, len(clean)), random_state=42)[sample_cols].to_csv(
        GROUP_DIRS["samples"] / "manual_relevance_sample_50.csv",
        index=False,
    )

    recent = clean[clean["publication_year"] >= 2020].copy()
    if len(recent) > 0:
        recent.sample(n=min(50, len(recent)), random_state=42)[sample_cols].to_csv(
            GROUP_DIRS["samples"] / "recent_sample_2020_2025.csv",
            index=False,
        )

    older = clean[clean["publication_year"] <= 2000].copy()
    if len(older) > 0:
        older.sample(n=min(50, len(older)), random_state=42)[sample_cols].to_csv(
            GROUP_DIRS["samples"] / "older_sample_1976_2000.csv",
            index=False,
        )

    # ------------------------------------------------------------
    # I. Save reports
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df.to_csv(GROUP_DIRS["quality"] / "diagnostic_test_results.csv", index=False)

    report_lines.append("All-in-One OpenAlex Data Diagnostic Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("Main summary:")
    report_lines.append(f"- Final clean records: {total_clean:,}")
    report_lines.append(f"- Records with abstracts: {records_with_abstracts:,}")
    report_lines.append(f"- Abstract coverage: {abstract_coverage:.2f}%")
    report_lines.append(f"- Year range: {min_year}–{max_year}")
    if raw_candidate_rows is not None:
        report_lines.append(f"- Raw candidate rows: {raw_candidate_rows:,}")
        report_lines.append(
            f"- Candidate rows kept after cleaning: {total_clean / raw_candidate_rows * 100:.2f}%"
        )
    report_lines.append(f"- Raw download progress: {done_count}/{EXPECTED_TASKS} term-year tasks")
    report_lines.append("")
    report_lines.append("Quality checks:")

    for row in results:
        report_lines.append(f"- [{row['status']}] {row['test']}: {row['detail']}")

    report_lines.append("")
    report_lines.append("Grouped outputs created:")
    report_lines.append(f"- Download progress CSVs: {GROUP_DIRS['progress']}")
    report_lines.append(f"- Quality-check CSVs: {GROUP_DIRS['quality']}")
    report_lines.append(f"- Corpus overview CSVs: {GROUP_DIRS['overview']}")
    report_lines.append(f"- Term-pattern CSVs: {GROUP_DIRS['terms']}")
    report_lines.append(f"- Metadata-pattern CSVs: {GROUP_DIRS['metadata']}")
    report_lines.append(f"- Manual review samples: {GROUP_DIRS['samples']}")
    report_lines.append(f"- Figures: {GROUP_DIRS['figures']}")
    report_lines.append("")
    report_lines.append("How to use this before analysis:")
    report_lines.append(
        "- Check records_by_year, top_terms_over_time, abstract_coverage_by_year, "
        "and manual samples before deciding the final analysis design."
    )

    report_text = "\n".join(report_lines)

    report_path = OUT_DIR / "all_data_diagnostic_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print()
    print(f"Saved full diagnostic report to: {report_path}")


if __name__ == "__main__":
    main()
