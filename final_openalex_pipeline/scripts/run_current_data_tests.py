from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_openalex"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR / "current_data_tests"

OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1975
END_YEAR = 2025
N_YEARS = END_YEAR - START_YEAR + 1
N_TERMS = 11
EXPECTED_TERM_YEAR_TASKS = N_YEARS * N_TERMS


def add_result(results, test_name, status, detail):
    results.append(
        {
            "test": test_name,
            "status": status,
            "detail": detail,
        }
    )


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    results = []

    # ------------------------------------------------------------
    # 1. Raw download progress
    # ------------------------------------------------------------

    done_files = list(RAW_DIR.rglob("*.done"))
    jsonl_files = list(RAW_DIR.rglob("*.jsonl"))
    tmp_files = list(RAW_DIR.rglob("*.tmp"))

    done_count = len(done_files)
    jsonl_count = len(jsonl_files)
    tmp_count = len(tmp_files)

    progress_pct = done_count / EXPECTED_TERM_YEAR_TASKS * 100

    add_result(
        results,
        "Raw download progress",
        "INFO",
        f"{done_count}/{EXPECTED_TERM_YEAR_TASKS} term-year tasks complete "
        f"({progress_pct:.2f}%).",
    )

    if tmp_count == 0:
        add_result(results, "Temporary raw files", "PASS", "No unfinished .tmp files found.")
    else:
        add_result(
            results,
            "Temporary raw files",
            "WARN",
            f"{tmp_count} unfinished .tmp files found. Delete them before resuming.",
        )

    if jsonl_count == done_count:
        add_result(
            results,
            "Raw JSONL vs DONE count",
            "PASS",
            f"{jsonl_count} JSONL files and {done_count} DONE markers.",
        )
    else:
        add_result(
            results,
            "Raw JSONL vs DONE count",
            "WARN",
            f"{jsonl_count} JSONL files but {done_count} DONE markers.",
        )

    # ------------------------------------------------------------
    # 2. Year-level raw coverage
    # ------------------------------------------------------------

    year_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = RAW_DIR / str(year)
        year_done = len(list(year_dir.glob("*.done"))) if year_dir.exists() else 0
        year_jsonl = len(list(year_dir.glob("*.jsonl"))) if year_dir.exists() else 0

        year_rows.append(
            {
                "year": year,
                "done_files": year_done,
                "jsonl_files": year_jsonl,
                "complete_all_11_terms": year_done == N_TERMS,
            }
        )

    year_progress = pd.DataFrame(year_rows)
    year_progress.to_csv(OUT_DIR / "raw_year_progress.csv", index=False)

    completed_years = year_progress["complete_all_11_terms"].sum()

    add_result(
        results,
        "Complete years",
        "INFO",
        f"{completed_years}/{N_YEARS} years have all {N_TERMS} search terms downloaded.",
    )

    incomplete_years = year_progress[~year_progress["complete_all_11_terms"]]
    incomplete_years.to_csv(OUT_DIR / "incomplete_years.csv", index=False)

    # ------------------------------------------------------------
    # 3. Load processed files
    # ------------------------------------------------------------

    clean = load_csv(PROCESSED_DIR / "openalex_clean_corpus.csv")
    candidates = load_csv(PROCESSED_DIR / "openalex_candidates_all.csv")
    yearly = load_csv(PROCESSED_DIR / "openalex_yearly_summary.csv")
    term_summary = load_csv(PROCESSED_DIR / "openalex_term_summary.csv")
    type_summary = load_csv(PROCESSED_DIR / "openalex_type_summary.csv")

    if clean is None:
        add_result(
            results,
            "Clean corpus file",
            "FAIL",
            "Missing openalex_clean_corpus.csv. Run: python scripts/run_full_openalex_search.py --process --validate",
        )
        final = pd.DataFrame(results)
        final.to_csv(OUT_DIR / "current_data_test_results.csv", index=False)
        print(final.to_string(index=False))
        return

    add_result(
        results,
        "Clean corpus file",
        "PASS",
        f"Loaded openalex_clean_corpus.csv with {len(clean):,} records.",
    )

    # ------------------------------------------------------------
    # 4. Basic corpus checks
    # ------------------------------------------------------------

    required_cols = [
        "openalex_id",
        "title",
        "abstract",
        "publication_year",
        "has_abstract",
        "matches_title_or_abstract",
        "detected_terms_in_title_abstract",
    ]

    missing_cols = [col for col in required_cols if col not in clean.columns]

    if missing_cols:
        add_result(
            results,
            "Required columns",
            "FAIL",
            f"Missing columns: {missing_cols}",
        )
    else:
        add_result(results, "Required columns", "PASS", "All required columns are present.")

    if "publication_year" in clean.columns:
        clean["publication_year"] = pd.to_numeric(clean["publication_year"], errors="coerce")
        min_year = clean["publication_year"].min()
        max_year = clean["publication_year"].max()

        if min_year >= START_YEAR and max_year <= END_YEAR:
            add_result(
                results,
                "Year range",
                "PASS",
                f"Current processed data range: {int(min_year)}–{int(max_year)}.",
            )
        else:
            add_result(
                results,
                "Year range",
                "FAIL",
                f"Found year range outside project window: {min_year}–{max_year}.",
            )

    if "title" in clean.columns:
        missing_titles = clean["title"].fillna("").str.strip().eq("").sum()

        if missing_titles == 0:
            add_result(results, "Missing titles", "PASS", "No missing titles.")
        else:
            add_result(results, "Missing titles", "FAIL", f"{missing_titles:,} rows have missing titles.")

    if "matches_title_or_abstract" in clean.columns:
        match_values = as_bool(clean["matches_title_or_abstract"])
        nonmatching = (~match_values).sum()

        if nonmatching == 0:
            add_result(
                results,
                "Title/abstract boundary",
                "PASS",
                "All clean records match at least one Search B term in title or abstract.",
            )
        else:
            add_result(
                results,
                "Title/abstract boundary",
                "FAIL",
                f"{nonmatching:,} clean records do not match the title/abstract boundary.",
            )

    # ------------------------------------------------------------
    # 5. Duplicate checks
    # ------------------------------------------------------------

    if "openalex_id" in clean.columns:
        duplicate_ids = clean["openalex_id"].duplicated().sum()

        if duplicate_ids == 0:
            add_result(results, "Duplicate OpenAlex IDs", "PASS", "No duplicate OpenAlex IDs.")
        else:
            add_result(results, "Duplicate OpenAlex IDs", "FAIL", f"{duplicate_ids:,} duplicate IDs found.")

    if "doi" in clean.columns:
        doi_norm = clean["doi"].fillna("").astype(str).str.lower().str.strip()
        duplicate_dois = doi_norm[doi_norm.ne("")].duplicated().sum()

        if duplicate_dois == 0:
            add_result(results, "Duplicate DOIs", "PASS", "No duplicate non-empty DOIs.")
        else:
            add_result(results, "Duplicate DOIs", "FAIL", f"{duplicate_dois:,} duplicate DOIs found.")

    if "title_norm" in clean.columns and "publication_year" in clean.columns:
        duplicate_title_year = clean.duplicated(subset=["title_norm", "publication_year"]).sum()

        if duplicate_title_year == 0:
            add_result(
                results,
                "Duplicate title/year",
                "PASS",
                "No duplicate normalized title + publication year pairs.",
            )
        else:
            add_result(
                results,
                "Duplicate title/year",
                "WARN",
                f"{duplicate_title_year:,} duplicate normalized title/year pairs found.",
            )

    # ------------------------------------------------------------
    # 6. Abstract coverage checks
    # ------------------------------------------------------------

    if "has_abstract" in clean.columns:
        abstract_bool = as_bool(clean["has_abstract"])
        abstract_count = abstract_bool.sum()
        abstract_pct = abstract_count / len(clean) * 100 if len(clean) else 0

        add_result(
            results,
            "Abstract coverage",
            "INFO",
            f"{abstract_count:,}/{len(clean):,} records have abstracts ({abstract_pct:.2f}%).",
        )

    if "abstract_word_count" in clean.columns:
        clean["abstract_word_count"] = pd.to_numeric(clean["abstract_word_count"], errors="coerce").fillna(0)
        short_abstracts = clean[(clean["abstract_word_count"] > 0) & (clean["abstract_word_count"] < 20)]

        if len(short_abstracts) == 0:
            add_result(results, "Short abstracts", "PASS", "No very short abstracts under 20 words.")
        else:
            add_result(
                results,
                "Short abstracts",
                "WARN",
                f"{len(short_abstracts):,} abstracts are under 20 words.",
            )
            short_abstracts.to_csv(OUT_DIR / "short_abstracts_under_20_words.csv", index=False)

    # ------------------------------------------------------------
    # 7. Summary consistency checks
    # ------------------------------------------------------------

    if yearly is not None:
        yearly_total = yearly["total_records"].sum()

        if yearly_total == len(clean):
            add_result(
                results,
                "Yearly summary total",
                "PASS",
                f"Yearly summary total equals clean corpus total: {yearly_total:,}.",
            )
        else:
            add_result(
                results,
                "Yearly summary total",
                "FAIL",
                f"Yearly total {yearly_total:,} does not equal clean total {len(clean):,}.",
            )

    if term_summary is not None:
        add_result(
            results,
            "Term summary file",
            "PASS",
            f"Loaded term summary with {len(term_summary):,} rows.",
        )
    else:
        add_result(results, "Term summary file", "WARN", "Missing openalex_term_summary.csv.")

    if type_summary is not None:
        add_result(
            results,
            "Type summary file",
            "PASS",
            f"Loaded type summary with {len(type_summary):,} rows.",
        )
    else:
        add_result(results, "Type summary file", "WARN", "Missing openalex_type_summary.csv.")

    # ------------------------------------------------------------
    # 8. Candidate-level checks
    # ------------------------------------------------------------

    if candidates is not None:
        add_result(
            results,
            "Candidate file",
            "PASS",
            f"Loaded openalex_candidates_all.csv with {len(candidates):,} candidate rows.",
        )

        if "matches_title_or_abstract" in candidates.columns:
            candidate_match = as_bool(candidates["matches_title_or_abstract"])
            rejected = candidates[~candidate_match]

            rejected.to_csv(OUT_DIR / "candidate_records_rejected_by_boundary.csv", index=False)

            add_result(
                results,
                "Candidate rejection count",
                "INFO",
                f"{len(rejected):,}/{len(candidates):,} candidate rows do not match the final title/abstract boundary.",
            )
    else:
        add_result(results, "Candidate file", "WARN", "Missing openalex_candidates_all.csv.")

    # ------------------------------------------------------------
    # 9. Save useful manual-review samples
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

    sample_n = min(25, len(clean))

    if sample_n > 0:
        sample = clean.sample(n=sample_n, random_state=42)[sample_cols]
        sample.to_csv(OUT_DIR / "manual_relevance_sample_25.csv", index=False)

        add_result(
            results,
            "Manual relevance sample",
            "INFO",
            f"Saved {sample_n} random records to manual_relevance_sample_25.csv.",
        )

    if "publication_year" in clean.columns and len(clean) > 0:
        year_counts = (
            clean["publication_year"]
            .value_counts()
            .sort_index()
            .rename_axis("publication_year")
            .reset_index(name="records")
        )
        year_counts.to_csv(OUT_DIR / "current_year_counts.csv", index=False)

    if "detected_terms_in_title_abstract" in clean.columns:
        term_rows = []

        for _, row in clean.iterrows():
            terms = str(row.get("detected_terms_in_title_abstract", "")).split(";")
            for term in terms:
                term = term.strip()
                if term:
                    term_rows.append(
                        {
                            "openalex_id": row.get("openalex_id"),
                            "publication_year": row.get("publication_year"),
                            "term": term,
                        }
                    )

        if term_rows:
            exploded_terms = pd.DataFrame(term_rows)
            term_counts = (
                exploded_terms["term"]
                .value_counts()
                .rename_axis("term")
                .reset_index(name="records")
            )

            term_counts.to_csv(OUT_DIR / "current_detected_term_counts.csv", index=False)

    # ------------------------------------------------------------
    # 10. Save final report
    # ------------------------------------------------------------

    final = pd.DataFrame(results)
    final.to_csv(OUT_DIR / "current_data_test_results.csv", index=False)

    report_lines = []
    report_lines.append("Current Data Diagnostic Report")
    report_lines.append("=" * 40)
    report_lines.append("")

    for _, row in final.iterrows():
        report_lines.append(f"[{row['status']}] {row['test']}: {row['detail']}")

    report_text = "\n".join(report_lines)
    report_path = OUT_DIR / "current_data_diagnostic_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print()
    print(f"Saved report to: {report_path}")
    print(f"Saved test outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
