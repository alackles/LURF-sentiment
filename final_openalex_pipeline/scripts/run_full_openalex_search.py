from __future__ import annotations

"""
OpenAlex full search pipeline for the Conversational AI project.

Purpose
-------
This script builds the main body of literature for a project studying
how academic language around conversational AI changed from 1975 to 2025.

The pipeline is designed to be:

1. Reproducible
   The time window, search terms, and cleaning rules are written directly
   in the code.

2. Restartable
   Raw OpenAlex results are saved one term and one year at a time. If the
   script stops halfway, finished files are skipped the next time.

3. Transparent
   The script keeps track of which search term retrieved each record.
   This helps us explain how the corpus was built.

4. Safe
   The OpenAlex API key is stored in a hidden .env file and should never
   be pushed to GitHub.

Important project choice
------------------------
OpenAlex does not provide abstracts for every record, especially older
records. Because of this, the script keeps title-only records in the main
corpus and creates a separate abstract subset for analysis requiring
abstracts.
"""

import argparse
import html
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# 1. Project paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_openalex"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = ROOT / "logs"

for folder in [RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_PATH)

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")

if not OPENALEX_API_KEY:
    raise RuntimeError(
        "OPENALEX_API_KEY was not found.\n"
        "Check that your .env file is here:\n"
        f"{ENV_PATH}\n\n"
        "The .env file should contain:\n"
        "OPENALEX_API_KEY=your_key_here"
    )

BASE_URL = "https://api.openalex.org/works"

START_YEAR = 1975
END_YEAR = 2025

PER_PAGE = 100
REQUEST_SLEEP_SECONDS = 3.0


# ============================================================
# 2. Search terms
# ============================================================

# These terms define the boundary of the project.
# They are meant to capture papers specifically about conversational systems,
# not every paper about artificial intelligence.

SEARCH_TERMS = [
    {"label": "chatbot", "query": "chatbot"},
    {"label": "chatterbot", "query": "chatterbot"},
    {"label": "conversational agent", "query": "conversational agent"},
    {"label": "embodied conversational agent", "query": "embodied conversational agent"},
    {"label": "dialogue system", "query": "dialogue system"},
    {"label": "dialog system", "query": "dialog system"},
    {"label": "spoken dialogue system", "query": "spoken dialogue system"},
    {"label": "virtual assistant", "query": "virtual assistant"},
    {"label": "voice assistant", "query": "voice assistant"},
    {"label": "conversational AI", "query": "conversational AI"},
    {"label": "natural language interface", "query": "natural language interface"},
]

# These regex patterns are used after downloading.
# They make the final corpus stricter by keeping only records where the term
# appears in the title or reconstructed abstract.

TERM_PATTERNS = {
    "chatbot": r"\bchatbots?\b",
    "chatterbot": r"\bchatterbots?\b",
    "conversational agent": r"\bconversational agents?\b",
    "embodied conversational agent": r"\bembodied conversational agents?\b",
    "dialogue system": r"\bdialogue systems?\b",
    "dialog system": r"\bdialog systems?\b",
    "spoken dialogue system": r"\bspoken dialogue systems?\b",
    "virtual assistant": r"\bvirtual assistants?\b",
    "voice assistant": r"\bvoice assistants?\b",
    "conversational AI": r"\bconversational\s+AI\b",
    "natural language interface": r"\bnatural language interfaces?\b",
}


# ============================================================
# 3. Helper functions
# ============================================================

def safe_filename(text: str) -> str:
    """Create a safe file name from a search label."""

    text = text.lower()
    text = re.sub(r'["\']', "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_title(value: Any) -> str:
    """
    Normalize title for duplicate checking.

    This is not used for text analysis. It only helps catch duplicates
    when DOI is missing.
    """

    if value is None:
        return ""

    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(value: Any) -> str:
    """Clean basic HTML and repeated whitespace from text fields."""

    if value is None:
        return ""

    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def reconstruct_abstract(inverted_index: Any) -> str:
    """
    Reconstruct OpenAlex abstracts.

    OpenAlex stores abstracts as an inverted index:
        word -> list of word positions

    This function converts that structure into normal text.
    """

    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    positions = {}

    for word, idxs in inverted_index.items():
        if not isinstance(idxs, list):
            continue

        for idx in idxs:
            positions[idx] = word

    if not positions:
        return ""

    return " ".join(positions[i] for i in sorted(positions))


def title_abstract_text(title: Any, abstract: Any) -> str:
    """Combine title and abstract for term matching."""

    title = "" if title is None else str(title)
    abstract = "" if abstract is None else str(abstract)
    return f"{title} {abstract}".strip()


def find_terms_in_title_abstract(title: Any, abstract: Any) -> list[str]:
    """
    Return Search B terms that appear in the title or abstract.

    This step is important because it makes the final corpus match the
    research boundary even if the API search itself returns broader records.
    """

    text = title_abstract_text(title, abstract)
    found_terms = []

    for label, pattern in TERM_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            found_terms.append(label)

    return found_terms


def get_rate_limit_wait_seconds() -> int:
    """
    Check OpenAlex rate-limit status and return how long to wait.

    If the daily budget is used up, this returns the reset time plus
    a small buffer.
    """

    try:
        response = requests.get(
            "https://api.openalex.org/rate-limit",
            params={"api_key": OPENALEX_API_KEY},
            timeout=60,
        )

        if response.status_code != 200:
            return 1800

        data = response.json()
        rate_limit = data.get("rate_limit", {})

        remaining = float(rate_limit.get("daily_remaining_usd", 0))
        resets_in_seconds = int(rate_limit.get("resets_in_seconds", 1800))

        if remaining <= 0:
            return resets_in_seconds + 120

        return 600

    except Exception:
        return 1800


def request_openalex(params: dict[str, Any], max_attempts: int = 50) -> dict[str, Any]:
    """
    Request one OpenAlex API page with retries.

    If the daily API budget is used up, this function waits until the
    reset time instead of crashing.
    """

    params = dict(params)
    params["api_key"] = OPENALEX_API_KEY

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=90)

            if response.status_code == 200:
                time.sleep(REQUEST_SLEEP_SECONDS)
                return response.json()

            if response.status_code in {401, 403}:
                raise RuntimeError(
                    f"OpenAlex returned {response.status_code}.\n"
                    "Check your API key.\n"
                    f"Response: {response.text[:1000]}"
                )

            if response.status_code == 429:
                wait = get_rate_limit_wait_seconds()
                print(f"Rate limit or daily budget hit. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:
                wait = min(1800, max(120, 2 ** attempt))
                print(f"Server error {response.status_code}. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"OpenAlex returned HTTP {response.status_code}.\n"
                f"URL: {response.url}\n"
                f"Response: {response.text[:1000]}"
            )

        except requests.RequestException as error:
            if attempt == max_attempts:
                raise RuntimeError("Request failed after repeated attempts.") from error

            wait = min(1800, max(120, 2 ** attempt))
            print(f"Connection error. Waiting {wait} seconds...")
            time.sleep(wait)

    raise RuntimeError("OpenAlex request failed after too many retries.")

# ============================================================
# 4. Download raw OpenAlex records
# ============================================================

def download_term_year(
    year: int,
    label: str,
    query: str,
    force: bool = False,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """
    Download all OpenAlex records for one term in one publication year.
    Why one term and one year at a time?
    - It is easier to restart.
    - It creates an audit trail.
    - It lets us see which term retrieved each record.
    - It avoids relying on one giant Boolean query.
    """

    year_dir = RAW_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    file_stem = safe_filename(label)
    raw_path = year_dir / f"{file_stem}.jsonl"
    done_path = year_dir / f"{file_stem}.done"
    temp_path = year_dir / f"{file_stem}.jsonl.tmp"

    if raw_path.exists() and done_path.exists() and not force:
        downloaded = sum(1 for _ in raw_path.open("r", encoding="utf-8"))
        return {
            "year": year,
            "label": label,
            "query": query,
            "status": "skipped_existing",
            "total_available": None,
            "downloaded": downloaded,
            "pages": None,
            "raw_file": str(raw_path),
        }

    if temp_path.exists():
        temp_path.unlink()

    if force and raw_path.exists():
        raw_path.unlink()

    if force and done_path.exists():
        done_path.unlink()

    cursor = "*"
    total_available = None
    downloaded = 0
    pages = 0

    select_fields = ",".join(
        [
            "id",
            "doi",
            "title",
            "display_name",
            "publication_year",
            "publication_date",
            "type",
            "language",
            "cited_by_count",
            "is_retracted",
            "is_paratext",
            "abstract_inverted_index",
            "primary_location",
            "authorships",
            "open_access",
            "topics",
            "keywords",
        ]
    )

    with temp_path.open("w", encoding="utf-8") as out:
        while True:
            params = {
                "filter": f"title_and_abstract.search:{query},publication_year:{year}",
                "per_page": PER_PAGE,
                "cursor": cursor,
                "select": select_fields,
            }

            payload = request_openalex(params)
            meta = payload.get("meta", {})
            results = payload.get("results", [])

            if total_available is None:
                total_available = meta.get("count", 0)

            pages += 1

            for work in results:
                work["_retrieved_by_term"] = label
                work["_openalex_query"] = query
                work["_search_year"] = year

                out.write(json.dumps(work, ensure_ascii=False) + "\n")
                downloaded += 1

            cursor = meta.get("next_cursor")

            if not results or not cursor:
                break

            if max_pages is not None and pages >= max_pages:
                print(f"Stopped early because max_pages={max_pages}")
                break

            time.sleep(REQUEST_SLEEP_SECONDS)

    temp_path.rename(raw_path)
    done_path.write_text("done\n", encoding="utf-8")

    return {
        "year": year,
        "label": label,
        "query": query,
        "status": "downloaded",
        "total_available": total_available,
        "downloaded": downloaded,
        "pages": pages,
        "raw_file": str(raw_path),
    }


def run_download(
    start_year: int,
    end_year: int,
    terms: list[dict[str, str]],
    force: bool = False,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """Run all term-year downloads and save a manifest."""

    manifest_rows = []
    tasks = []

    for year in range(start_year, end_year + 1):
        for term in terms:
            tasks.append((year, term["label"], term["query"]))

    for year, label, query in tqdm(tasks, desc="Downloading term-year searches"):
        print(f"\nSearching year={year}, term={label}")

        row = download_term_year(
            year=year,
            label=label,
            query=query,
            force=force,
            max_pages=max_pages,
        )

        manifest_rows.append(row)

        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(PROCESSED_DIR / "download_manifest.csv", index=False)

    return pd.DataFrame(manifest_rows)


# ============================================================
# 5. Parse raw JSONL into a table
# ============================================================

def iter_raw_records():
    """Yield every raw OpenAlex work from saved JSONL files."""

    for path in sorted(RAW_DIR.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping bad JSON line in {path}")


def parse_authors(work: dict[str, Any], max_authors: int = 20) -> str:
    """Extract author names from OpenAlex authorships."""

    names = []

    for item in work.get("authorships") or []:
        author = item.get("author") or {}
        name = author.get("display_name")

        if name:
            names.append(name)

        if len(names) >= max_authors:
            break

    return "; ".join(names)


def parse_topics(work: dict[str, Any], max_topics: int = 5) -> str:
    """Extract top OpenAlex topics."""

    topics = []

    for item in work.get("topics") or []:
        name = item.get("display_name")

        if name:
            topics.append(name)

        if len(topics) >= max_topics:
            break

    return "; ".join(topics)


def parse_keywords(work: dict[str, Any], max_keywords: int = 10) -> str:
    """Extract OpenAlex keywords."""

    keywords = []

    for item in work.get("keywords") or []:
        name = item.get("display_name")

        if name:
            keywords.append(name)

        if len(keywords) >= max_keywords:
            break

    return "; ".join(keywords)


def parse_work(work: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw OpenAlex work into one row for our dataset."""

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}

    title = clean_text(work.get("title") or work.get("display_name") or "")
    abstract = clean_text(reconstruct_abstract(work.get("abstract_inverted_index")))

    detected_terms = find_terms_in_title_abstract(title, abstract)

    return {
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": title,
        "abstract": abstract,
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "language": work.get("language"),
        "cited_by_count": work.get("cited_by_count"),
        "is_retracted": work.get("is_retracted"),
        "is_paratext": work.get("is_paratext"),
        "is_oa": open_access.get("is_oa"),
        "oa_status": open_access.get("oa_status"),
        "venue": source.get("display_name"),
        "source_type": source.get("type"),
        "landing_page_url": primary_location.get("landing_page_url"),
        "pdf_url": primary_location.get("pdf_url"),
        "authors": parse_authors(work),
        "topics": parse_topics(work),
        "keywords": parse_keywords(work),
        "retrieved_by_term": work.get("_retrieved_by_term"),
        "openalex_query": work.get("_openalex_query"),
        "search_year": work.get("_search_year"),
        "detected_terms_in_title_abstract": "; ".join(detected_terms),
        "matches_title_or_abstract": len(detected_terms) > 0,
        "has_title": bool(title.strip()),
        "has_abstract": bool(abstract.strip()),
        "abstract_word_count": len(abstract.split()) if abstract else 0,
        "title_norm": normalize_title(title),
        "doi_norm": str(work.get("doi") or "").lower().strip(),
    }


def combine_raw_records() -> pd.DataFrame:
    """Combine all saved raw records into one candidate table."""

    rows = [parse_work(work) for work in iter_raw_records()]
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No raw records found. Run the download step first.")

    df.to_csv(PROCESSED_DIR / "openalex_candidates_all.csv", index=False)
    df.to_parquet(PROCESSED_DIR / "openalex_candidates_all.parquet", index=False)

    print(f"Candidate rows before deduplication: {len(df):,}")
    return df


# ============================================================
# 6. Clean and deduplicate
# ============================================================

def combine_duplicate_openalex_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine records with the same OpenAlex ID.

    A work can be retrieved by multiple search terms. We keep one row
    per work, but preserve all terms that retrieved it.
    """

    df = df.copy()
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0)

    rows = []

    for openalex_id, group in df.groupby("openalex_id", dropna=False):
        group = group.sort_values("cited_by_count", ascending=False)
        row = group.iloc[0].to_dict()

        retrieved_terms = sorted(
            set(
                str(x)
                for x in group["retrieved_by_term"].dropna().tolist()
                if str(x).strip()
            )
        )

        detected_terms = sorted(
            set(
                term.strip()
                for value in group["detected_terms_in_title_abstract"].dropna().tolist()
                for term in str(value).split(";")
                if term.strip()
            )
        )

        search_years = sorted(
            set(
                int(y)
                for y in group["search_year"].dropna().tolist()
                if str(y).strip()
            )
        )

        row["retrieved_by_terms"] = "; ".join(retrieved_terms)
        row["detected_terms_in_title_abstract"] = "; ".join(detected_terms)
        row["search_years"] = "; ".join(str(y) for y in search_years)

        rows.append(row)

    return pd.DataFrame(rows)


def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw OpenAlex candidates into the final clean corpus.

    Main cleaning logic:
    - keep only 1975–2025;
    - remove retracted and paratext records;
    - keep records with a title;
    - enforce that at least one Search B term appears in title or abstract;
    - deduplicate by DOI when available;
    - otherwise deduplicate by normalized title and publication year.
    """

    df = combine_duplicate_openalex_ids(df)

    before = len(df)

    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df[df["publication_year"].between(START_YEAR, END_YEAR, inclusive="both")]
    df["publication_year"] = df["publication_year"].astype(int)

    df = df[df["is_retracted"] != True]
    df = df[df["is_paratext"] != True]
    df = df[df["has_title"] == True]
    df = df[df["matches_title_or_abstract"] == True]

    print(f"Rows before main cleaning: {before:,}")
    print(f"Rows after main cleaning: {len(df):,}")

    df["doi_norm"] = df["doi_norm"].fillna("").astype(str)

    with_doi = df[df["doi_norm"].ne("")].copy()
    without_doi = df[df["doi_norm"].eq("")].copy()

    with_doi = (
        with_doi.sort_values("cited_by_count", ascending=False)
        .drop_duplicates(subset=["doi_norm"], keep="first")
    )

    without_doi = (
        without_doi.sort_values("cited_by_count", ascending=False)
        .drop_duplicates(subset=["title_norm", "publication_year"], keep="first")
    )

    clean = pd.concat([with_doi, without_doi], ignore_index=True)

    clean = clean.sort_values(
        ["publication_year", "title"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return clean


# ============================================================
# 7. Summary tables
# ============================================================

def create_yearly_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Create year-by-year corpus counts and abstract coverage."""

    summary = (
        clean.groupby("publication_year")
        .agg(
            total_records=("openalex_id", "count"),
            records_with_abstract=("has_abstract", "sum"),
            average_citations=("cited_by_count", "mean"),
        )
        .reset_index()
    )

    summary["abstract_coverage_percent"] = (
        summary["records_with_abstract"] / summary["total_records"] * 100
    ).round(2)

    return summary


def create_term_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often each Search B term appears in the final corpus."""

    rows = []

    for label, pattern in TERM_PATTERNS.items():
        matched = clean[
            clean.apply(
                lambda row: bool(
                    re.search(
                        pattern,
                        title_abstract_text(row["title"], row["abstract"]),
                        flags=re.IGNORECASE,
                    )
                ),
                axis=1,
            )
        ]

        rows.append(
            {
                "term": label,
                "records_containing_term": len(matched),
                "first_year": matched["publication_year"].min() if len(matched) else None,
                "last_year": matched["publication_year"].max() if len(matched) else None,
                "mean_publication_year": round(matched["publication_year"].mean(), 2)
                if len(matched)
                else None,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "records_containing_term",
        ascending=False,
    )


def create_type_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Summarize OpenAlex work types in the final corpus."""

    return (
        clean["type"]
        .fillna("missing")
        .value_counts()
        .rename_axis("type")
        .reset_index(name="records")
    )


def save_outputs(clean: pd.DataFrame) -> None:
    """Save final corpus and summary files."""

    if clean.empty:
        raise RuntimeError(
            "The clean corpus is empty. The API search may have returned no usable records."
        )

    abstract_subset = clean[clean["has_abstract"] == True].copy()

    yearly_summary = create_yearly_summary(clean)
    term_summary = create_term_summary(clean)
    type_summary = create_type_summary(clean)

    clean.to_csv(PROCESSED_DIR / "openalex_clean_corpus.csv", index=False)
    clean.to_parquet(PROCESSED_DIR / "openalex_clean_corpus.parquet", index=False)

    abstract_subset.to_csv(PROCESSED_DIR / "openalex_abstract_subset.csv", index=False)

    yearly_summary.to_csv(PROCESSED_DIR / "openalex_yearly_summary.csv", index=False)
    term_summary.to_csv(PROCESSED_DIR / "openalex_term_summary.csv", index=False)
    type_summary.to_csv(PROCESSED_DIR / "openalex_type_summary.csv", index=False)

    print("\nSaved outputs:")
    print(PROCESSED_DIR / "openalex_clean_corpus.csv")
    print(PROCESSED_DIR / "openalex_abstract_subset.csv")
    print(PROCESSED_DIR / "openalex_yearly_summary.csv")
    print(PROCESSED_DIR / "openalex_term_summary.csv")
    print(PROCESSED_DIR / "openalex_type_summary.csv")

    print("\nFinal corpus summary:")
    print(f"Total clean records: {len(clean):,}")
    print(f"Records with abstracts: {clean['has_abstract'].sum():,}")
    print(f"Abstract coverage: {clean['has_abstract'].mean() * 100:.2f}%")
    print(
        "Year range: "
        f"{int(clean['publication_year'].min())}–{int(clean['publication_year'].max())}"
    )


# ============================================================
# 8. Validation checks
# ============================================================

def validate_outputs() -> None:
    """
    Check whether the processed outputs make logical sense.

    These checks are not the final analysis. They are safety checks to
    catch obvious problems before we use the dataset.
    """

    clean_path = PROCESSED_DIR / "openalex_clean_corpus.csv"
    yearly_path = PROCESSED_DIR / "openalex_yearly_summary.csv"
    term_path = PROCESSED_DIR / "openalex_term_summary.csv"

    problems = []
    notes = []

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}")

    clean = pd.read_csv(clean_path)

    notes.append(f"Clean corpus records: {len(clean):,}")

    required_columns = [
        "openalex_id",
        "title",
        "abstract",
        "publication_year",
        "has_abstract",
        "matches_title_or_abstract",
        "detected_terms_in_title_abstract",
    ]

    for col in required_columns:
        if col not in clean.columns:
            problems.append(f"Missing required column: {col}")

    if len(clean) == 0:
        problems.append("Clean corpus is empty.")

    if "publication_year" in clean.columns and len(clean) > 0:
        min_year = clean["publication_year"].min()
        max_year = clean["publication_year"].max()

        notes.append(f"Year range found: {min_year}–{max_year}")

        if min_year < START_YEAR or max_year > END_YEAR:
            problems.append(
                f"Year range outside project window: {min_year}–{max_year}"
            )

    if "title" in clean.columns:
        missing_titles = clean["title"].fillna("").str.strip().eq("").sum()
        notes.append(f"Rows with missing title: {missing_titles:,}")

        if missing_titles > 0:
            problems.append("Some rows have missing titles.")

    if "matches_title_or_abstract" in clean.columns:
        match_values = clean["matches_title_or_abstract"].astype(str).str.lower()
        non_matching = (~match_values.isin(["true", "1"])).sum()

        notes.append(f"Rows not matching title/abstract boundary: {non_matching:,}")

        if non_matching > 0:
            problems.append("Some rows do not match any Search B term in title/abstract.")

    if "openalex_id" in clean.columns:
        duplicate_ids = clean["openalex_id"].duplicated().sum()
        notes.append(f"Duplicate OpenAlex IDs: {duplicate_ids:,}")

        if duplicate_ids > 0:
            problems.append("Duplicate OpenAlex IDs remain.")

    if "doi" in clean.columns:
        doi_norm = clean["doi"].fillna("").astype(str).str.lower().str.strip()
        duplicate_dois = doi_norm[doi_norm.ne("")].duplicated().sum()

        notes.append(f"Duplicate non-empty DOIs: {duplicate_dois:,}")

        if duplicate_dois > 0:
            problems.append("Duplicate DOIs remain.")

    if "has_abstract" in clean.columns:
        abstract_values = clean["has_abstract"].astype(str).str.lower()
        abstract_count = abstract_values.isin(["true", "1"]).sum()
        abstract_pct = abstract_count / len(clean) * 100 if len(clean) else 0

        notes.append(f"Records with abstracts: {abstract_count:,} ({abstract_pct:.2f}%)")

    if yearly_path.exists():
        yearly = pd.read_csv(yearly_path)
        yearly_total = yearly["total_records"].sum()

        notes.append(f"Yearly summary total: {yearly_total:,}")

        if yearly_total != len(clean):
            problems.append(
                f"Yearly summary total ({yearly_total}) does not equal clean corpus total ({len(clean)})."
            )
    else:
        problems.append(f"Missing {yearly_path}")

    if term_path.exists():
        term_summary = pd.read_csv(term_path)
        notes.append(f"Term summary rows: {len(term_summary):,}")
    else:
        problems.append(f"Missing {term_path}")

    report_lines = []
    report_lines.append("OpenAlex Pipeline Validation Report")
    report_lines.append("=" * 40)
    report_lines.append("")
    report_lines.append("Notes:")
    report_lines.extend(f"- {note}" for note in notes)
    report_lines.append("")
    report_lines.append("Problems:")

    if problems:
        report_lines.extend(f"- {problem}" for problem in problems)
    else:
        report_lines.append("- No obvious problems found.")

    report_text = "\n".join(report_lines)
    report_path = PROCESSED_DIR / "pipeline_validation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nValidation report saved to {report_path}")

    if problems:
        raise RuntimeError("Validation found problems. Read the validation report.")


# ============================================================
# 9. Command line interface
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, clean, and validate OpenAlex conversational AI literature."
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw OpenAlex JSONL files.",
    )

    parser.add_argument(
        "--process",
        action="store_true",
        help="Combine, clean, deduplicate, and export processed files.",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run logic checks on processed outputs.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Small test run: 2024–2025, first 2 terms, 1 page per term-year.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if raw files already exist.",
    )

    args = parser.parse_args()

    if not args.download and not args.process and not args.validate:
        raise RuntimeError(
            "Choose at least one action: --download, --process, or --validate."
        )

    if args.test:
        print("Running test mode.")
        terms = SEARCH_TERMS[:2]
        start_year = 2024
        end_year = 2025
        max_pages = 1
    else:
        terms = SEARCH_TERMS
        start_year = START_YEAR
        end_year = END_YEAR
        max_pages = None

    if args.download:
        manifest = run_download(
            start_year=start_year,
            end_year=end_year,
            terms=terms,
            force=args.force,
            max_pages=max_pages,
        )

        print("\nDownload manifest preview:")
        print(manifest.head())

    if args.process:
        candidates = combine_raw_records()
        clean = clean_corpus(candidates)
        save_outputs(clean)

    if args.validate:
        validate_outputs()


if __name__ == "__main__":
    main()
