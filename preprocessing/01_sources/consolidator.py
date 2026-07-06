#!/usr/bin/env python3
"""Merge dataset CSV files into a single consolidated file."""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
MERGED_PATH = DATA_DIR / "merged_dataset.csv"
SAMPLE_PATH = DATA_DIR / "merged_dataset_sampled_49MiB.csv"
MAX_SIZE_MIB = 49
MIB = 1024 * 1024

DATASETS: dict[str, Path] = {
    "critique_brainz": DATA_DIR / "critique_brainz_reviews_cleaned.csv",
    "pitchfork": DATA_DIR / "pitchfork_reviews_cleaned.csv",
    "resident_advisor": DATA_DIR / "resident_advisor_reviews_cleaned.csv",
}

# Canonical column -> source column when datasets use different names.
COLUMN_ALIASES: dict[str, str] = {
    "published_on": "pub_date",
    "text": "body",
    "rating": "score",
    "cleaned_text": "cleaned_body",
    "album": "title",
    "reviewer_name": "author",
    "source_url": "review_url",
    "genre": "genres",
}

DROP_COLUMNS = [
    "review_url",
    "is_standard_review",
    "pub_date",
    "body",
    "title",
    "score",
    "artist_count",
    "author",
    "cleaned_body",
    "genres",
]

DATASET_RENAMES: dict[str, dict[str, str]] = {
    "resident_advisor": {
        "date": "pub_date",
        "review": "body",
        "cleaned_review": "cleaned_body",
    }
}


def review_id_from_url(review_url: object) -> str | None:
    if pd.isna(review_url):
        return None

    match = re.search(r"/(\d+)", str(review_url))
    return match.group(1) if match else None


def _fill_from_alias(df: pd.DataFrame, canonical: str, source: str) -> None:
    if source not in df.columns:
        return

    if canonical in df.columns:
        df[canonical] = df[canonical].fillna(df[source])
    else:
        df[canonical] = df[source]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for canonical, source in COLUMN_ALIASES.items():
        _fill_from_alias(df, canonical, source)

    # Pitchfork scores are on a 0-10 scale; normalize to 0-5.
    if "score" in df.columns and "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce") / 2

    if "review_url" in df.columns:
        extracted_review_id = df["review_url"].map(review_id_from_url)
        if "review_id" in df.columns:
            df["review_id"] = df["review_id"].fillna(extracted_review_id)
        else:
            df["review_id"] = extracted_review_id

    return df


def load_and_merge(datasets: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        df = pd.read_csv(path)
        if name in DATASET_RENAMES:
            df = df.rename(columns=DATASET_RENAMES[name])
        df = normalize_columns(df)
        df.insert(0, "dataset", name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged.drop(columns=[col for col in DROP_COLUMNS if col in merged.columns])


def file_size_mib(path: Path) -> float:
    return path.stat().st_size / MIB


def save_random_sample_if_large(
    source: Path,
    sample_path: Path,
    *,
    max_mib: float = MAX_SIZE_MIB,
    random_state: int = 42,
) -> Path | None:
    size_mib = file_size_mib(source)
    if size_mib <= max_mib:
        return None

    df = pd.read_csv(source, low_memory=False)
    if df.empty:
        return None

    fraction = min(max_mib / size_mib, 1.0)
    sample_size = max(1, int(len(df) * fraction))
    sample = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    sample.to_csv(sample_path, index=False)
    return sample_path


def consolidate(
    datasets: dict[str, Path] | None = None,
    *,
    merged_path: Path = MERGED_PATH,
    sample_path: Path = SAMPLE_PATH,
) -> pd.DataFrame:
    datasets = datasets or DATASETS
    merged = load_and_merge(datasets)
    merged.to_csv(merged_path, index=False)

    print(f"Saved {len(merged):,} rows to {merged_path} ({file_size_mib(merged_path):.1f} MiB)")

    saved_sample = save_random_sample_if_large(merged_path, sample_path)
    if saved_sample:
        print(
            f"Merged file exceeds {MAX_SIZE_MIB} MiB; "
            f"saved random sample to {saved_sample} ({file_size_mib(saved_sample):.1f} MiB)"
        )

    return merged


if __name__ == "__main__":
    consolidate()
