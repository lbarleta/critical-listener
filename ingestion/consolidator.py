#!/usr/bin/env python3
"""Merge dataset CSV files into a single consolidated file."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
MERGED_PATH = DATA_DIR / "merged_dataset.csv"
SAMPLE_PATH = DATA_DIR / "merged_dataset_sampled_99MiB.csv"
MAX_SIZE_MIB = 100
MIB = 1024 * 1024

DATASETS: dict[str, Path] = {
    "critique_brainz": DATA_DIR / "critique_brainz_reviews_cleaned.csv",
    "pitchfork": DATA_DIR / "pitchfork_reviews_cleaned.csv",
}


def load_and_merge(datasets: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        df = pd.read_csv(path)
        df.insert(0, "dataset", name)
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False)


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
