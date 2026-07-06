"""Disk cache for lastfm_bench.ipynb to skip repeated metric recomputation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_loader import get_baseline, get_catalog, get_recs
from metrics import BenchInputs, compute_benchmark_metrics
from utils import build_subsets, filter_to_baseline, query_key

CACHE_VERSION = 2
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = MODULE_DIR / ".bench_cache"
CODE_PATHS = (
    MODULE_DIR / "bench_cache.py",
    MODULE_DIR / "data_loader.py",
    MODULE_DIR / "metrics.py",
    MODULE_DIR / "utils.py",
)


def _file_stat(path: Path) -> dict:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def build_fingerprint(
    *,
    catalog_path: Path,
    recs_path: Path,
    bench_path: Path,
    top_n: int,
) -> dict:
    sources = {
        "catalog": _file_stat(catalog_path),
        "recs": _file_stat(recs_path),
        "baseline": _file_stat(bench_path),
    }
    for code_path in CODE_PATHS:
        sources[f"code:{code_path.name}"] = _file_stat(code_path)
    return {"version": CACHE_VERSION, "top_n": top_n, "sources": sources}


def _cache_files(cache_dir: Path) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "manifest": cache_dir / "manifest.json",
        "metrics": cache_dir / "metrics_df.parquet",
        "recs_topn": cache_dir / "recs_topn.parquet",
        "baseline_topn": cache_dir / "baseline_topn.parquet",
    }


def cache_is_valid(cache_dir: Path, fingerprint: dict) -> bool:
    files = _cache_files(cache_dir)
    if not all(path.exists() for path in files.values()):
        return False
    saved = json.loads(files["manifest"].read_text())
    return saved == fingerprint


def load_cache(cache_dir: Path) -> dict:
    files = _cache_files(cache_dir)
    manifest = json.loads(files["manifest"].read_text())
    metrics_df = pd.read_parquet(files["metrics"])
    recs_n = pd.read_parquet(files["recs_topn"])
    baseline_n = pd.read_parquet(files["baseline_topn"])
    return {
        "top_n": manifest["top_n"],
        "metrics_df": metrics_df,
        "recs_n": recs_n,
        "baseline_n": baseline_n,
        "from_cache": True,
    }


def save_cache(
    cache_dir: Path,
    fingerprint: dict,
    *,
    metrics_df: pd.DataFrame,
    recs_n: pd.DataFrame,
    baseline_n: pd.DataFrame,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _cache_files(cache_dir)
    metrics_df.to_parquet(files["metrics"], index=False)
    recs_n.to_parquet(files["recs_topn"], index=False)
    baseline_n.to_parquet(files["baseline_topn"], index=False)
    files["manifest"].write_text(json.dumps(fingerprint, indent=2) + "\n")


def _compute_bench(
    catalog_path: Path,
    recs_path: Path,
    bench_path: Path,
    *,
    top_n: int,
) -> dict:
    catalog, catalog_reviews, catalog_ids, catalog_listeners = get_catalog(catalog_path)
    print(f"Catalog: {len(catalog):,} unique albums")

    recs = get_recs(recs_path)
    baseline = get_baseline(bench_path)
    print(
        f"Unique queries (recs): {len(set(query_key(recs))):,} (total recs: {len(recs):,})"
    )
    print(
        f"Unique queries (baseline): {len(set(query_key(baseline))):,} "
        f"(total baseline: {len(baseline):,})"
    )

    print("Embedding recs (matched to baseline):")
    recs_matched = filter_to_baseline(recs, set(query_key(baseline)))

    subsets = build_subsets(recs_matched, baseline, top_n)
    recs_n = subsets["recs_by_n"][top_n]
    baseline_n = subsets["baseline_by_n"][top_n]

    inputs = BenchInputs(
        top_n=top_n,
        catalog_reviews=catalog_reviews,
        catalog_ids=catalog_ids,
        catalog_listeners=catalog_listeners,
        recs_by_n=subsets["recs_by_n"],
        baseline_by_n=subsets["baseline_by_n"],
        recs_tags_by_n=subsets["recs_tags_by_n"],
        baseline_tags_by_n=subsets["baseline_tags_by_n"],
        tag_slots_by_n=subsets["tag_slots_by_n"],
        recs_listeners_by_n=subsets["recs_listeners_by_n"],
        baseline_listeners_by_n=subsets["baseline_listeners_by_n"],
        listener_slots_by_n=subsets["listener_slots_by_n"],
    )
    metrics_df = compute_benchmark_metrics(inputs)

    return {
        "top_n": top_n,
        "metrics_df": metrics_df,
        "recs_n": recs_n,
        "baseline_n": baseline_n,
        "from_cache": False,
    }


def prepare_bench(
    catalog_path: Path | str,
    recs_path: Path | str,
    bench_path: Path | str,
    *,
    top_n: int = 5,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
) -> dict:
    """Load cached benchmark outputs or compute and persist them."""
    catalog_path = Path(catalog_path)
    recs_path = Path(recs_path)
    bench_path = Path(bench_path)
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)

    fingerprint = build_fingerprint(
        catalog_path=catalog_path,
        recs_path=recs_path,
        bench_path=bench_path,
        top_n=top_n,
    )

    if not refresh and cache_is_valid(cache_dir, fingerprint):
        result = load_cache(cache_dir)
        print(f"Loaded cached bench data from {cache_dir}")
        print(f"  metrics_df: {len(result['metrics_df']):,} rows")
        print(
            f"  recs @ n={result['top_n']}: {len(result['recs_n']):,} rows, "
            f"{result['recs_n'].groupby(['query_artist', 'query_album']).ngroups:,} queries"
        )
        print(
            f"  baseline @ n={result['top_n']}: {len(result['baseline_n']):,} rows, "
            f"{result['baseline_n'].groupby(['query_artist', 'query_album']).ngroups:,} queries"
        )
        return result

    if refresh:
        print("REFRESH_CACHE=True — recomputing bench data")
    else:
        print("Cache miss — computing bench data and saving cache")

    result = _compute_bench(catalog_path, recs_path, bench_path, top_n=top_n)
    save_cache(
        cache_dir,
        fingerprint,
        metrics_df=result["metrics_df"],
        recs_n=result["recs_n"],
        baseline_n=result["baseline_n"],
    )
    print(f"Saved cache to {cache_dir}")
    return result
