"""
Data loading and parsing for MIND and EB-NeRD datasets.
Handles downloading, extraction, and conversion to Polars DataFrames.
"""
import io
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import polars as pl
from tqdm import tqdm

from src.config import (
    PROJECT_ROOT, RAW_DIR, PROCESSED_DIR,
    MIND_RAW, MIND_TRAIN_DIR, MIND_DEV_DIR, MIND_LARGE_TEST_DIR,
    MIND_URLS, MIND_BEHAVIOR_COLS, MIND_NEWS_COLS, MIND_PROCESSED,
    EBNERD_RAW, EBNERD_DEMO_DIR, EBNERD_SMALL_DIR, EBNERD_LARGE_DIR,
    EBNERD_TEST_DIR, EBNERD_URLS, EBNERD_PROCESSED,
)


# ═══════════════════════════════════════════════════════════════════════
#  DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════

class _DownloadProgress(tqdm):
    """Progress bar for urlretrieve."""
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def flatten_nested_dir(target_dir: Path, key: Optional[str] = None) -> Path:
    """If target_dir contains a nested directory with news.tsv or behaviors.tsv, move files up."""
    import shutil

    if not target_dir.exists() or not target_dir.is_dir():
        return target_dir

    # If both files are already present directly in target_dir, nothing to flatten
    if (target_dir / "behaviors.tsv").exists() and (target_dir / "news.tsv").exists():
        return target_dir

    # Check candidates for nested directories:
    # 1. key subfolder (e.g. target_dir / MINDsmall_train)
    # 2. Any child subfolder that has news.tsv or behaviors.tsv
    nested_candidates = []
    if key and (target_dir / key).is_dir():
        nested_candidates.append(target_dir / key)
    for child in target_dir.iterdir():
        if child.is_dir() and child not in nested_candidates:
            if (child / "news.tsv").exists() or (child / "behaviors.tsv").exists():
                nested_candidates.append(child)

    for nested in nested_candidates:
        if (nested / "news.tsv").exists() or (nested / "behaviors.tsv").exists():
            print(f"  Unwrapping nested directory: {nested.name} -> {target_dir.name}")
            for item in list(nested.iterdir()):
                dest = target_dir / item.name
                if dest.resolve() == item.resolve():
                    continue
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest, ignore_errors=True)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            try:
                nested.rmdir()
            except Exception:
                pass
            break

    return target_dir


def resolve_mind_dir(data_dir: Path, key: Optional[str] = None) -> Path:
    """Resolve directory containing news.tsv / behaviors.tsv, flattening nested dirs if present."""
    if not data_dir.exists():
        return data_dir

    data_dir = flatten_nested_dir(data_dir, key=key)
    if (data_dir / "news.tsv").exists() or (data_dir / "behaviors.tsv").exists():
        return data_dir

    # Check child folder fallback
    if key and (data_dir / key / "news.tsv").exists():
        return data_dir / key
    if data_dir.is_dir():
        for child in data_dir.iterdir():
            if child.is_dir() and ((child / "news.tsv").exists() or (child / "behaviors.tsv").exists()):
                return child

    return data_dir


def _download_and_extract(url: str, extract_to: Path, name: str):
    """Download a zip URL and extract it."""
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = extract_to / f"{name}.zip"

    # Remove any 0-byte corrupt zip file from interrupted downloads
    if zip_path.exists() and zip_path.stat().st_size == 0:
        try:
            zip_path.unlink()
        except Exception:
            pass

    if not zip_path.exists():
        print(f"  Downloading {name}...")
        try:
            with _DownloadProgress(unit="B", unit_scale=True, desc=name) as pbar:
                urlretrieve(url, zip_path, reporthook=pbar.update_to)
        except Exception:
            if zip_path.exists() and zip_path.stat().st_size == 0:
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            raise
    else:
        print(f"  {name}.zip already exists, skipping download.")

    target_dir = extract_to / name
    if not target_dir.exists() or not any(target_dir.iterdir()):
        print(f"  Extracting {name}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        flatten_nested_dir(target_dir, key=name)
        print(f"  Extracted to {target_dir}")
    else:
        flatten_nested_dir(target_dir, key=name)
        print(f"  {name} already extracted.")


def download_mind(size: str = "small", hf_token: Optional[str] = None):
    """Download MIND dataset files from Hugging Face (yjw1029/MIND).

    Args:
        size: 'small' for dev, 'large' for full training + Codabench test.
        hf_token: Optional Hugging Face token (or uses logged-in token / HF_TOKEN env var).
    """
    import os
    import shutil

    MIND_RAW.mkdir(parents=True, exist_ok=True)
    if size == "small":
        keys = ["MINDsmall_train", "MINDsmall_dev"]
    elif size == "test":
        keys = ["MINDlarge_test"]
    else:
        keys = ["MINDsmall_train", "MINDsmall_dev",
                "MINDlarge_train", "MINDlarge_dev", "MINDlarge_test"]

    alt_dirs = [
        MIND_RAW.parent.parent / "MIND_data",
        MIND_RAW.parent / "MIND_data",
        MIND_RAW,
    ]

    for key in keys:
        target_dir = MIND_RAW / key
        flatten_nested_dir(target_dir, key=key)
        if target_dir.exists() and (target_dir / "behaviors.tsv").exists() and (target_dir / "news.tsv").exists():
            print(f"  {key} already extracted at {target_dir}")
            continue

        zip_path = MIND_RAW / f"{key}.zip"

        # Check alternative locations for zip or pre-extracted folder
        if not zip_path.exists():
            for ad in alt_dirs:
                cand_zip = ad / f"{key}.zip"
                if cand_zip.exists():
                    try:
                        shutil.copy2(cand_zip, zip_path)
                        print(f"  Found {key}.zip in {ad}")
                        break
                    except Exception:
                        pass
                cand_dir = ad / key
                flatten_nested_dir(cand_dir, key=key)
                if cand_dir.exists() and (cand_dir / "behaviors.tsv").exists() and (cand_dir / "news.tsv").exists():
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for f in cand_dir.glob("*"):
                        if f.is_file():
                            shutil.copy2(f, target_dir / f.name)
                    print(f"  Copied {key} from {cand_dir}")
                    break

        flatten_nested_dir(target_dir, key=key)
        if target_dir.exists() and (target_dir / "behaviors.tsv").exists() and (target_dir / "news.tsv").exists():
            continue

        # If zip still does not exist, download it via huggingface_hub
        if not zip_path.exists():
            token = hf_token or os.environ.get("HF_TOKEN")
            try:
                from huggingface_hub import hf_hub_download
                print(f"  Downloading {key} from Hugging Face (yjw1029/MIND)...")
                downloaded = hf_hub_download(
                    repo_id="yjw1029/MIND",
                    repo_type="dataset",
                    filename=f"{key}.zip",
                    local_dir=str(MIND_RAW),
                    token=token,
                )
                zip_path = Path(downloaded)
            except Exception as e:
                print(f"  ⚠️  hf_hub_download failed: {e}")
                print(f"  Attempting direct URL download for {key}...")
                try:
                    _download_and_extract(MIND_URLS[key], MIND_RAW, key)
                    continue
                except Exception as e2:
                    print(f"\n❌ Could not download {key}: {e2}")
                    print("  Please log into Hugging Face in Colab via:")
                    print("    from huggingface_hub import login; login()")
                    print("  Or run:")
                    print("    !hf download yjw1029/MIND --repo-type dataset --local-dir data/raw/mind\n")
                    raise

        # Extract zip into target_dir
        if zip_path.exists() and not ((target_dir / "behaviors.tsv").exists() and (target_dir / "news.tsv").exists()):
            print(f"  Extracting {key}...")
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
            flatten_nested_dir(target_dir, key=key)
            print(f"  Extracted to {target_dir}")


def download_ebnerd(size: str = "demo"):
    """Download EB-NeRD dataset files.

    Args:
        size: 'demo' for quick dev, 'small' for dev, 'large' for Codabench.
    """
    EBNERD_RAW.mkdir(parents=True, exist_ok=True)
    if size == "demo":
        keys = ["ebnerd_demo"]
    elif size == "small":
        keys = ["ebnerd_small"]
    else:
        keys = ["ebnerd_large", "ebnerd_testset"]
    for key in keys:
        if key in EBNERD_URLS:
            _download_and_extract(EBNERD_URLS[key], EBNERD_RAW, key)


# ═══════════════════════════════════════════════════════════════════════
#  MIND PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_mind_news(data_dir: Path) -> pl.DataFrame:
    """Parse MIND news.tsv → Polars DataFrame."""
    data_dir = resolve_mind_dir(data_dir)
    news_path = data_dir / "news.tsv"
    if not news_path.exists():
        items = [f.name for f in data_dir.iterdir()] if data_dir.exists() else "Directory does not exist"
        raise FileNotFoundError(f"news.tsv not found in {data_dir}. Available items: {items}")
    df = pl.read_csv(
        news_path, separator="\t", has_header=False,
        new_columns=MIND_NEWS_COLS, infer_schema_length=0,
        quote_char=None, truncate_ragged_lines=True,
    )
    return df


def parse_mind_behaviors(data_dir: Path, is_test: bool = False) -> pl.DataFrame:
    """Parse MIND behaviors.tsv → Polars DataFrame.

    Returns columns: impression_id, user_id, time, history (list),
                     candidates (list), labels (list or None for test).
    """
    data_dir = resolve_mind_dir(data_dir)
    beh_path = data_dir / "behaviors.tsv"
    if not beh_path.exists():
        items = [f.name for f in data_dir.iterdir()] if data_dir.exists() else "Directory does not exist"
        raise FileNotFoundError(f"behaviors.tsv not found in {data_dir}. Available items: {items}")
    df = pl.read_csv(
        beh_path, separator="\t", has_header=False,
        new_columns=MIND_BEHAVIOR_COLS, infer_schema_length=0,
        quote_char=None, truncate_ragged_lines=True,
    )

    # Parse history column: "N1 N2 N3" → ["N1", "N2", "N3"], or empty list if None/blank
    df = df.with_columns(
        pl.when(pl.col("history").is_not_null() & (pl.col("history").str.strip_chars() != ""))
        .then(pl.col("history").str.strip_chars().str.split(" "))
        .otherwise(pl.lit([], dtype=pl.List(pl.String)))
        .alias("history")
    )

    if is_test:
        # Test set: impressions = "N1 N2 N3" (or "N1-1 N2-0" if using dev set as test)
        df = df.with_columns(
            pl.col("impressions").str.split(" ").alias("candidates")
        ).with_columns(
            pl.col("candidates").list.eval(pl.element().str.split("-").list.first()).alias("candidates")
        ).drop("impressions")
        return df

    # Train/dev: impressions = "N1-1 N2-0 N3-1" → candidates + labels
    # Vectorized extraction without explode/group_by to preserve ordering and minimize memory:
    df = df.with_columns(
        pl.col("impressions").str.split(" ").alias("imp_list")
    ).with_columns(
        pl.col("imp_list").list.eval(pl.element().str.split("-").list.first()).alias("candidates"),
        pl.col("imp_list").list.eval(pl.element().str.split("-").list.last().cast(pl.Int8)).alias("labels"),
    ).drop("imp_list", "impressions")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  EB-NeRD PARSING
# ═══════════════════════════════════════════════════════════════════════

def _find_ebnerd_base(data_dir: Path) -> Path:
    """Find the actual EB-NeRD base directory (may be nested)."""
    if (data_dir / "train").exists() or (data_dir / "validation").exists():
        return data_dir
    # Check one level deeper
    for child in data_dir.iterdir():
        if child.is_dir() and ((child / "train").exists() or (child / "validation").exists()):
            return child
    return data_dir


def parse_ebnerd_articles(data_dir: Path) -> pl.DataFrame:
    """Parse EB-NeRD articles.parquet."""
    base = _find_ebnerd_base(data_dir)
    # Try multiple possible locations
    for candidate in [base / "articles.parquet",
                      data_dir / "articles.parquet"]:
        if candidate.exists():
            return pl.read_parquet(candidate)
    raise FileNotFoundError(f"Cannot find articles.parquet under {data_dir}")


def parse_ebnerd_behaviors(data_dir: Path, split: str = "train",
                           is_test: bool = False) -> pl.DataFrame:
    """Parse EB-NeRD behaviors.parquet for a given split.

    Returns columns: impression_id, user_id, timestamp,
                     candidates (list), labels (list or None for test).
    """
    base = _find_ebnerd_base(data_dir)
    beh_path = base / split / "behaviors.parquet"
    df = pl.read_parquet(beh_path)

    # Standardize column names
    rename_map = {}
    if "impression_time" in df.columns:
        rename_map["impression_time"] = "timestamp"
    if "article_ids_inview" in df.columns:
        rename_map["article_ids_inview"] = "candidates"
    if "article_ids_clicked" in df.columns:
        rename_map["article_ids_clicked"] = "clicked"
    df = df.rename(rename_map)

    # Select relevant columns
    if is_test or "clicked" not in df.columns:
        cols = ["impression_id", "user_id", "timestamp", "candidates"]
        return df.select([c for c in cols if c in df.columns])

    # Build labels: 1 if candidate was clicked, 0 otherwise (preserves exact candidate order)
    cands_series = df["candidates"].to_list()
    clicked_series = df["clicked"].to_list()
    labels = [
        [1 if c in (set(clk) if clk is not None else set()) else 0 for c in (cands or [])]
        for cands, clk in zip(cands_series, clicked_series)
    ]
    df = df.with_columns(pl.Series("labels", labels, dtype=pl.List(pl.Int8)))
    return df.select(["impression_id", "user_id", "timestamp", "candidates", "labels"])


def parse_ebnerd_history(data_dir: Path, split: str = "train") -> pl.DataFrame:
    """Parse EB-NeRD history.parquet → user history DataFrame."""
    base = _find_ebnerd_base(data_dir)
    hist_path = base / split / "history.parquet"
    hist = pl.read_parquet(hist_path)
    return hist.select(
        "user_id",
        pl.col("article_id_fixed").alias("history"),
        pl.col("impression_time_fixed").alias("history_timestamps"),
    )


# ═══════════════════════════════════════════════════════════════════════
#  FEATURE STORE BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def build_mind_feature_store(mind_train_dir: Path = MIND_TRAIN_DIR,
                              mind_dev_dir: Path = MIND_DEV_DIR,
                              output_dir: Path = MIND_PROCESSED):
    """Build and persist MIND feature store (articles, behaviors, popularity)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n🔧 Building MIND feature store...")

    # Flatten nested dirs if present
    mind_train_dir = resolve_mind_dir(mind_train_dir, key="MINDsmall_train")
    mind_dev_dir = resolve_mind_dir(mind_dev_dir, key="MINDsmall_dev")

    # Fallback to alternative locations if default paths don't exist
    for cand in [mind_train_dir, PROJECT_ROOT / "MIND_data" / "MINDsmall_train", PROJECT_ROOT / "data" / "MIND_data" / "MINDsmall_train"]:
        cand = resolve_mind_dir(cand, key="MINDsmall_train")
        if (cand / "news.tsv").exists():
            mind_train_dir = cand
            break
    for cand in [mind_dev_dir, PROJECT_ROOT / "MIND_data" / "MINDsmall_dev", PROJECT_ROOT / "data" / "MIND_data" / "MINDsmall_dev"]:
        cand = resolve_mind_dir(cand, key="MINDsmall_dev")
        if (cand / "news.tsv").exists():
            mind_dev_dir = cand
            break

    # Articles (combine train + dev)
    news_train = parse_mind_news(mind_train_dir)
    news_dev = parse_mind_news(mind_dev_dir)
    articles = pl.concat([news_train, news_dev]).unique(subset=["news_id"])
    articles.write_parquet(output_dir / "articles.parquet")
    print(f"  Articles: {articles.height:,} unique")

    # Train behaviors
    train_beh = parse_mind_behaviors(mind_train_dir, is_test=False)
    train_beh.write_parquet(output_dir / "train_behaviors.parquet")
    print(f"  Train behaviors: {train_beh.height:,} impressions")

    # Dev behaviors
    dev_beh = parse_mind_behaviors(mind_dev_dir, is_test=False)
    dev_beh.write_parquet(output_dir / "dev_behaviors.parquet")
    print(f"  Dev behaviors: {dev_beh.height:,} impressions")

    # User histories from train
    user_histories = (
        train_beh
        .filter(pl.col("history").list.len() > 0)
        .group_by("user_id")
        .agg(pl.col("history").first())
        .unique(subset=["user_id"])
    )
    user_histories.write_parquet(output_dir / "user_histories.parquet")
    print(f"  User histories: {user_histories.height:,} users")

    # Article popularity
    popularity = (
        train_beh
        .explode(["candidates", "labels"])
        .filter(pl.col("labels") == 1)
        .group_by("candidates")
        .agg(pl.len().alias("click_count"))
        .rename({"candidates": "news_id"})
        .sort("click_count", descending=True)
    )
    popularity.write_parquet(output_dir / "popularity.parquet")
    print(f"  Popularity: {popularity.height:,} articles with clicks")

    print("✅ MIND feature store built\n")
    return {
        "articles": articles,
        "train_behaviors": train_beh,
        "dev_behaviors": dev_beh,
        "user_histories": user_histories,
        "popularity": popularity,
    }


def build_ebnerd_feature_store(ebnerd_dir: Path = EBNERD_DEMO_DIR,
                                output_dir: Path = EBNERD_PROCESSED):
    """Build and persist EB-NeRD feature store."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n🔧 Building EB-NeRD feature store...")

    articles = parse_ebnerd_articles(ebnerd_dir)
    if "category_str" in articles.columns and "category" in articles.columns:
        articles = articles.with_columns(
            pl.col("category_str").fill_null(pl.col("category").cast(pl.String)).alias("category")
        )
    elif "category" in articles.columns:
        articles = articles.with_columns(pl.col("category").cast(pl.String))
    articles.write_parquet(output_dir / "articles.parquet")
    print(f"  Articles: {articles.height:,}")

    train_beh = parse_ebnerd_behaviors(ebnerd_dir, split="train", is_test=False)
    train_beh.write_parquet(output_dir / "train_behaviors.parquet")
    print(f"  Train behaviors: {train_beh.height:,} impressions")

    val_beh = parse_ebnerd_behaviors(ebnerd_dir, split="validation", is_test=False)
    val_beh.write_parquet(output_dir / "val_behaviors.parquet")
    print(f"  Validation behaviors: {val_beh.height:,} impressions")

    history = parse_ebnerd_history(ebnerd_dir, split="train")
    history.write_parquet(output_dir / "user_histories.parquet")
    print(f"  User histories: {history.height:,} users")

    popularity = (
        train_beh
        .explode(["candidates", "labels"])
        .filter(pl.col("labels") == 1)
        .group_by("candidates")
        .agg(pl.len().alias("click_count"))
        .rename({"candidates": "article_id"})
        .sort("click_count", descending=True)
    )
    popularity.write_parquet(output_dir / "popularity.parquet")
    print(f"  Popularity: {popularity.height:,} articles with clicks")

    print("✅ EB-NeRD feature store built\n")
    return {
        "articles": articles,
        "train_behaviors": train_beh,
        "val_behaviors": val_beh,
        "user_histories": history,
        "popularity": popularity,
    }


# ═══════════════════════════════════════════════════════════════════════
#  TOP-LEVEL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_data_pipeline(dataset: str = "all", size: str = "small"):
    """Download data and build feature stores.

    Args:
        dataset: 'mind', 'ebnerd', or 'all'
        size:    'small'/'demo' for dev, 'large' for Codabench
    """
    print("=" * 60)
    print(" IRE Assignment 2 — Data Pipeline")
    print("=" * 60)

    if dataset in ("mind", "all"):
        if size == "test":
            download_mind(size="test")
        else:
            mind_size = "small" if size in ("small", "demo") else "large"
            download_mind(size=mind_size)
            build_mind_feature_store()

    if dataset in ("ebnerd", "all") and size != "test":
        ebnerd_size = "demo" if size in ("small", "demo") else "large"
        download_ebnerd(size=ebnerd_size)
        ebnerd_dir = {
            "demo": EBNERD_DEMO_DIR,
            "small": EBNERD_SMALL_DIR,
            "large": EBNERD_LARGE_DIR,
        }[ebnerd_size]
        build_ebnerd_feature_store(ebnerd_dir=ebnerd_dir)

    print("=" * 60)
    print(" Data pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IRE A2 — Data Pipeline")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--size", choices=["small", "demo", "large", "test"], default="small")
    args = parser.parse_args()
    run_data_pipeline(dataset=args.dataset, size=args.size)
