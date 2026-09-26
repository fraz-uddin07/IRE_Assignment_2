"""
Prediction file generation for Codabench submissions.

Generates prediction files for both MIND and EB-NeRD test sets,
with batch processing for memory efficiency.
"""
import gc
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import (
    MIND_PROCESSED, MIND_LARGE_TEST_DIR,
    EBNERD_PROCESSED, EBNERD_TEST_DIR,
    OUTPUTS_DIR, MODELS_DIR,
)
from src.feature_engineering import (
    ALL_FEATURE_NAMES,
    build_article_lookup, build_popularity_lookup,
    build_category_popularity, build_publish_time_lookup,
    compute_features_for_impression,
)


# ═══════════════════════════════════════════════════════════════════════
#  RANKING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def rank_by_scores(scores: List[float]) -> List[int]:
    """Convert scores to 1-based ranks (highest score = rank 1)."""
    order = np.argsort(scores)[::-1]
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks


def rank_by_popularity(candidates: List, popularity: Dict) -> List[int]:
    """Rank candidates by popularity count (descending)."""
    scores = [popularity.get(c, 0) for c in candidates]
    return rank_by_scores(scores)


# ═══════════════════════════════════════════════════════════════════════
#  MIND PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_mind_predictions(
    test_dir: Path,
    output_path: Path,
    model=None,
    method: str = "popularity",
):
    """Generate MIND predictions for Codabench.

    Args:
        test_dir: Path to MINDlarge_test directory.
        output_path: Path for output prediction.txt.
        model: Trained LightGBM model (for reranker method).
        method: 'popularity' or 'reranker'.
    """
    print(f"\n🔮 Generating MIND predictions ({method})...")

    proc = MIND_PROCESSED
    pop_df = pl.read_parquet(proc / "popularity.parquet")
    popularity = dict(zip(pop_df["news_id"].to_list(), pop_df["click_count"].to_list()))

    articles = pl.read_parquet(proc / "articles.parquet")

    # Load user histories
    user_hist_df = pl.read_parquet(proc / "user_histories.parquet")
    user_histories = {}
    for row in user_hist_df.iter_rows(named=True):
        hist = row.get("history")
        if hist:
            user_histories[row["user_id"]] = hist

    # For reranker: build lookups
    article_lookup, cat_pop, pub_times = None, None, None
    if method == "reranker" and model is not None:
        article_lookup = build_article_lookup(articles, id_col="news_id")
        cat_pop = build_category_popularity(pop_df, articles, id_col="news_id")
        pub_times = build_publish_time_lookup(articles, id_col="news_id")

    from src.config import MIND_RAW, MIND_DEV_DIR
    from src.data_loader import parse_mind_behaviors, resolve_mind_dir, flatten_nested_dir

    # Check if a zip file for MINDlarge_test is present and needs unzipping
    for zip_loc in [
        MIND_RAW / "MINDlarge_test.zip",
        Path("data/raw/mind/MINDlarge_test.zip"),
        Path("MIND_data/MINDlarge_test.zip"),
        Path("data/MIND_data/MINDlarge_test.zip"),
    ]:
        if not (test_dir / "behaviors.tsv").exists() and zip_loc.exists():
            print(f"  Found {zip_loc}, extracting...")
            test_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_loc, "r") as zf:
                zf.extractall(test_dir)
            flatten_nested_dir(test_dir, key="MINDlarge_test")
            break

    test_dir = resolve_mind_dir(test_dir, key="MINDlarge_test")

    # Read test behaviors
    beh_path = test_dir / "behaviors.tsv"
    if not beh_path.exists():
        # Check if dev set is available as fallback
        dev_cand = resolve_mind_dir(MIND_RAW / "MINDsmall_dev", key="MINDsmall_dev")
        if not (dev_cand / "behaviors.tsv").exists():
            dev_cand = resolve_mind_dir(MIND_DEV_DIR, key="MINDsmall_dev")

        if (dev_cand / "behaviors.tsv").exists():
            print(f"  ⚠️  MINDlarge_test not found. Falling back to dev set ({dev_cand.name}) for predictions.")
            print(f"     (Note: For official Codabench submission, run: !python -m src.data_loader --dataset mind --size large)")
            test_dir = dev_cand
            beh_path = test_dir / "behaviors.tsv"
        else:
            print(f"\n⚠️  Test behaviors not found at {beh_path}")
            print("   For official Codabench submission, MIND requires MINDlarge_test.zip (~140 MB).")
            print("   Download with: !python -m src.data_loader --dataset mind --size large")
            print("   Or run: !python -m src.predict --dataset mind --method reranker --test-dir data/raw/mind/MINDsmall_dev\n")
            return None

    test_beh = parse_mind_behaviors(test_dir, is_test=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_written = 0

    with open(output_path, "w") as f:
        for row in tqdm(test_beh.iter_rows(named=True), total=len(test_beh),
                        desc="  MIND predictions", unit="imp"):
            imp_id = row["impression_id"]
            candidates = row.get("candidates", [])
            if candidates is None:
                candidates = []

            if method == "reranker" and model is not None:
                user_id = row["user_id"]
                history = user_histories.get(user_id, [])

                features = compute_features_for_impression(
                    user_id=user_id,
                    candidates=candidates,
                    impression_time=row.get("time"),
                    user_history=history,
                    history_timestamps=None,
                    article_lookup=article_lookup,
                    popularity_lookup=popularity,
                    category_popularity=cat_pop,
                    publish_time_lookup=pub_times,
                )
                X = np.array([[feat[name] for name in ALL_FEATURE_NAMES]
                              for feat in features], dtype=np.float32)
                scores = model.predict(X, num_iteration=model.best_iteration).tolist()
                ranks = rank_by_scores(scores)
            else:
                ranks = rank_by_popularity(candidates, popularity)

            f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
            predictions_written += 1

    print(f"  Done! {predictions_written:,} predictions → {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
#  EB-NeRD PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_ebnerd_predictions(
    test_dir: Path,
    output_path: Path,
    model=None,
    method: str = "popularity",
):
    """Generate EB-NeRD predictions for Codabench (batch processing)."""
    print(f"\n🔮 Generating EB-NeRD predictions ({method})...")

    proc = EBNERD_PROCESSED
    pop_df = pl.read_parquet(proc / "popularity.parquet")
    popularity = dict(zip(pop_df["article_id"].to_list(), pop_df["click_count"].to_list()))

    articles = pl.read_parquet(proc / "articles.parquet")

    # User histories
    user_hist_df = pl.read_parquet(proc / "user_histories.parquet")
    user_histories = {}
    for row in user_hist_df.iter_rows(named=True):
        hist = row.get("history")
        if hist:
            user_histories[row["user_id"]] = hist

    # For reranker
    article_lookup, cat_pop, pub_times = None, None, None
    if method == "reranker" and model is not None:
        article_lookup = build_article_lookup(articles, id_col="article_id")
        cat_pop = build_category_popularity(pop_df, articles, id_col="article_id")
        pub_times = build_publish_time_lookup(articles, id_col="article_id")

    # Find test behaviors
    from src.data_loader import _find_ebnerd_base
    base = _find_ebnerd_base(test_dir)
    test_beh_path = None
    for split_name in ["test", "validation"]:
        candidate = base / split_name / "behaviors.parquet"
        if candidate.exists():
            test_beh_path = candidate
            break
    if test_beh_path is None:
        print(f"  ⚠️  Test behaviors not found under {test_dir}")
        return

    pf = pq.ParquetFile(test_beh_path)
    n_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"  Processing {total_rows:,} impressions across {n_row_groups} row groups...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_written = 0

    pbar = tqdm(total=total_rows, desc="  EB-NeRD predictions", unit="imp")

    with open(output_path, "w") as f:
        for rg_idx in range(n_row_groups):
            batch = pl.from_arrow(pf.read_row_group(rg_idx))

            for row in batch.iter_rows(named=True):
                imp_id = row["impression_id"]
                candidates = row.get("article_ids_inview", [])
                if candidates is None:
                    candidates = []

                if method == "reranker" and model is not None:
                    user_id = row["user_id"]
                    history = user_histories.get(user_id, [])

                    features = compute_features_for_impression(
                        user_id=user_id,
                        candidates=candidates,
                        impression_time=row.get("impression_time"),
                        user_history=history,
                        history_timestamps=None,
                        article_lookup=article_lookup,
                        popularity_lookup=popularity,
                        category_popularity=cat_pop,
                        publish_time_lookup=pub_times,
                    )
                    X = np.array([[feat[name] for name in ALL_FEATURE_NAMES]
                                  for feat in features], dtype=np.float32)
                    scores = model.predict(X, num_iteration=model.best_iteration).tolist()
                    ranks = rank_by_scores(scores)
                else:
                    ranks = rank_by_popularity(candidates, popularity)

                f.write(f"{imp_id} [{','.join(map(str, ranks))}]\n")
                predictions_written += 1
                pbar.update(1)

            del batch
            gc.collect()

    pbar.close()

    print(f"  Done! {predictions_written:,} predictions → {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
#  ZIP FOR CODABENCH
# ═══════════════════════════════════════════════════════════════════════

def zip_predictions(prediction_path: Path) -> Path:
    """Zip prediction file for Codabench upload."""
    zip_path = prediction_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, "prediction.txt")
    size_mb = zip_path.stat().st_size / 1e6
    print(f"  📦 Created {zip_path} ({size_mb:.1f} MB)")
    return zip_path


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Codabench predictions")
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--method", choices=["popularity", "reranker"], default="popularity")
    parser.add_argument("--test-dir", type=str, default=None)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model = None
    if args.method == "reranker":
        import lightgbm as lgb
        model_path = MODELS_DIR / f"reranker_{args.dataset}_full.txt"
        if model_path.exists():
            model = lgb.Booster(model_file=str(model_path))
            print(f"  📂 Loaded model from {model_path}")
        else:
            print(f"  ⚠️  No model found at {model_path}, falling back to popularity")
            args.method = "popularity"

    if args.dataset == "mind":
        test_dir = Path(args.test_dir) if args.test_dir else MIND_LARGE_TEST_DIR
        out = OUTPUTS_DIR / f"mind_prediction_{args.method}.txt"
        res = generate_mind_predictions(test_dir, out, model=model, method=args.method)
        if res and out.exists():
            zip_predictions(out)
            print("\n✅ Done! Upload the .zip file to Codabench.")
        else:
            print(f"\n⚠️  Skipped zip creation: {out} was not generated.")
    else:
        if args.test_dir:
            test_dir = Path(args.test_dir)
        elif EBNERD_TEST_DIR.exists():
            test_dir = EBNERD_TEST_DIR
        else:
            # Fallback: use the best available dataset's validation split
            from src.config import EBNERD_LARGE_DIR, EBNERD_SMALL_DIR, EBNERD_DEMO_DIR
            fallback = None
            for fb_dir in [EBNERD_LARGE_DIR, EBNERD_SMALL_DIR, EBNERD_DEMO_DIR]:
                if fb_dir.exists():
                    fallback = fb_dir
                    break
            if fallback is None:
                print("  ❌ No EB-NeRD data found at all. Run data_loader first.")
                return
            print(f"  ⚠️  ebnerd_testset not found, using validation split from {fallback.name}")
            test_dir = fallback

        out = OUTPUTS_DIR / f"ebnerd_prediction_{args.method}.txt"
        res = generate_ebnerd_predictions(test_dir, out, model=model, method=args.method)
        if res and out.exists():
            zip_predictions(out)
            print("\n✅ Done! Upload the .zip file to Codabench.")
        else:
            print(f"\n⚠️  Skipped zip creation: {out} was not generated.")


if __name__ == "__main__":
    main()
