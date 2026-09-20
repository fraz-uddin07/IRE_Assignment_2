"""
IRE Assignment 2 — One-Command Reproduce Pipeline.

Usage:
    python run_pipeline.py --dataset mind --size small
    python run_pipeline.py --dataset ebnerd --size demo
    python run_pipeline.py --dataset all --size small
    python run_pipeline.py --dataset mind --size small --skip-download
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def run_pipeline(dataset: str, size: str, skip_download: bool = False,
                 max_train: int = None, max_val: int = None):
    """Run the full Assignment 2 pipeline.

    Steps:
        1. Download & parse data → feature store
        2. Build behavioural features
        3. Train LightGBM re-ranker
        4. Evaluate: baseline vs improved vs ablated
        5. Serving & scale analysis
        6. Generate Codabench predictions
    """
    from src.data_loader import run_data_pipeline
    from src.baseline import run_ablation
    from src.serving_analysis import run_serving_analysis
    from src.config import MODELS_DIR

    print("=" * 60)
    print(" IRE Assignment 2 — Full Pipeline")
    print(f" Dataset: {dataset.upper()}, Size: {size}")
    print("=" * 60)

    datasets = [dataset] if dataset != "all" else ["mind", "ebnerd"]

    # ── Step 1: Data Pipeline ───────────────────────────────────────
    if not skip_download:
        print("\n" + "─" * 60)
        print(" STEP 1: Data Pipeline (download + feature store)")
        print("─" * 60)
        run_data_pipeline(dataset=dataset, size=size)
    else:
        print("\n⏭️  Skipping download (--skip-download)")

    # ── Steps 2-5: Per-dataset ──────────────────────────────────────
    for ds in datasets:
        print(f"\n{'═' * 60}")
        print(f" Processing: {ds.upper()}")
        print(f"{'═' * 60}")

        # Step 2-3: Ablation study (includes baseline, training, evaluation)
        print("\n" + "─" * 60)
        print(f" STEPS 2-3: Ablation Study ({ds.upper()})")
        print("─" * 60)
        ablation_results = run_ablation(
            dataset=ds,
            max_train_impressions=max_train,
            max_val_impressions=max_val,
        )

        # Step 4: Serving analysis
        print("\n" + "─" * 60)
        print(f" STEP 4: Serving & Scale Analysis ({ds.upper()})")
        print("─" * 60)

        # Load the trained model for latency benchmarking
        import lightgbm as lgb
        model_path = MODELS_DIR / f"reranker_{ds}_full.txt"
        model = None
        X_val, meta_val = None, None
        if model_path.exists():
            model = lgb.Booster(model_file=str(model_path))

            # Rebuild validation features for latency test
            # (use a small subset)
            from src.config import MIND_PROCESSED, EBNERD_PROCESSED
            from src.feature_engineering import build_features_for_dataset
            import polars as pl

            id_col = "news_id" if ds == "mind" else "article_id"
            proc = MIND_PROCESSED if ds == "mind" else EBNERD_PROCESSED
            dev_key = "dev_behaviors.parquet" if ds == "mind" else "val_behaviors.parquet"

            articles = pl.read_parquet(proc / "articles.parquet")
            val_beh = pl.read_parquet(proc / dev_key)
            pop_df = pl.read_parquet(proc / "popularity.parquet")

            user_hist_df = pl.read_parquet(proc / "user_histories.parquet")
            user_histories = {}
            for row in user_hist_df.iter_rows(named=True):
                hist = row.get("history")
                if hist:
                    user_histories[row["user_id"]] = hist

            X_val, _, meta_val = build_features_for_dataset(
                val_beh, articles, pop_df, user_histories,
                dataset=ds, max_impressions=200,
            )

        run_serving_analysis(
            dataset=ds, model=model,
            X_val=X_val, meta_val=meta_val,
        )

    # ── Step 5: Anti-gaming tests ───────────────────────────────────
    print("\n" + "─" * 60)
    print(" STEP 5: Anti-Gaming Tests")
    print("─" * 60)
    from tests.test_no_leakage import (
        test_mind_no_temporal_leakage,
        test_ebnerd_no_temporal_leakage,
        test_features_use_only_past_clicks,
        test_no_future_features_in_test,
    )
    for ds in datasets:
        if ds == "mind":
            test_mind_no_temporal_leakage()
        else:
            test_ebnerd_no_temporal_leakage()
    test_features_use_only_past_clicks()
    test_no_future_features_in_test()

    print("\n" + "=" * 60)
    print(" ✅ Pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review outputs/ for evaluation results and serving analysis")
    print("  2. To generate Codabench predictions:")
    print("     python -m src.predict --dataset mind --method reranker")
    print("     python -m src.predict --dataset ebnerd --method reranker")
    print("  3. Upload .zip files to Codabench leaderboards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IRE Assignment 2 — One-Command Reproduce Pipeline")
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all",
                        help="Dataset to process")
    parser.add_argument("--size", choices=["small", "demo", "large"], default="small",
                        help="Dataset size: small/demo for dev, large for Codabench")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use existing data)")
    parser.add_argument("--max-train", type=int, default=25000,
                        help="Limit training impressions (default: 25000 for fast run; -1 for unlimited)")
    parser.add_argument("--max-val", type=int, default=25000,
                        help="Limit validation impressions (default: 25000 for fast run; -1 for unlimited)")
    args = parser.parse_args()

    run_pipeline(
        dataset=args.dataset,
        size=args.size,
        skip_download=args.skip_download,
        max_train=args.max_train,
        max_val=args.max_val,
    )
