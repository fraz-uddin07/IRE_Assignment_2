"""
Q3 — Baseline Reproduced, Then Beaten.

1. Reproduce the popularity baseline on both datasets.
2. Train LightGBM re-ranker (all features) = improved model.
3. Ablation: Train without category-aware features → measure delta.
4. Paired bootstrap 95% CI on the improvement (must exclude zero).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from src.config import MIND_PROCESSED, EBNERD_PROCESSED, MODELS_DIR
from src.candidate_generation import generate_candidate_scores
from src.evaluation import (
    compute_auc, compute_mrr, ndcg_at_k,
    paired_bootstrap_ci, run_full_evaluation, print_results,
)
from src.feature_engineering import (
    ALL_FEATURE_NAMES, BASE_FEATURE_NAMES,
    build_features_for_dataset,
)
from src.reranker import (
    train_reranker, rerank_impressions, save_model,
    print_feature_importance, select_feature_columns,
)


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1: REPRODUCE BASELINE
# ═══════════════════════════════════════════════════════════════════════

def reproduce_baseline(dataset: str = "mind") -> Tuple[List[Dict], Dict]:
    """Reproduce the popularity baseline and evaluate it.

    Returns:
        rankings: List of impression rankings (popularity-scored).
        eval_results: Evaluation metrics dict.
    """
    id_col = "news_id" if dataset == "mind" else "article_id"
    proc = MIND_PROCESSED if dataset == "mind" else EBNERD_PROCESSED
    dev_key = "dev_behaviors.parquet" if dataset == "mind" else "val_behaviors.parquet"

    print(f"\n{'='*60}")
    print(f" Step 1: Reproduce Popularity Baseline ({dataset.upper()})")
    print(f"{'='*60}")

    articles = pl.read_parquet(proc / "articles.parquet")
    dev_beh = pl.read_parquet(proc / dev_key)
    pop_df = pl.read_parquet(proc / "popularity.parquet")
    popularity = dict(zip(pop_df[id_col].to_list(), pop_df["click_count"].to_list()))
    total_interactions = int(sum(popularity.values()))

    # User history lengths for slicing
    user_hist_df = pl.read_parquet(proc / "user_histories.parquet")
    user_history_lens = {}
    for row in user_hist_df.iter_rows(named=True):
        hist = row.get("history")
        if hist:
            user_history_lens[row["user_id"]] = len(hist)

    # Generate popularity-based rankings
    rankings = generate_candidate_scores(dev_beh, popularity, dataset=dataset)

    # Evaluate
    eval_results = run_full_evaluation(
        rankings=rankings,
        dataset=dataset,
        popularity=popularity,
        user_history_lens=user_history_lens,
        total_articles=articles.height,
        total_interactions=total_interactions,
        method_name="popularity_baseline",
    )

    return rankings, eval_results


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2: TRAIN IMPROVED MODEL
# ═══════════════════════════════════════════════════════════════════════

def train_improved_model(
    dataset: str = "mind",
    max_train_impressions: Optional[int] = None,
    max_val_impressions: Optional[int] = None,
) -> Tuple:
    """Train the full LightGBM re-ranker with all features.

    Returns:
        model: Trained LightGBM booster.
        X_val, y_val, meta_val: Validation data for re-ranking.
        val_beh: Validation behaviors DataFrame.
        popularity, user_history_lens, articles: For evaluation.
    """
    id_col = "news_id" if dataset == "mind" else "article_id"
    proc = MIND_PROCESSED if dataset == "mind" else EBNERD_PROCESSED
    dev_key = "dev_behaviors.parquet" if dataset == "mind" else "val_behaviors.parquet"

    print(f"\n{'='*60}")
    print(f" Step 2: Train Improved Model ({dataset.upper()})")
    print(f"{'='*60}")

    # Load data
    articles = pl.read_parquet(proc / "articles.parquet")
    train_beh = pl.read_parquet(proc / "train_behaviors.parquet")
    val_beh = pl.read_parquet(proc / dev_key)
    pop_df = pl.read_parquet(proc / "popularity.parquet")
    popularity = dict(zip(pop_df[id_col].to_list(), pop_df["click_count"].to_list()))
    total_interactions = int(sum(popularity.values()))

    # User histories
    user_hist_df = pl.read_parquet(proc / "user_histories.parquet")
    user_histories = {}
    user_history_lens = {}
    history_timestamps_map = {}
    for row in user_hist_df.iter_rows(named=True):
        uid = row["user_id"]
        hist = row.get("history")
        if hist:
            user_histories[uid] = hist
            user_history_lens[uid] = len(hist)
        hist_ts = row.get("history_timestamps")
        if hist_ts:
            history_timestamps_map[uid] = hist_ts

    # Build training features
    print("\n📐 Building training features...")
    X_train, y_train, meta_train = build_features_for_dataset(
        behaviors_df=train_beh,
        articles_df=articles,
        popularity_df=pop_df,
        user_histories=user_histories,
        history_timestamps_map=history_timestamps_map if history_timestamps_map else None,
        dataset=dataset,
        max_impressions=max_train_impressions,
    )

    # Build validation features
    print("\n📐 Building validation features...")
    X_val, y_val, meta_val = build_features_for_dataset(
        behaviors_df=val_beh,
        articles_df=articles,
        popularity_df=pop_df,
        user_histories=user_histories,
        history_timestamps_map=history_timestamps_map if history_timestamps_map else None,
        dataset=dataset,
        max_impressions=max_val_impressions,
    )

    # Train full model
    model = train_reranker(X_train, y_train, X_val, y_val,
                            feature_names=ALL_FEATURE_NAMES)
    print_feature_importance(model)

    # Save model
    model_path = MODELS_DIR / f"reranker_{dataset}_full.txt"
    save_model(model, model_path)

    return (model, X_val, y_val, meta_val, val_beh,
            popularity, user_history_lens, articles, total_interactions)


# ═══════════════════════════════════════════════════════════════════════
#  STEP 3: ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════

def run_ablation(
    dataset: str = "mind",
    max_train_impressions: Optional[int] = None,
    max_val_impressions: Optional[int] = None,
) -> Dict:
    """Run the full ablation study: baseline vs improved vs ablated.

    1. Popularity baseline (no ML).
    2. LightGBM with ALL features (improved).
    3. LightGBM WITHOUT category features (ablated).
    4. Paired bootstrap CI on (improved - ablated) and (improved - baseline).
    """
    id_col = "news_id" if dataset == "mind" else "article_id"
    proc = MIND_PROCESSED if dataset == "mind" else EBNERD_PROCESSED
    dev_key = "dev_behaviors.parquet" if dataset == "mind" else "val_behaviors.parquet"

    # ── Load data ───────────────────────────────────────────────────
    articles = pl.read_parquet(proc / "articles.parquet")
    train_beh = pl.read_parquet(proc / "train_behaviors.parquet")
    val_beh = pl.read_parquet(proc / dev_key)
    pop_df = pl.read_parquet(proc / "popularity.parquet")
    popularity = dict(zip(pop_df[id_col].to_list(), pop_df["click_count"].to_list()))
    total_interactions = int(sum(popularity.values()))

    user_hist_df = pl.read_parquet(proc / "user_histories.parquet")
    user_histories, user_history_lens, history_timestamps_map = {}, {}, {}
    for row in user_hist_df.iter_rows(named=True):
        uid = row["user_id"]
        hist = row.get("history")
        if hist:
            user_histories[uid] = hist
            user_history_lens[uid] = len(hist)
        hist_ts = row.get("history_timestamps")
        if hist_ts:
            history_timestamps_map[uid] = hist_ts

    # ── Build features ──────────────────────────────────────────────
    print("\n📐 Building training features...")
    X_train, y_train, _ = build_features_for_dataset(
        train_beh, articles, pop_df, user_histories,
        history_timestamps_map or None, dataset, max_train_impressions)

    print("\n📐 Building validation features...")
    X_val, y_val, meta_val = build_features_for_dataset(
        val_beh, articles, pop_df, user_histories,
        history_timestamps_map or None, dataset, max_val_impressions)

    val_beh_eval = val_beh.head(max_val_impressions) if max_val_impressions is not None else val_beh

    # ── 1. Popularity baseline ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Ablation Study: {dataset.upper()}")
    print(f"{'='*60}")

    baseline_rankings = generate_candidate_scores(val_beh_eval, popularity, dataset)
    baseline_results = run_full_evaluation(
        baseline_rankings, dataset, popularity, user_history_lens,
        articles.height, total_interactions, method_name="popularity_baseline")

    # ── 2. Full model (all features) ────────────────────────────────
    print("\n🔧 Training FULL model (all features)...")
    model_full = train_reranker(X_train, y_train, X_val, y_val,
                                 feature_names=ALL_FEATURE_NAMES)
    full_rankings = rerank_impressions(model_full, X_val, meta_val, val_beh_eval)
    full_results = run_full_evaluation(
        full_rankings, dataset, popularity, user_history_lens,
        articles.height, total_interactions, method_name="reranker_full")

    # ── 3. Ablated model (without category features) ────────────────
    print("\n🔧 Training ABLATED model (no category features)...")
    X_train_abl = select_feature_columns(X_train, ALL_FEATURE_NAMES, BASE_FEATURE_NAMES)
    X_val_abl = select_feature_columns(X_val, ALL_FEATURE_NAMES, BASE_FEATURE_NAMES)
    model_ablated = train_reranker(X_train_abl, y_train, X_val_abl, y_val,
                                    feature_names=BASE_FEATURE_NAMES)
    ablated_rankings = rerank_impressions(model_ablated, X_val_abl, meta_val, val_beh_eval)
    ablated_results = run_full_evaluation(
        ablated_rankings, dataset, popularity, user_history_lens,
        articles.height, total_interactions, method_name="reranker_ablated")

    # ── 4. Statistical significance ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Statistical Significance Tests")
    print(f"{'='*60}")

    # Compute per-impression AUC for each model
    def _per_impression_metric(rankings, metric_fn):
        values = []
        for r in rankings:
            if r.get("labels") is None:
                continue
            values.append(metric_fn(r["labels"], r["scores"]))
        return values

    for metric_name, metric_fn in [
        ("AUC", compute_auc),
        ("MRR", compute_mrr),
        ("nDCG@5", lambda l, s: ndcg_at_k(l, s, 5)),
        ("nDCG@10", lambda l, s: ndcg_at_k(l, s, 10)),
    ]:
        full_vals = _per_impression_metric(full_rankings, metric_fn)
        ablated_vals = _per_impression_metric(ablated_rankings, metric_fn)
        baseline_vals = _per_impression_metric(baseline_rankings, metric_fn)

        # Full vs Ablated
        n = min(len(full_vals), len(ablated_vals))
        if n > 0:
            ci = paired_bootstrap_ci(ablated_vals[:n], full_vals[:n])
            sig = "✅ SIGNIFICANT" if ci["significant"] else "❌ NOT significant"
            print(f"\n  {metric_name} (Full - Ablated):")
            print(f"    Δ = {ci['mean_diff']:.4f}  "
                  f"[{ci['CI_lower']:.4f}, {ci['CI_upper']:.4f}]  {sig}")

        # Full vs Baseline
        n = min(len(full_vals), len(baseline_vals))
        if n > 0:
            ci = paired_bootstrap_ci(baseline_vals[:n], full_vals[:n])
            sig = "✅ SIGNIFICANT" if ci["significant"] else "❌ NOT significant"
            print(f"  {metric_name} (Full - Baseline):")
            print(f"    Δ = {ci['mean_diff']:.4f}  "
                  f"[{ci['CI_lower']:.4f}, {ci['CI_upper']:.4f}]  {sig}")

    # Save full model
    save_model(model_full, MODELS_DIR / f"reranker_{dataset}_full.txt")

    return {
        "baseline": baseline_results,
        "full": full_results,
        "ablated": ablated_results,
    }
