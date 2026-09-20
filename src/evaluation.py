"""
Q5 — Extended Evaluation Harness.

Reports: AUC, MRR, nDCG@5, nDCG@10, Diversity (ILS), Novelty, Coverage.
Includes slicing (cold-start vs warm, head vs tail) and bootstrap 95% CIs.
"""
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from scipy.stats import rankdata

from src.config import (
    BOOTSTRAP_CI, BOOTSTRAP_ITERATIONS,
    COLD_START_THRESHOLD, HEAD_PERCENTILE,
    NDCG_CUTOFFS, OUTPUTS_DIR,
)


# ═══════════════════════════════════════════════════════════════════════
#  CORE METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_auc(labels: List[int], scores: List[float]) -> float:
    """Compute AUC for a single impression using rank statistics (handles ties)."""
    n_pos = sum(labels)
    n = len(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    pos_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return float((pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_mrr(labels: List[int], scores: List[float]) -> float:
    """Compute MRR for a single impression."""
    order = np.argsort(scores)[::-1]
    for rank, idx in enumerate(order, 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def dcg_at_k(labels: List[int], scores: List[float], k: int) -> float:
    """Compute DCG@k."""
    order = np.argsort(scores)[::-1][:k]
    dcg = 0.0
    for i, idx in enumerate(order):
        rel = labels[idx]
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(labels: List[int], scores: List[float], k: int) -> float:
    """Compute nDCG@k."""
    dcg = dcg_at_k(labels, scores, k)
    ideal_scores = sorted(labels, reverse=True)
    ideal_labels = [1] * len(labels)
    idcg = dcg_at_k(ideal_scores, ideal_scores, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ═══════════════════════════════════════════════════════════════════════
#  BEYOND-ACCURACY METRICS
# ═══════════════════════════════════════════════════════════════════════

def intra_list_similarity(candidates: List, scores: List[float],
                           embeddings_matrix: Optional[np.ndarray] = None,
                           id_to_idx: Optional[Dict] = None,
                           top_k: int = 10) -> float:
    """Compute Intra-List Similarity (diversity = 1 - ILS).

    Lower ILS = more diverse recommendations.
    """
    if embeddings_matrix is None or id_to_idx is None:
        return float("nan")

    order = np.argsort(scores)[::-1][:top_k]
    top_ids = [candidates[i] for i in order]
    vecs = []
    for cid in top_ids:
        idx = id_to_idx.get(cid)
        if idx is not None:
            vecs.append(embeddings_matrix[idx])

    if len(vecs) < 2:
        return 0.0

    vecs = np.array(vecs)
    sims = vecs @ vecs.T
    n = len(vecs)
    total = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(total)


def self_information(candidate_id, popularity: Dict,
                      total_interactions: int) -> float:
    """Compute self-information (novelty) for a single article."""
    count = popularity.get(candidate_id, 0)
    if count == 0 or total_interactions == 0:
        return 0.0
    prob = count / total_interactions
    return -math.log2(prob)


def novelty_score(candidates: List, scores: List[float],
                   popularity: Dict, total_interactions: int,
                   top_k: int = 10) -> float:
    """Average self-information of top-K recommended articles."""
    order = np.argsort(scores)[::-1][:top_k]
    infos = [self_information(candidates[i], popularity, total_interactions)
             for i in order]
    return float(np.mean(infos)) if infos else 0.0


def catalog_coverage(rankings: List[Dict], total_articles: int) -> float:
    """Fraction of all articles that appear in any top-10 recommendation."""
    recommended = set()
    for r in rankings:
        order = np.argsort(r["scores"])[::-1][:10]
        for idx in order:
            recommended.add(r["candidates"][idx])
    return len(recommended) / total_articles if total_articles > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_ci(values: List[float],
                  n_bootstrap: int = BOOTSTRAP_ITERATIONS,
                  ci: float = BOOTSTRAP_CI) -> Dict:
    """Compute bootstrap confidence interval for a list of per-impression values (vectorized)."""
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return {"mean": 0.0, "CI_lower": 0.0, "CI_upper": 0.0, "n": 0}

    rng = np.random.RandomState(42)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    # Vectorized bootstrap resample: (n_bootstrap, n)
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_means = np.sort(arr[indices].mean(axis=1))

    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1 - alpha) * n_bootstrap)]

    return {
        "mean": float(arr.mean()),
        "CI_lower": float(lo),
        "CI_upper": float(hi),
        "n": len(values),
    }


def paired_bootstrap_ci(values_a: List[float], values_b: List[float],
                          n_bootstrap: int = BOOTSTRAP_ITERATIONS,
                          ci: float = BOOTSTRAP_CI) -> Dict:
    """Paired bootstrap CI for the difference (B - A) (vectorized).

    If the CI excludes zero, the difference is statistically significant.
    """
    assert len(values_a) == len(values_b), "Must have same number of impressions"
    diffs = [b - a for a, b in zip(values_a, values_b)
             if not (math.isnan(a) or math.isnan(b))]

    if not diffs:
        return {"mean_diff": 0.0, "CI_lower": 0.0, "CI_upper": 0.0,
                "significant": False, "n": 0}

    rng = np.random.RandomState(42)
    arr = np.array(diffs, dtype=np.float64)
    n = len(arr)
    # Vectorized bootstrap resample: (n_bootstrap, n)
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_means = np.sort(arr[indices].mean(axis=1))

    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1 - alpha) * n_bootstrap)]
    significant = bool((lo > 0) or (hi < 0))  # CI excludes zero

    return {
        "mean_diff": float(arr.mean()),
        "CI_lower": float(lo),
        "CI_upper": float(hi),
        "significant": significant,
        "n": len(diffs),
    }


# ═══════════════════════════════════════════════════════════════════════
#  SLICING
# ═══════════════════════════════════════════════════════════════════════

def slice_by_user_warmth(rankings: List[Dict],
                          user_history_lens: Dict) -> Tuple[List, List]:
    """Split rankings into cold-start and warm user groups."""
    cold, warm = [], []
    for r in rankings:
        uid = r.get("user_id")
        hist_len = user_history_lens.get(uid, 0)
        if hist_len <= COLD_START_THRESHOLD:
            cold.append(r)
        else:
            warm.append(r)
    return cold, warm


def slice_by_article_popularity(rankings: List[Dict],
                                 popularity: Dict) -> Tuple[List, List]:
    """Split rankings based on whether majority of candidates are head or tail."""
    if not popularity:
        return [], rankings

    sorted_counts = sorted(popularity.values(), reverse=True)
    threshold_idx = int(len(sorted_counts) * (1 - HEAD_PERCENTILE))
    threshold = sorted_counts[threshold_idx] if threshold_idx < len(sorted_counts) else 0

    head_articles = {aid for aid, cnt in popularity.items() if cnt >= threshold}

    head_rankings, tail_rankings = [], []
    for r in rankings:
        head_count = sum(1 for c in r["candidates"] if c in head_articles)
        if head_count > len(r["candidates"]) / 2:
            head_rankings.append(r)
        else:
            tail_rankings.append(r)
    return head_rankings, tail_rankings


# ═══════════════════════════════════════════════════════════════════════
#  FULL EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_rankings(
    rankings: List[Dict],
    popularity: Optional[Dict] = None,
    total_articles: int = 0,
    total_interactions: int = 0,
    embeddings_matrix: Optional[np.ndarray] = None,
    id_to_idx: Optional[Dict] = None,
) -> Dict:
    """Compute all metrics for a list of impression rankings.

    Each ranking is a dict with: candidates, scores, labels.
    """
    aucs, mrrs = [], []
    ndcgs = {k: [] for k in NDCG_CUTOFFS}
    novelties, diversities = [], []

    for r in rankings:
        labels = r.get("labels")
        if labels is None:
            continue

        scores = r["scores"]
        candidates = r["candidates"]

        aucs.append(compute_auc(labels, scores))
        mrrs.append(compute_mrr(labels, scores))
        for k in NDCG_CUTOFFS:
            ndcgs[k].append(ndcg_at_k(labels, scores, k))

        if popularity:
            novelties.append(novelty_score(
                candidates, scores, popularity, total_interactions))

        if embeddings_matrix is not None:
            div = intra_list_similarity(
                candidates, scores, embeddings_matrix, id_to_idx)
            diversities.append(1.0 - div)  # diversity = 1 - ILS

    results = {}
    results["AUC"] = bootstrap_ci(aucs)
    results["MRR"] = bootstrap_ci(mrrs)
    for k in NDCG_CUTOFFS:
        results[f"nDCG@{k}"] = bootstrap_ci(ndcgs[k])

    if novelties:
        results["Novelty"] = bootstrap_ci(novelties)
    if diversities:
        results["Diversity"] = bootstrap_ci(diversities)
    if total_articles > 0:
        cov = catalog_coverage(rankings, total_articles)
        results["Coverage"] = {"value": cov}

    return results


def print_results(results: Dict, label: str = ""):
    """Pretty-print evaluation results with CIs."""
    if label:
        print(f"\n  📊 {label}")
    for metric, vals in results.items():
        if "mean" in vals:
            print(f"    {metric:12s}: {vals['mean']:.4f}  "
                  f"[{vals['CI_lower']:.4f}, {vals['CI_upper']:.4f}]  "
                  f"(n={vals['n']:,})")
        elif "value" in vals:
            print(f"    {metric:12s}: {vals['value']:.4f}")


def run_full_evaluation(
    rankings: List[Dict],
    dataset: str = "mind",
    popularity: Optional[Dict] = None,
    user_history_lens: Optional[Dict] = None,
    total_articles: int = 0,
    total_interactions: int = 0,
    embeddings_matrix: Optional[np.ndarray] = None,
    id_to_idx: Optional[Dict] = None,
    method_name: str = "",
) -> Dict:
    """Run full evaluation with slicing and persist results to JSON."""
    print(f"\n{'='*60}")
    print(f" Evaluation: {method_name.upper()} on {dataset.upper()}")
    print(f"{'='*60}")

    # Overall
    overall = evaluate_rankings(
        rankings, popularity=popularity,
        total_articles=total_articles,
        total_interactions=total_interactions,
        embeddings_matrix=embeddings_matrix,
        id_to_idx=id_to_idx,
    )
    print_results(overall, "Overall")

    all_results = {
        "dataset": dataset,
        "method": method_name,
        "overall": overall,
    }

    # Cold-start vs warm
    if user_history_lens:
        cold, warm = slice_by_user_warmth(rankings, user_history_lens)
        if cold:
            cold_res = evaluate_rankings(
                cold, popularity=popularity,
                total_articles=total_articles,
                total_interactions=total_interactions)
            print_results(cold_res, f"Cold-start (≤{COLD_START_THRESHOLD} clicks)")
            all_results["cold_start"] = cold_res
        if warm:
            warm_res = evaluate_rankings(
                warm, popularity=popularity,
                total_articles=total_articles,
                total_interactions=total_interactions)
            print_results(warm_res, f"Warm (>{COLD_START_THRESHOLD} clicks)")
            all_results["warm"] = warm_res

    # Head vs tail
    if popularity:
        head, tail = slice_by_article_popularity(rankings, popularity)
        if head:
            head_res = evaluate_rankings(
                head, popularity=popularity,
                total_articles=total_articles,
                total_interactions=total_interactions)
            print_results(head_res, "Head articles (popular)")
            all_results["head"] = head_res
        if tail:
            tail_res = evaluate_rankings(
                tail, popularity=popularity,
                total_articles=total_articles,
                total_interactions=total_interactions)
            print_results(tail_res, "Tail articles (long-tail)")
            all_results["tail"] = tail_res

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"eval_{dataset}_{method_name}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  💾 Results saved to {out_path}")

    # Update master summary
    summary_path = OUTPUTS_DIR / "all_eval_summary.json"
    master = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                master = json.load(f)
        except Exception:
            pass
    master[f"{dataset}_{method_name}"] = all_results
    with open(summary_path, "w") as f:
        json.dump(master, f, indent=2)

    return all_results
