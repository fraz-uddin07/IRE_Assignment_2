"""
Q4 — Serving & Scale Analysis.

Measures index memory, p99 retrieval latency, cost/QPS estimates,
and provides a scaling argument for 10x load.
"""
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from src.config import MIND_PROCESSED, EBNERD_PROCESSED, MODELS_DIR, OUTPUTS_DIR


# ═══════════════════════════════════════════════════════════════════════
#  MEMORY MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════

def _deep_getsizeof(obj, seen=None) -> int:
    """Recursively compute the memory footprint of an object."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(_deep_getsizeof(k, seen) + _deep_getsizeof(v, seen)
                    for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_getsizeof(i, seen) for i in obj)

    return size


def measure_index_memory(
    popularity: Dict,
    user_histories: Dict,
    articles_df: pl.DataFrame,
) -> Dict[str, float]:
    """Measure memory footprint of key data structures.

    Returns dict of component_name → size_MB.
    """
    results = {}

    pop_size = _deep_getsizeof(popularity)
    results["popularity_index"] = pop_size / 1e6

    hist_size = _deep_getsizeof(user_histories)
    results["user_histories"] = hist_size / 1e6

    # Estimate articles DataFrame size
    art_size = articles_df.estimated_size("mb")
    results["articles_dataframe"] = art_size

    results["total_MB"] = sum(results.values())

    print("\n📏 Index Memory Footprint:")
    for name, mb in results.items():
        print(f"  {name:30s}: {mb:8.2f} MB")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  LATENCY MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════

def measure_latency(
    model,
    X_sample: np.ndarray,
    meta_sample: List[Dict],
    popularity: Dict,
    n_trials: int = 1000,
) -> Dict[str, float]:
    """Measure p50, p95, p99 latency for single-user request.

    Each trial simulates: candidate generation → feature lookup → re-ranking.
    """
    import lightgbm as lgb

    # Group samples by impression for realistic simulation
    impressions = {}
    for i, m in enumerate(meta_sample):
        imp_id = m["impression_id"]
        if imp_id not in impressions:
            impressions[imp_id] = []
        impressions[imp_id].append(i)

    imp_keys = list(impressions.keys())
    if not imp_keys:
        return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0}

    latencies = []

    for trial in range(n_trials):
        # Pick a random impression
        imp_id = imp_keys[trial % len(imp_keys)]
        indices = impressions[imp_id]

        start = time.perf_counter()

        # Stage 1: Candidate generation (popularity lookup)
        candidates = [meta_sample[i]["candidate_id"] for i in indices]
        _ = [popularity.get(c, 0) for c in candidates]

        # Stage 2: Feature retrieval (already computed, simulate lookup)
        X_batch = X_sample[indices]

        # Stage 3: Re-ranking (model prediction)
        _ = model.predict(X_batch, num_iteration=model.best_iteration)

        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms

    latencies.sort()
    results = {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
        "max_ms": float(max(latencies)),
        "n_trials": n_trials,
    }

    print("\n⏱️  Latency Measurement:")
    for name, val in results.items():
        if name != "n_trials":
            print(f"  {name:12s}: {val:8.3f} ms")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  COST / QPS ESTIMATE
# ═══════════════════════════════════════════════════════════════════════

def estimate_cost_qps(latency_results: Dict,
                       target_sla_p99_ms: float = 100.0) -> Dict:
    """Back-of-envelope cost/QPS estimate.

    Args:
        latency_results: Output from measure_latency().
        target_sla_p99_ms: Target SLA for p99 latency (default 100ms).
    """
    p99 = latency_results["p99_ms"]
    mean = latency_results["mean_ms"]

    # QPS per core (based on mean latency)
    qps_per_core = 1000.0 / mean if mean > 0 else 0

    # Cores needed for target SLA
    # If p99 > target, we need more parallelism
    sla_ratio = p99 / target_sla_p99_ms if target_sla_p99_ms > 0 else 1.0

    results = {
        "qps_per_core": qps_per_core,
        "p99_ms": p99,
        "target_sla_p99_ms": target_sla_p99_ms,
        "meets_sla": p99 <= target_sla_p99_ms,
        "cost_per_1000_queries_usd": 0.0,  # estimated below
    }

    # Cost estimate: assume $0.05/hr per vCPU (cloud spot pricing)
    # Queries per hour per core: qps_per_core * 3600
    queries_per_hour = qps_per_core * 3600
    cost_per_hour = 0.05  # USD
    if queries_per_hour > 0:
        results["cost_per_1000_queries_usd"] = (1000 / queries_per_hour) * cost_per_hour

    print("\n💰 Cost / QPS Estimate:")
    print(f"  QPS per core:          {results['qps_per_core']:.1f}")
    print(f"  Meets SLA (p99<{target_sla_p99_ms}ms): {'✅ Yes' if results['meets_sla'] else '❌ No'}")
    print(f"  Cost per 1K queries:   ${results['cost_per_1000_queries_usd']:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  10x SCALING ARGUMENT
# ═══════════════════════════════════════════════════════════════════════

def scaling_analysis(memory_results: Dict, latency_results: Dict,
                      dataset: str = "mind") -> str:
    """Generate a scaling argument for 10x load."""
    total_mb = memory_results.get("total_MB", 0)

    analysis = f"""
╔══════════════════════════════════════════════════════════════╗
║  SCALING ANALYSIS: 10x Load ({dataset.upper()})
╚══════════════════════════════════════════════════════════════╝

Current State:
  • Memory footprint:     {total_mb:.1f} MB
  • p99 latency:          {latency_results.get('p99_ms', 0):.1f} ms
  • Mean latency:         {latency_results.get('mean_ms', 0):.1f} ms

At 10x Scale:
  • User histories:       ~{total_mb * 10:.0f} MB → still fits in RAM on a
                          single machine (< 64 GB). Sharding by user_id
                          across 2-4 nodes would be needed at 100x.

  • Popularity index:     Scales linearly with articles, not users.
                          10x users ≠ 10x articles. Negligible growth.

  • Re-ranker inference:  LightGBM prediction is O(candidates × trees).
                          At ~{latency_results.get('mean_ms', 0):.1f} ms/request, 10x QPS needs
                          10 cores → still a single machine.

  • What breaks first:    Feature store memory. User history storage
                          grows linearly with users. At 10x, the history
                          map exceeds ~{total_mb * 10:.0f} MB, requiring either:
                          (a) LRU eviction of inactive users, or
                          (b) External key-value store (Redis/Memcached).

  • Candidate generation: Popularity lookup is O(1) per candidate.
                          No bottleneck at 10x.

  • Mitigation strategy:  Horizontal sharding by user_id hash.
                          Each shard serves a subset of users with
                          local feature store + shared model replicas.
"""
    print(analysis)
    return analysis


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def run_serving_analysis(dataset: str = "mind",
                          model=None,
                          X_val: np.ndarray = None,
                          meta_val: List[Dict] = None) -> Dict:
    """Run the full serving & scale analysis."""
    import json
    id_col = "news_id" if dataset == "mind" else "article_id"
    proc = MIND_PROCESSED if dataset == "mind" else EBNERD_PROCESSED

    print(f"\n{'='*60}")
    print(f" Serving & Scale Analysis: {dataset.upper()}")
    print(f"{'='*60}")

    # Load data structures for memory measurement
    articles = pl.read_parquet(proc / "articles.parquet")
    pop_df = pl.read_parquet(proc / "popularity.parquet")
    popularity = dict(zip(pop_df[id_col].to_list(), pop_df["click_count"].to_list()))

    user_hist_df = pl.read_parquet(proc / "user_histories.parquet")
    user_histories = {}
    for row in user_hist_df.iter_rows(named=True):
        hist = row.get("history")
        if hist:
            user_histories[row["user_id"]] = hist

    # Memory
    memory = measure_index_memory(popularity, user_histories, articles)

    # Latency (requires model and validation data)
    latency = {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "mean_ms": 0, "max_ms": 0}
    cost = {}
    if model is not None and X_val is not None and meta_val is not None:
        latency = measure_latency(model, X_val, meta_val, popularity)
        cost = estimate_cost_qps(latency)

    # Scaling argument
    analysis_text = scaling_analysis(memory, latency, dataset)

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": dataset,
        "memory": memory,
        "latency": latency,
        "cost": cost,
        "scaling_analysis": analysis_text,
    }
    out_path = OUTPUTS_DIR / f"serving_analysis_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Serving analysis saved to {out_path}")

    return results
