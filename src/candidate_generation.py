"""
Q2 (Stage 1) — Candidate Generation.

Uses popularity-based scoring to retrieve top-K candidates per impression.
In production, this would be backed by an ANN index; here we score
the impression's inview candidates directly (as provided by the dataset).
"""
from typing import Dict, List

import numpy as np
import polars as pl
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  POPULARITY-BASED CANDIDATE GENERATION
# ═══════════════════════════════════════════════════════════════════════

def rank_by_popularity(candidates: List, popularity: Dict) -> List[int]:
    """Rank candidates by popularity (descending). Returns 1-based ranks."""
    scores = [popularity.get(c, 0) for c in candidates]
    # argsort descending → rank
    order = np.argsort(scores)[::-1]
    ranks = [0] * len(candidates)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks


def score_by_popularity(candidates: List, popularity: Dict) -> List[float]:
    """Score candidates by popularity count."""
    return [float(popularity.get(c, 0)) for c in candidates]


def generate_candidate_scores(
    behaviors_df: pl.DataFrame,
    popularity: Dict,
    dataset: str = "mind",
) -> List[Dict]:
    """Score all candidates in a behaviors DataFrame using popularity.

    Returns list of dicts: {impression_id, user_id, candidates, scores, labels}.
    This is the Stage-1 output that feeds into the re-ranker.
    """
    results = []
    for row in tqdm(behaviors_df.iter_rows(named=True),
                     total=behaviors_df.height,
                     desc="Candidate generation"):
        candidates = row["candidates"]
        scores = score_by_popularity(candidates, popularity)
        results.append({
            "impression_id": row["impression_id"],
            "user_id": row["user_id"],
            "candidates": candidates,
            "scores": scores,
            "labels": row.get("labels"),
        })
    return results
