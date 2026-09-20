"""
Q2 — LightGBM Re-Ranker.

Trains a GBDT binary classifier on behavioural features,
then re-ranks Stage-1 candidates by predicted click probability.
Reports AUC, MRR, nDCG@5, nDCG@10 before and after re-ranking.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl

from src.config import (
    LGBM_PARAMS, LGBM_NUM_ROUNDS, LGBM_EARLY_STOPPING,
    MODELS_DIR, OUTPUTS_DIR,
)
from src.feature_engineering import (
    ALL_FEATURE_NAMES, BASE_FEATURE_NAMES, CATEGORY_FEATURE_NAMES,
    build_features_for_dataset,
)


# ═══════════════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════════════

def train_reranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str] = ALL_FEATURE_NAMES,
    params: Optional[Dict] = None,
) -> lgb.Booster:
    """Train a LightGBM binary classifier for re-ranking.

    Args:
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        feature_names: Feature column names.
        params: LightGBM parameters (defaults to config).

    Returns:
        Trained LightGBM Booster.
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    train_data = lgb.Dataset(X_train, label=y_train,
                              feature_name=feature_names, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val,
                            feature_name=feature_names, free_raw_data=False)

    print(f"\n🔧 Training LightGBM re-ranker...")
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print(f"  Positive rate: train={y_train.mean():.3f}, val={y_val.mean():.3f}")

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=LGBM_NUM_ROUNDS,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best val AUC: {model.best_score['val']['auc']:.4f}")

    return model


def save_model(model: lgb.Booster, path: Path):
    """Save LightGBM model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    print(f"  💾 Model saved to {path}")


def load_model(path: Path) -> lgb.Booster:
    """Load LightGBM model from disk."""
    model = lgb.Booster(model_file=str(path))
    print(f"  📂 Model loaded from {path}")
    return model


# ═══════════════════════════════════════════════════════════════════════
#  PREDICT / RE-RANK
# ═══════════════════════════════════════════════════════════════════════

def predict_scores(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Predict click probability for each sample."""
    return model.predict(X, num_iteration=model.best_iteration)


def rerank_impressions(
    model: lgb.Booster,
    X: np.ndarray,
    meta: List[Dict],
    behaviors_df: pl.DataFrame,
) -> List[Dict]:
    """Re-rank candidates per impression using model predictions.

    Returns list of dicts: {impression_id, user_id, candidates, scores, labels}.
    """
    # Predict scores for all samples
    pred_scores = predict_scores(model, X)

    # Group by impression_id
    impression_data: Dict[str, Dict] = {}
    for i, m in enumerate(meta):
        imp_id = m["impression_id"]
        if imp_id not in impression_data:
            impression_data[imp_id] = {
                "impression_id": imp_id,
                "user_id": m["user_id"],
                "candidates": [],
                "scores": [],
            }
        impression_data[imp_id]["candidates"].append(m["candidate_id"])
        impression_data[imp_id]["scores"].append(float(pred_scores[i]))

    # Add labels from behaviors_df
    labels_map = {}
    for row in behaviors_df.iter_rows(named=True):
        labels_map[row["impression_id"]] = {
            "labels": row.get("labels"),
            "candidates": row["candidates"],
        }

    results = []
    for imp_id, data in impression_data.items():
        orig = labels_map.get(imp_id, {})
        orig_candidates = orig.get("candidates", [])
        orig_labels = orig.get("labels")

        # Match labels to our candidate ordering
        if orig_labels is not None:
            label_map = dict(zip(orig_candidates, orig_labels))
            data["labels"] = [label_map.get(c, 0) for c in data["candidates"]]
        else:
            data["labels"] = None

        results.append(data)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════

def print_feature_importance(model: lgb.Booster,
                              feature_names: List[str] = ALL_FEATURE_NAMES):
    """Print feature importance from the trained model."""
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    print("\n📊 Feature Importance (gain):")
    for name, imp in pairs:
        bar = "█" * int(imp / max(importance) * 30) if max(importance) > 0 else ""
        print(f"  {name:30s} {imp:10.1f}  {bar}")


# ═══════════════════════════════════════════════════════════════════════
#  SUBSET FEATURES (for ablation)
# ═══════════════════════════════════════════════════════════════════════

def select_feature_columns(X: np.ndarray,
                            all_names: List[str],
                            selected_names: List[str]) -> np.ndarray:
    """Select a subset of feature columns by name."""
    indices = [all_names.index(name) for name in selected_names]
    return X[:, indices]
