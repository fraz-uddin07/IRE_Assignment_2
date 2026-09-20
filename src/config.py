"""
Centralized configuration for IRE Assignment 2.
All paths are relative to PROJECT_ROOT.
"""
import sys
from pathlib import Path

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

# ── Project root (parent of src/) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ───────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# ── MIND paths ─────────────────────────────────────────────────────────
MIND_RAW = RAW_DIR / "mind"
MIND_TRAIN_DIR = MIND_RAW / "MINDsmall_train"
MIND_DEV_DIR = MIND_RAW / "MINDsmall_dev"
MIND_LARGE_TRAIN_DIR = MIND_RAW / "MINDlarge_train"
MIND_LARGE_DEV_DIR = MIND_RAW / "MINDlarge_dev"
MIND_LARGE_TEST_DIR = MIND_RAW / "MINDlarge_test"

MIND_PROCESSED = PROCESSED_DIR / "mind"

# ── EB-NeRD paths ─────────────────────────────────────────────────────
EBNERD_RAW = RAW_DIR / "ebnerd"
EBNERD_DEMO_DIR = EBNERD_RAW / "ebnerd_demo"
EBNERD_SMALL_DIR = EBNERD_RAW / "ebnerd_small"
EBNERD_LARGE_DIR = EBNERD_RAW / "ebnerd_large"
EBNERD_TEST_DIR = EBNERD_RAW / "ebnerd_testset"

EBNERD_PROCESSED = PROCESSED_DIR / "ebnerd"

# ── Download URLs ──────────────────────────────────────────────────────
MIND_URLS = {
    "MINDsmall_train": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip",
    "MINDsmall_dev": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip",
    "MINDlarge_train": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_train.zip",
    "MINDlarge_dev": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_dev.zip",
    "MINDlarge_test": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_test.zip",
}

EBNERD_URLS = {
    "ebnerd_demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "ebnerd_small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    "ebnerd_large": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    "ebnerd_testset": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
}

# ── Column definitions ─────────────────────────────────────────────────
MIND_BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
MIND_NEWS_COLS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]

# ── Retrieval constants ────────────────────────────────────────────────
TOP_K_CANDIDATES = 150          # Stage-1 candidate count
RECALL_K_VALUES = [50, 100, 200]
NDCG_CUTOFFS = [5, 10]

# ── Evaluation constants ──────────────────────────────────────────────
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI = 0.95
COLD_START_THRESHOLD = 5        # users with <= this many clicks = cold-start
HEAD_PERCENTILE = 0.80          # top 20% articles by popularity = head

# ── Feature engineering constants ─────────────────────────────────────
DECAY_HALF_LIFE_HOURS = 24.0    # exponential decay half-life for recency weighting

# ── LightGBM re-ranker hyperparameters ────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
LGBM_NUM_ROUNDS = 300
LGBM_EARLY_STOPPING = 30
