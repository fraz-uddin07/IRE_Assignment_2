# IRE Assignment 2 — News Recommendation Pipeline

Two-stage retrieve-then-rank news recommendation system evaluated on **EB-NeRD** and **MIND** datasets. Features a popularity-based candidate generator, LightGBM re-ranker with 11 behavioural features, ablation study with paired bootstrap significance tests, and serving/scale analysis.

## Quick Start (One-Command Reproduce)

```bash
# Full pipeline: download → features → train → evaluate → predict
python run_pipeline.py --dataset ebnerd --size demo
python run_pipeline.py --dataset mind --size small
```

## Project Structure

```
a2/
├── src/
│   ├── config.py                # Centralized paths, URLs, hyperparameters
│   ├── data_loader.py           # Download, extract, parse datasets
│   ├── feature_engineering.py   # 11 behavioural features
│   ├── candidate_generation.py  # Popularity-based top-K retrieval
│   ├── reranker.py              # LightGBM train/predict
│   ├── baseline.py              # Ablation study orchestration
│   ├── evaluation.py            # Metrics (AUC, MRR, nDCG, novelty, coverage)
│   ├── serving_analysis.py      # Memory, latency, cost benchmarks
│   └── predict.py               # Codabench prediction generation
├── run_pipeline.py              # End-to-end pipeline orchestrator
├── requirements.txt             # Python dependencies
├── report.tex                   # LaTeX report
├── tests/                       # Unit tests
├── data/                        # Raw + processed data (gitignored)
├── models/                      # Trained LightGBM models (gitignored)
└── outputs/                     # Evaluation results + predictions
```

## Requirements

```bash
pip install -r requirements.txt
```

**Dependencies**: Python 3.10+, polars, pyarrow, lightgbm, numpy, scikit-learn, tqdm, requests

## Step-by-Step Pipeline

### 1. Download & Prepare Data

```bash
# EB-NeRD (demo for dev, large for submission)
python -m src.data_loader --dataset ebnerd --size demo

# MIND (small for dev, large for submission)
python -m src.data_loader --dataset mind --size small
```

### 2. Run Ablation Study (Train + Evaluate)

```bash
python run_pipeline.py --dataset ebnerd --size demo
python run_pipeline.py --dataset mind --size small
```

This runs:
- Feature engineering on train/validation splits
- Full model training (11 features) and ablated model (10 features, no `category_match`)
- Extended evaluation with bootstrap CIs, cold-start/warm and head/tail slicing
- Serving & scale analysis (memory, latency, cost/QPS)

### 3. Generate Codabench Predictions

```bash
# EB-NeRD predictions
python -m src.predict --dataset ebnerd --method reranker

# MIND predictions
python -m src.predict --dataset mind --method reranker
```

Output: `outputs/ebnerd_prediction_reranker.zip` and `outputs/mind_prediction_reranker.zip` — upload these to Codabench.

### 4. Compile Report

```bash
pdflatex report.tex
```

## Google Colab

Upload `IRE_Assignment2_Colab.ipynb` to Colab and follow the cells. The notebook mounts Google Drive and runs the full pipeline.

```python
# In Colab:
!pip install polars pyarrow lightgbm tqdm
!python run_pipeline.py --dataset ebnerd --size demo --skip-download
!python -m src.predict --dataset ebnerd --method reranker
```

## Key Design Decisions

| Choice | Rationale |
|--------|-----------|
| **LightGBM** (Option A) | Fast training (<1 min), interpretable feature importance, no GPU needed |
| **Popularity retriever** | Simple, reproducible baseline that isolates re-ranker contribution |
| **Category-match feature** | Strong topical signal, O(1) at serving time, works on both datasets |
| **Polars + PyArrow** | Memory-efficient processing for 13.5M impressions (EB-NeRD test) |

## Outputs

After running the pipeline, results are saved to `outputs/`:

- `ire_assignment2_results.json` — Full results (training, evaluation, serving, anti-gaming)
- `all_eval_summary.json` — Evaluation metrics summary
- `eval_ebnerd_*.json` / `eval_mind_*.json` — Per-dataset evaluation
- `*_prediction_reranker.zip` — Codabench submission files

## Anti-Gaming

- Strict temporal boundary: no future clicks leak into features
- All features available at serving time (no oracle features)
- Metrics reported with and without ablated feature
- Test sets have no click labels
