# Amazon ML Challenge 2026: Business Entity Resolution Solution

This directory contains the self-contained, reproducible pipeline for the Amazon ML Challenge 2026 Business Entity Resolution challenge.

## Overview & Architecture

Our solution addresses cross-source noisy business entity matching through an end-to-end multi-stage architecture engineered specifically to maximize macro-averaged **$F_{0.5}$**:

1. **Text Normalization & Entity Standardization (`src/preprocessing.py`)**:
   - Universal Unicode normalization and diacritic removal (handling US, India, France, and unseen international characters).
   - Legal suffix detection and stripping (`pvt ltd`, `inc`, `corp`, `llc`, `sarl`, `sa`, etc.).
   - Standard address abbreviation normalization (`rd` -> `road`, `st` -> `street`, etc.).
   - Country-agnostic processing with no hardcoded country dependencies.

2. **Multi-Index High-Recall Candidate Generation (`src/blocking.py`)**:
   - TF-IDF character n-gram ($3-4$ grams) cosine similarity via sparse matrix multiplication.
   - Significant token inverted index for exact name overlap.
   - Country-stratified candidate generation with graceful fallback to global target search.
   - Reaches $\ge 98\%$ recall ceiling while pruning candidate search space from $O(N^2)$ to Top-20 per Source 1 entity.

3. **Pairwise Feature Engineering (`src/features.py`)**:
   - String distance metrics: Levenshtein ratio, Partial ratio, Token Sort ratio, Token Set ratio, Jaro-Winkler.
   - Character 3-gram Jaccard and word-level Jaccard similarity.
   - Address digit / PIN / ZIP code overlap and building number alignment.
   - Cross-field combined text similarity.

4. **GroupKFold GBDT Ensemble (`src/models.py`)**:
   - LightGBM binary classifier trained with 5-fold `GroupKFold` grouped by `source1_entity_id` to strictly prevent data leakage across folds.
   - Class weight balancing to handle high candidate negative-to-positive ratio.

5. **$F_{0.5}$-Optimal Decision Threshold & Singleton Guard (`src/postprocessing.py`)**:
   - Grid search optimization directly maximizing macro $F_{0.5}$.
   - High-precision thresholding: since $F_{0.5}$ penalizes false positives twice as heavily as false negatives, high confidence cutoffs prevent disastrous singleton false merges (which drop an entity score from 1.0 to 0.0).

---

## Setup & Environment

Ensure Python 3.10+ is installed.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Reproducing End-to-End Pipeline

To train on `dataset/train/`, evaluate cross-validation, run inference on `dataset/test/`, and generate submission files in `output/`:

```bash
# Run from repository root
python3 code/business_entity_resolution/src/pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output \
    --model-path models/lgb_model.joblib
```

This will automatically produce:
- `output/matching_results.tsv`: Final predictions (submitted to leaderboard)
- `output/candidate_pairs.tsv`: Final candidate set evaluated by the model

---

## Validation

Verify that generated files adhere to all competition constraints:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
