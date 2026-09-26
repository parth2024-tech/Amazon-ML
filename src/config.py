#!/usr/bin/env python3
"""
config.py
Centralized configuration, paths, and hyperparameters for Business Entity Resolution.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths (checks both data/ and dataset/ directories)
def get_data_dir(split: str = "train") -> Path:
    candidates = [
        PROJECT_ROOT / "data" / split,
        PROJECT_ROOT / "dataset" / split,
    ]
    for p in candidates:
        if p.exists() and any(p.glob("*.tsv")):
            return p
    # Default to data/ if neither has files yet
    return PROJECT_ROOT / "data" / split

DATA_DIR_TRAIN = get_data_dir("train")
DATA_DIR_TEST = get_data_dir("test")

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LEGACY_OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Metric Configuration
BETA = 0.5  # Precision-heavy macro F_beta

# Blocking Configuration
BLOCKING_CONFIG = {
    "tfidf_ngram_range": (3, 4),
    "tfidf_top_k": 20,
    "min_token_len": 4,
    "max_tokens_per_name": 4,
    "inverted_index_top_k": 10,
    "max_total_candidates_per_entity": 25
}

# Model Hyperparameters
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "max_depth": 8,
    "num_leaves": 127,
    "min_child_samples": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 1500,
    "n_jobs": -1,
    "verbose": -1
}

# GroupKFold CV Configuration
CV_SPLITS = 5
RANDOM_SEED = 42
