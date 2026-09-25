#!/usr/bin/env python3
"""
models.py
Gradient Boosted Decision Tree (LightGBM) pairwise matching model.
Uses GroupKFold grouped by Source 1 entity ID to prevent data leakage.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import joblib

FEATURE_COLS = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jw",
    "name_word_jaccard", "name_char3_jaccard", "name_first_token_match", "name_exact_match",
    "name_len_diff", "name_len_ratio",
    "addr_ratio", "addr_partial", "addr_token_sort", "addr_token_set", "addr_jw",
    "addr_word_jaccard", "digit_overlap", "has_common_digit",
    "full_token_set", "full_jw", "country_match", "is_source2", "is_source3"
]

class EntityMatchingModel:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 6):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_estimators": n_estimators,
            "verbose": -1
        }
        self.models: List[lgb.LGBMClassifier] = []
        self.feature_cols = FEATURE_COLS

    def train_cv(self, df_pairs: pd.DataFrame, n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train ensemble with GroupKFold over Source 1 entities.
        Returns out-of-fold probability predictions and true labels.
        """
        X = df_pairs[self.feature_cols].values
        y = df_pairs["label"].values
        groups = df_pairs["s1_id"].values

        n_groups = len(np.unique(groups))
        n_splits = max(min(n_splits, n_groups), 2)
        if len(df_pairs) < 40:
            self.params["min_child_samples"] = 1

        gkf = GroupKFold(n_splits=n_splits)
        oof_preds = np.zeros(len(df_pairs))
        self.models = []

        print(f"Training LightGBM with {n_splits}-fold GroupKFold on {len(df_pairs)} candidate pairs...")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Calculate class weight ratio for imbalance
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale_pos = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            
            clf = lgb.LGBMClassifier(**self.params, scale_pos_weight=min(scale_pos, 10.0))
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            val_preds = clf.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds
            self.models.append(clf)

        print("CV Training complete.")
        return oof_preds, y

    def predict_proba(self, df_pairs: pd.DataFrame) -> np.ndarray:
        """Ensemble average prediction across all trained fold models."""
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")
        X = df_pairs[self.feature_cols].values
        preds = np.zeros(len(df_pairs))
        for model in self.models:
            preds += model.predict_proba(X)[:, 1]
        return preds / len(self.models)

    def save(self, path: str):
        joblib.dump({"models": self.models, "features": self.feature_cols, "params": self.params}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.models = data["models"]
        self.feature_cols = data["features"]
        self.params = data["params"]
