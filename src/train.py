#!/usr/bin/env python3
"""
train.py
Supervised pairwise entity matching models:
- LightGBM Gradient Boosting (Primary)
- Logistic Regression (Baseline)
- Random Forest (Alternative)
Enforces GroupKFold grouped strictly by Source 1 entity ID to prevent pair leakage.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import joblib

from .features import FEATURE_COLUMNS
from .config import LIGHTGBM_PARAMS, CV_SPLITS

class EntityClassifier:
    def __init__(self, model_type: str = "lightgbm", params: dict | None = None):
        self.model_type = model_type
        self.params = params or LIGHTGBM_PARAMS
        self.models: list = []
        self.feature_cols = FEATURE_COLUMNS

    def train_cv(
        self,
        df_pairs: pd.DataFrame,
        n_splits: int = CV_SPLITS
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trains models using GroupKFold grouped by Source 1 ID.

        Returns:
            oof_probs   : full out-of-fold probability array (all rows)
            y           : true labels (all rows)
            thresh_mask : boolean mask marking the LAST fold (held out for
                          threshold calibration only — excluded from reported F0.5)

        Design: The last fold is reserved as an independent threshold-calibration
        holdout. The reported OOF F0.5 is computed on the remaining folds only.
        This avoids circular evaluation where the threshold is tuned on the same
        predictions used to measure it.
        """
        X = df_pairs[self.feature_cols].values
        y = df_pairs["label"].values
        groups = df_pairs["s1_id"].values

        n_groups = len(np.unique(groups))
        n_splits = max(min(n_splits, n_groups), 2)

        gkf = GroupKFold(n_splits=n_splits)
        oof_probs = np.zeros(len(df_pairs))
        thresh_mask = np.zeros(len(df_pairs), dtype=bool)  # last fold = threshold holdout
        self.models = []

        all_splits = list(gkf.split(X, y, groups))
        n_actual = len(all_splits)
        print(f"Training {self.model_type.upper()} with {n_actual}-fold GroupKFold over {n_groups} S1 entities...")
        print(f"  Fold {n_actual} reserved as threshold-calibration holdout (not included in reported F0.5)")

        for fold, (train_idx, val_idx) in enumerate(all_splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Mark last fold as threshold holdout
            if fold == n_actual - 1:
                thresh_mask[val_idx] = True

            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            scale_pos = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

            if self.model_type == "lightgbm":
                p = self.params.copy()
                if len(df_pairs) < 40:
                    p["min_child_samples"] = 1
                clf = lgb.LGBMClassifier(**p, scale_pos_weight=min(scale_pos, 10.0))
                clf.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                val_preds = np.asarray(clf.predict_proba(X_val))[:, 1]

            elif self.model_type == "xgboost":
                import xgboost as xgb
                clf = xgb.XGBClassifier(
                    tree_method="hist",
                    device="cuda",
                    n_estimators=1500,
                    max_depth=8,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    scale_pos_weight=min(scale_pos, 10.0),
                    random_state=42,
                    early_stopping_rounds=50
                )
                clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                val_preds = np.asarray(clf.predict_proba(X_val))[:, 1]

            elif self.model_type == "ensemble":
                import xgboost as xgb
                # Model 1: LightGBM on multi-core CPU
                p = self.params.copy()
                clf_lgb = lgb.LGBMClassifier(**p, scale_pos_weight=min(scale_pos, 10.0))
                clf_lgb.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                p_lgb = np.asarray(clf_lgb.predict_proba(X_val))[:, 1]

                # Model 2: XGBoost on NVIDIA GPU (CUDA)
                clf_xgb = xgb.XGBClassifier(
                    tree_method="hist",
                    device="cuda",
                    n_estimators=1500,
                    max_depth=8,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    scale_pos_weight=min(scale_pos, 10.0),
                    random_state=42,
                    early_stopping_rounds=50
                )
                clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                p_xgb = np.asarray(clf_xgb.predict_proba(X_val))[:, 1]

                val_preds = 0.5 * p_lgb + 0.5 * p_xgb
                clf = (clf_lgb, clf_xgb)

            elif self.model_type == "logistic_regression":
                clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
                clf.fit(X_train, y_train)
                val_preds = np.asarray(clf.predict_proba(X_val))[:, 1]

            elif self.model_type == "random_forest":
                clf = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train)
                val_preds = np.asarray(clf.predict_proba(X_val))[:, 1]

            oof_probs[val_idx] = val_preds
            self.models.append(clf)

        print(f"Completed {n_actual}-fold cross validation.")
        return oof_probs, y, thresh_mask

    def predict_proba(self, df_pairs: pd.DataFrame) -> np.ndarray:
        """Ensemble average across all trained fold models."""
        if not self.models:
            raise RuntimeError("Classifier has not been trained yet.")
        X = df_pairs[self.feature_cols].values
        probs = np.zeros(len(df_pairs))
        if self.model_type == "ensemble":
            for m_lgb, m_xgb in self.models:
                p_lgb = np.asarray(m_lgb.predict_proba(X))[:, 1]
                p_xgb = np.asarray(m_xgb.predict_proba(X))[:, 1]
                probs += 0.5 * p_lgb + 0.5 * p_xgb
            return probs / len(self.models)
        else:
            for m in self.models:
                probs += np.asarray(m.predict_proba(X))[:, 1]
            return probs / len(self.models)

    def save(self, path: str):
        joblib.dump({"models": self.models, "features": self.feature_cols, "type": self.model_type}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.models = data["models"]
        self.feature_cols = data["features"]
        self.model_type = data["type"]
