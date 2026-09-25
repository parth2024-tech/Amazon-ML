#!/usr/bin/env python3
"""
threshold.py
Entity-level F_0.5 decision threshold calibration and singleton guard.
Directly optimizes macro-averaged F_0.5 over out-of-fold validation predictions.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from .evaluate import evaluate_predictions

def optimize_f05_threshold(
    df_pairs: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: dict[str, set[str]],
    threshold_grid: np.ndarray = np.linspace(0.20, 0.85, 35)
) -> tuple[float, float, dict]:
    """
    Searches the decision threshold grid to maximize macro F_0.5 at the entity level.
    Evaluates full singleton accuracy and false-merge penalties at each threshold.
    """
    df_eval = df_pairs[["s1_id", "target_id"]].copy()
    df_eval["prob"] = probabilities

    all_s1_ids = list(ground_truth.keys())
    best_thresh = 0.50
    best_f05 = -1.0
    best_metrics = {}

    print(f"Optimizing F_0.5 decision threshold over {len(threshold_grid)} points in [{threshold_grid[0]:.2f}, {threshold_grid[-1]:.2f}]...")

    for thresh in threshold_grid:
        passed = df_eval[df_eval["prob"] >= thresh]

        pred_map: dict[str, set[str]] = {sid: set() for sid in all_s1_ids}
        for s1_id, group in passed.groupby("s1_id"):
            if s1_id in pred_map:
                pred_map[s1_id] = set(group["target_id"].tolist())

        metrics = evaluate_predictions(pred_map, ground_truth)
        score = metrics["macro_f05"]

        if score > best_f05:
            best_f05 = score
            best_thresh = thresh
            best_metrics = metrics

    print(f"Optimal Threshold: {best_thresh:.3f} | Macro F_0.5: {best_f05:.4f} | Singleton Acc: {best_metrics.get('singleton_accuracy', 0):.4f}")
    return float(best_thresh), float(best_f05), best_metrics

def apply_threshold_to_candidates(
    df_pairs: pd.DataFrame,
    probabilities: np.ndarray,
    all_s1_ids: list[str],
    threshold: float,
    max_matches_per_entity: int = 10
) -> dict[str, list[str]]:
    """
    Applies calibrated threshold to generate final predictions:
    - Guaranteed exactly one entry per Source 1 entity.
    - Preserves multi-match from S2 and S3 if probabilities exceed threshold.
    - Yields empty list for singletons.
    """
    df_pred = df_pairs[["s1_id", "target_id"]].copy()
    df_pred["prob"] = probabilities

    # Filter above threshold and sort by confidence descending
    passed = df_pred[df_pred["prob"] >= threshold].sort_values("prob", ascending=False)

    predictions: dict[str, list[str]] = {sid: [] for sid in all_s1_ids}

    for s1_id, group in passed.groupby("s1_id"):
        if s1_id in predictions:
            # Drop duplicates while preserving rank order
            targets = list(dict.fromkeys(group["target_id"].tolist()))[:max_matches_per_entity]
            predictions[s1_id] = targets

    return predictions
