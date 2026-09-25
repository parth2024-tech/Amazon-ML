#!/usr/bin/env python3
"""
postprocessing.py
F_0.5 optimal threshold tuning, singleton guard, and submission file generation.
"""

from typing import Dict, List, Set, Tuple
import sys
import os
import numpy as np
import pandas as pd

# Handle path import for utils
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from utils.metrics import compute_macro_f05, compute_entity_f05
except ImportError:
    from ....utils.metrics import compute_macro_f05, compute_entity_f05

def find_optimal_threshold(
    df_pairs: pd.DataFrame,
    probs: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold_range: np.ndarray = np.linspace(0.2, 0.85, 30)
) -> Tuple[float, float, Dict]:
    """
    Grid search over decision thresholds to directly maximize macro-averaged F_0.5.
    Returns (best_threshold, best_macro_f05, detailed_metrics).
    """
    df_eval = df_pairs[["s1_id", "target_id"]].copy()
    df_eval["prob"] = probs

    best_thresh = 0.5
    best_score = -1.0
    best_metrics = {}

    all_s1_ids = list(ground_truth.keys())

    for thresh in threshold_range:
        # Filter candidate pairs above threshold
        passed = df_eval[df_eval["prob"] >= thresh]
        
        # Group by s1_id into sets of predicted target IDs
        pred_dict: Dict[str, Set[str]] = {sid: set() for sid in all_s1_ids}
        for s1_id, group in passed.groupby("s1_id"):
            if s1_id in pred_dict:
                pred_dict[s1_id] = set(group["target_id"].tolist())

        metrics = compute_macro_f05(pred_dict, ground_truth)
        score = metrics["macro_f05"]

        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_metrics = metrics

    print(f"Optimal threshold: {best_thresh:.3f} -> Macro F_0.5: {best_score:.4f}")
    return best_thresh, best_score, best_metrics

def generate_predictions(
    df_pairs: pd.DataFrame,
    probs: np.ndarray,
    all_s1_ids: List[str],
    threshold: float
) -> Dict[str, List[str]]:
    """
    Generates final predictions for every Source 1 entity:
    - Guaranteed exactly 1 row per S1 entity.
    - If no pairs pass threshold, predicts empty list (singleton).
    - Preserves candidates with highest probabilities first.
    """
    df_pred = df_pairs[["s1_id", "target_id"]].copy()
    df_pred["prob"] = probs

    # Filter above threshold
    passed = df_pred[df_pred["prob"] >= threshold].sort_values("prob", ascending=False)

    predictions: Dict[str, List[str]] = {sid: [] for sid in all_s1_ids}
    
    for s1_id, group in passed.groupby("s1_id"):
        if s1_id in predictions:
            # Drop any potential duplicates preserving order
            unique_targets = list(dict.fromkeys(group["target_id"].tolist()))
            predictions[s1_id] = unique_targets

    return predictions

def export_submission_tsv(
    predictions: Dict[str, List[str]],
    output_path: str,
    column_name: str = "matched_entity_ids"
):
    """
    Exports TSV file formatted strictly according to competition requirements.
    No quoting, tab delimiter, comma-separated IDs.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"source1_entity_id\t{column_name}\n")
        for s1_id in sorted(predictions.keys()):
            matches = predictions[s1_id]
            match_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{match_str}\n")
    print(f"Exported {len(predictions)} rows to {output_path}")
