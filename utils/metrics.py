#!/usr/bin/env python3
"""
metrics.py
Macro-averaged F_0.5 evaluation metric for Amazon ML Challenge 2026.
"""

from typing import Dict, Set, List

def compute_entity_f05(pred_set: set[str], gt_set: set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    Singletons:
      - If true is empty and pred is empty -> 1.0
      - If true is empty and pred is non-empty -> 0.0
    Non-singletons:
      - If true is non-empty and pred is empty -> 0.0
      - Otherwise standard F_0.5 with beta=0.5
    """
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    if len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & gt_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05

def compute_macro_f05(predictions: dict[str, set[str]], ground_truth: dict[str, set[str]]) -> dict[str, float]:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities in ground truth.
    Returns:
      {
        "macro_f05": float,
        "singleton_accuracy": float,
        "non_singleton_f05": float,
        "precision": float,
        "recall": float
      }
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0}

    scores = []
    singleton_scores = []
    non_singleton_scores = []
    total_tp = 0
    total_pred = 0
    total_gt = 0

    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_set, gt_set)
        scores.append(score)

        if len(gt_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            total_tp += len(pred_set & gt_set)
            total_gt += len(gt_set)
        total_pred += len(pred_set)

    micro_prec = (total_tp / total_pred) if total_pred > 0 else 0.0
    micro_rec = (total_tp / total_gt) if total_gt > 0 else 0.0

    return {
        "macro_f05": sum(scores) / len(scores),
        "singleton_count": len(singleton_scores),
        "singleton_accuracy": (sum(singleton_scores) / len(singleton_scores)) if singleton_scores else 1.0,
        "non_singleton_count": len(non_singleton_scores),
        "non_singleton_f05": (sum(non_singleton_scores) / len(non_singleton_scores)) if non_singleton_scores else 0.0,
        "overall_micro_precision": micro_prec,
        "overall_micro_recall": micro_rec
    }
