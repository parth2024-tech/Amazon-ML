#!/usr/bin/env python3
"""
evaluate.py
Official macro-averaged F_0.5 evaluation metric and diagnostic evaluation suite.
"""

from typing import Dict, List, Set

def compute_entity_f05(pred_set: set[str], gt_set: set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    - If true matches is empty (singleton):
        * pred empty -> 1.0
        * pred non-empty -> 0.0
    - If true matches is non-empty:
        * pred empty -> 0.0
        * otherwise F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
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

    return (1.25 * precision * recall) / denom

def evaluate_predictions(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]]
) -> dict[str, float]:
    """
    Evaluates predictions against ground truth for all Source 1 entities.
    Returns comprehensive evaluation statistics.
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
    false_positives = 0

    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_set, gt_set)
        scores.append(score)

        if len(gt_set) == 0:
            singleton_scores.append(score)
            if len(pred_set) > 0:
                false_positives += len(pred_set)
        else:
            non_singleton_scores.append(score)
            tp = len(pred_set & gt_set)
            total_tp += tp
            total_gt += len(gt_set)
            false_positives += len(pred_set - gt_set)

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
        "overall_micro_recall": micro_rec,
        "total_predictions": total_pred,
        "total_true_links": total_gt,
        "total_false_positives": false_positives
    }

def evaluate_blocking(
    candidates: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    n_s1: int,
    n_target: int
) -> dict[str, float]:
    """
    Evaluates blocking quality:
    - Recall ceiling: fraction of true ground-truth links present in candidate sets
    - Candidate reduction ratio: fraction of all Cartesian pairs eliminated
    - Average candidates per S1 entity
    """
    total_gt_links = sum(len(matches) for matches in ground_truth.values())
    captured_links = 0

    total_candidates = sum(len(cands) for cands in candidates.values())

    for s1_id, gt_matches in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        captured_links += len(gt_matches & cand_set)

    blocking_recall = (captured_links / total_gt_links) if total_gt_links > 0 else 1.0
    
    total_cartesian = n_s1 * n_target
    reduction_ratio = 1.0 - (total_candidates / total_cartesian) if total_cartesian > 0 else 1.0
    avg_cands_per_s1 = (total_candidates / n_s1) if n_s1 > 0 else 0.0

    return {
        "blocking_recall": blocking_recall,
        "captured_links": captured_links,
        "total_gt_links": total_gt_links,
        "total_candidates": total_candidates,
        "reduction_ratio": reduction_ratio,
        "avg_candidates_per_s1": avg_cands_per_s1
    }
