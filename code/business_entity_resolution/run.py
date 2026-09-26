#!/usr/bin/env python3
"""
run.py
Main entrypoint for Amazon ML Challenge 2026: Business Entity Resolution.
Supports:
  --mode eda        : Run exploratory analysis on training data
  --mode blocking   : Benchmark blocking recall and reduction ratio
  --mode train      : Train GroupKFold models and optimize F_0.5 threshold
  --mode test       : Run test inference and generate submission files
  --mode full       : Run end-to-end pipeline (train -> tune -> test -> validate)
  --mode package    : Package official submission zip
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR_TRAIN,
    DATA_DIR_TEST,
    MODELS_DIR,
    OUTPUT_DIR,
    BLOCKING_CONFIG
)
from src.load_data import load_dataset_split
from src.eda import analyze_full_dataset
from src.blocking import MultiIndexBlocker
from src.features import generate_features_dataset
from src.train import EntityClassifier
from src.evaluate import evaluate_blocking, evaluate_predictions
from src.threshold import optimize_f05_threshold, apply_threshold_to_candidates
from src.predict import run_test_inference
from src.submission import save_and_validate_submission, package_submission_zip

def cmd_eda(train_dir: Path):
    print(">>> Running Exploratory Data Analysis...")
    df_s1, df_s2, df_s3, gt = load_dataset_split(train_dir, "train")
    analyze_full_dataset(df_s1, df_s2, df_s3, gt)

def cmd_blocking(train_dir: Path, sample_size: int | None = None):
    print(">>> Benchmarking Multi-Index Blocking...")
    df_s1, df_s2, df_s3, gt = load_dataset_split(train_dir, "train", sample_s1=sample_size)
    assert gt is not None, "Ground truth is required for blocking evaluation."
    blocker = MultiIndexBlocker()
    cands = blocker.generate_candidates(df_s1, df_s2, df_s3)
    cand_sets = {k: set(v) for k, v in cands.items()}
    n_target = len(df_s2) + len(df_s3)
    metrics = evaluate_blocking(cand_sets, gt, len(df_s1), n_target)
    print(f"\n★ Blocking Recall Ceiling: {metrics['blocking_recall']*100:.2f}% ({metrics['captured_links']}/{metrics['total_gt_links']})")
    print(f"★ Reduction Ratio: {metrics['reduction_ratio']*100:.4f}%")
    print(f"★ Average Candidates per S1: {metrics['avg_candidates_per_s1']:.2f}")

def cmd_full(
    train_dir: Path,
    test_dir: Path,
    model_type: str = "ensemble",
    team_name: str = "team_submission",
    sample_train_s1: int | None = None,
    sample_test_s1: int | None = None
):
    print("=" * 65)
    print("🚀 AMAZON ML CHALLENGE 2026: END-TO-END ENTITY RESOLUTION")
    print("=" * 65)

    # 1. Load Data
    print("\n[Step 1/5] Loading Training Data...")
    df_s1, df_s2, df_s3, gt = load_dataset_split(train_dir, "train", sample_s1=sample_train_s1)
    assert gt is not None, "Ground truth is required for training."
    print(f"Loaded: {len(df_s1):,} S1, {len(df_s2):,} S2, {len(df_s3):,} S3 entities.")

    # 2. Blocking on Train (75 candidates — memory-safe for 50k S1 entities)
    print("\n[Step 2/5] Multi-Index Candidate Generation...")
    import gc
    train_blocker = MultiIndexBlocker(max_total_candidates=75)
    train_cands = train_blocker.generate_candidates(df_s1, df_s2, df_s3)
    del train_blocker  # Free blocker memory before feature extraction
    gc.collect()
    cand_sets = {k: set(v) for k, v in train_cands.items()}
    block_metrics = evaluate_blocking(cand_sets, gt, len(df_s1), len(df_s2) + len(df_s3))
    print(f"Blocking Recall Ceiling: {block_metrics['blocking_recall']*100:.2f}% ({block_metrics['captured_links']}/{block_metrics['total_gt_links']} links captured)")
    print(f"Avg Candidates/Entity: {block_metrics['avg_candidates_per_s1']:.1f}")
    del cand_sets
    gc.collect()

    # 3. Feature Extraction on Train
    print("\n[Step 3/5] Extracting Pairwise Features...")
    df_pairs = generate_features_dataset(train_cands, df_s1, [df_s2, df_s3])
    del train_cands  # Free candidates dict after feature extraction
    gc.collect()

    # Assign labels
    df_pairs["label"] = [1 if t in gt.get(s1, set()) else 0 for s1, t in zip(df_pairs["s1_id"], df_pairs["target_id"])]
    print(f"Constructed {len(df_pairs):,} training pairs ({int(df_pairs['label'].sum()):,} positive links).")

    # 4. Model Training & F_0.5 Calibration
    print(f"\n[Step 4/5] Training {model_type.upper()} with GroupKFold...")
    clf = EntityClassifier(model_type=model_type)
    oof_probs, y, thresh_mask = clf.train_cv(df_pairs)
    clf.save(str(MODELS_DIR / f"{model_type}_model.joblib"))

    # --- Threshold calibration on INDEPENDENT holdout fold (last fold) ---
    # This avoids circular evaluation: threshold is tuned on data the model
    # never trained on AND never saw during F0.5 reporting.
    df_thresh = df_pairs[thresh_mask].copy()
    probs_thresh = oof_probs[thresh_mask]
    gt_thresh = {sid: gt[sid] for sid in df_thresh["s1_id"].unique() if sid in gt}
    best_thresh, _, _ = optimize_f05_threshold(df_thresh, probs_thresh, gt_thresh)
    print(f"  → Threshold {best_thresh:.3f} calibrated on {thresh_mask.sum():,} holdout pairs")

    # --- Honest OOF F_0.5: evaluated on folds 1..N-1 (never part of threshold search) ---
    eval_mask = ~thresh_mask
    df_eval = df_pairs[eval_mask].copy()
    probs_eval = oof_probs[eval_mask]
    gt_eval = {sid: gt[sid] for sid in df_eval["s1_id"].unique() if sid in gt}
    _, best_f05, eval_metrics = optimize_f05_threshold(
        df_eval, probs_eval, gt_eval,
        threshold_grid=np.array([best_thresh])
    )
    print(f"\n★ Honest OOF Macro F_0.5 (independent folds, threshold not tuned here): {best_f05:.4f}")
    print(f"★ Singleton Accuracy: {eval_metrics.get('singleton_accuracy', 0):.4f}")
    print(f"★ Non-Singleton F_0.5: {eval_metrics.get('non_singleton_f05', 0):.4f}")
    print(f"★ Precision: {eval_metrics.get('overall_micro_precision', 0):.4f} | Recall: {eval_metrics.get('overall_micro_recall', 0):.4f}")

    # 5. Test Inference — free all training data first to reclaim RAM
    print("\n[Step 5/5] Generating Test Predictions & Validating...")
    del df_s1, df_s2, df_s3, gt, df_pairs, oof_probs, y, thresh_mask
    del df_thresh, df_eval
    gc.collect()

    df_test_s1, df_test_s2, df_test_s3, _ = load_dataset_split(test_dir, "test")
    all_s1_ids = df_test_s1["entity_id"].tolist()

    # Fresh 120-candidate blocker for test — predict.py handles country-by-country (memory safe)
    test_blocker = MultiIndexBlocker(max_total_candidates=120)

    if sample_test_s1 is not None and sample_test_s1 < len(df_test_s1):
        print(f"Sampling first {sample_test_s1:,} test entities for fast evaluation...")
        df_test_s1_eval = df_test_s1.iloc[:sample_test_s1]
        test_cands, final_matches = run_test_inference(
            df_test_s1_eval, df_test_s2, df_test_s3,
            classifier=clf, blocker=test_blocker, threshold=best_thresh
        )
        for sid in all_s1_ids:
            if sid not in test_cands:
                test_cands[sid] = []
            if sid not in final_matches:
                final_matches[sid] = []
    else:
        test_cands, final_matches = run_test_inference(
            df_test_s1, df_test_s2, df_test_s3,
            classifier=clf, blocker=test_blocker, threshold=best_thresh
        )

    valid = save_and_validate_submission(test_cands, final_matches, test_dir)
    tsv_path = OUTPUT_DIR / "matching_results.tsv"
    if valid:
        print(f"\n🎉 SUCCESS! matching_results.tsv written to:\n   {tsv_path.resolve()}")
        print(f"   Run '--mode package' when you want the submission zip.")
    else:
        print(f"\n⚠️ Validation warnings detected. TSV still at:\n   {tsv_path.resolve()}")

def cmd_test(
    test_dir: Path,
    model_type: str = "ensemble",
    team_name: str = "team_submission",
    threshold: float = 0.919,
    sample_test_s1: int | None = None
):
    print("=" * 65)
    print(f"🚀 INFERENCE WITH TRAINED {model_type.upper()} MODEL (Threshold: {threshold:.3f})")
    print("=" * 65)

    model_path = MODELS_DIR / f"{model_type}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first with --mode full.")

    print(f"Loading trained model from {model_path.resolve()}...")
    clf = EntityClassifier(model_type=model_type)
    clf.load(str(model_path))
    print(f"Loaded {len(clf.models)} trained fold models with {model_type}.")

    print("\nLoading Test Data...")
    df_test_s1, df_test_s2, df_test_s3, _ = load_dataset_split(test_dir, "test")
    all_s1_ids = df_test_s1["entity_id"].tolist()

    test_blocker = MultiIndexBlocker(max_total_candidates=120)

    if sample_test_s1 is not None and sample_test_s1 < len(df_test_s1):
        print(f"Sampling first {sample_test_s1:,} test entities for fast evaluation...")
        df_test_s1_eval = df_test_s1.iloc[:sample_test_s1]
        test_cands, final_matches = run_test_inference(
            df_test_s1_eval, df_test_s2, df_test_s3,
            classifier=clf, blocker=test_blocker, threshold=threshold
        )
        for sid in all_s1_ids:
            if sid not in test_cands:
                test_cands[sid] = []
            if sid not in final_matches:
                final_matches[sid] = []
    else:
        test_cands, final_matches = run_test_inference(
            df_test_s1, df_test_s2, df_test_s3,
            classifier=clf, blocker=test_blocker, threshold=threshold
        )

    valid = save_and_validate_submission(test_cands, final_matches, test_dir)
    tsv_path = OUTPUT_DIR / "matching_results.tsv"
    if valid:
        print(f"\n🎉 SUCCESS! matching_results.tsv written to:\n   {tsv_path.resolve()}")
        print(f"   Run '--mode package' when you want the submission zip.")
    else:
        print(f"\n⚠️ Validation warnings detected. TSV still at:\n   {tsv_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Runner")
    parser.add_argument("--mode", choices=["eda", "blocking", "test", "full", "package"], default="full")
    parser.add_argument("--train-dir", type=Path, default=DATA_DIR_TRAIN)
    parser.add_argument("--test-dir", type=Path, default=DATA_DIR_TEST)
    parser.add_argument("--model", choices=["ensemble", "lightgbm", "xgboost", "random_forest", "logistic_regression"], default="ensemble")
    parser.add_argument("--team-name", default="amazon_ml_team")
    parser.add_argument("--threshold", type=float, default=0.919, help="Decision threshold for test inference")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for fast benchmark/training")
    parser.add_argument("--test-sample", type=int, default=None, help="Optional test sample size for fast evaluation")
    args = parser.parse_args()

    if args.mode == "eda":
        cmd_eda(args.train_dir)
    elif args.mode == "blocking":
        cmd_blocking(args.train_dir, sample_size=args.sample)
    elif args.mode == "test":
        cmd_test(
            args.test_dir,
            model_type=args.model,
            team_name=args.team_name,
            threshold=args.threshold,
            sample_test_s1=args.test_sample
        )
    elif args.mode == "package":
        package_submission_zip(args.team_name)
    else:
        cmd_full(
            args.train_dir,
            args.test_dir,
            model_type=args.model,
            team_name=args.team_name,
            sample_train_s1=args.sample,
            sample_test_s1=args.test_sample
        )


if __name__ == "__main__":
    main()
