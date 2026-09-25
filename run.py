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

def cmd_blocking(train_dir: Path):
    print(">>> Benchmarking Multi-Index Blocking...")
    df_s1, df_s2, df_s3, gt = load_dataset_split(train_dir, "train")
    assert gt is not None, "Ground truth is required for blocking evaluation."
    blocker = MultiIndexBlocker()
    cands = blocker.generate_candidates(df_s1, df_s2, df_s3)
    cand_sets = {k: set(v) for k, v in cands.items()}
    n_target = len(df_s2) + len(df_s3)
    metrics = evaluate_blocking(cand_sets, gt, len(df_s1), n_target)
    print(f"Blocking Recall Ceiling: {metrics['blocking_recall']*100:.2f}% ({metrics['captured_links']}/{metrics['total_gt_links']})")
    print(f"Reduction Ratio: {metrics['reduction_ratio']*100:.4f}%")
    print(f"Average Candidates per S1: {metrics['avg_candidates_per_s1']:.2f}")

def cmd_full(train_dir: Path, test_dir: Path, model_type: str = "lightgbm", team_name: str = "team_submission"):
    print("=" * 65)
    print("🚀 AMAZON ML CHALLENGE 2026: END-TO-END ENTITY RESOLUTION")
    print("=" * 65)

    # 1. Load Data
    print("\n[Step 1/5] Loading Training Data...")
    df_s1, df_s2, df_s3, gt = load_dataset_split(train_dir, "train")
    assert gt is not None, "Ground truth is required for training."
    print(f"Loaded: {len(df_s1)} S1, {len(df_s2)} S2, {len(df_s3)} S3 entities.")

    # 2. Blocking on Train
    print("\n[Step 2/5] Multi-Index Candidate Generation...")
    blocker = MultiIndexBlocker()
    train_cands = blocker.generate_candidates(df_s1, df_s2, df_s3)
    cand_sets = {k: set(v) for k, v in train_cands.items()}
    block_metrics = evaluate_blocking(cand_sets, gt, len(df_s1), len(df_s2) + len(df_s3))
    print(f"Blocking Recall Ceiling: {block_metrics['blocking_recall']*100:.2f}% ({block_metrics['captured_links']}/{block_metrics['total_gt_links']} links captured)")
    print(f"Avg Candidates/Entity: {block_metrics['avg_candidates_per_s1']:.1f}")

    # 3. Feature Extraction on Train
    print("\n[Step 3/5] Extracting Pairwise Features...")
    df_target = pd.concat([df_s2, df_s3], ignore_index=True)
    df_pairs = generate_features_dataset(train_cands, df_s1, df_target)
    
    # Assign labels
    labels = []
    for _, row in df_pairs.iterrows():
        s1 = row["s1_id"]
        t = row["target_id"]
        labels.append(1 if t in gt.get(s1, set()) else 0)
    df_pairs["label"] = labels
    print(f"Constructed {len(df_pairs)} training pairs ({int(df_pairs['label'].sum())} positive links).")

    # 4. Model Training & F_0.5 Calibration
    print(f"\n[Step 4/5] Training {model_type.upper()} with GroupKFold...")
    clf = EntityClassifier(model_type=model_type)
    oof_probs, y = clf.train_cv(df_pairs)
    clf.save(str(MODELS_DIR / f"{model_type}_model.joblib"))

    # Threshold Optimization
    best_thresh, best_f05, eval_metrics = optimize_f05_threshold(df_pairs, oof_probs, gt)
    print(f"\n★ Out-of-Fold Macro F_0.5: {best_f05:.4f}")
    print(f"★ Singleton Accuracy: {eval_metrics.get('singleton_accuracy', 0):.4f}")
    print(f"★ Non-Singleton F_0.5: {eval_metrics.get('non_singleton_f05', 0):.4f}")
    print(f"★ Precision: {eval_metrics.get('overall_micro_precision', 0):.4f} | Recall: {eval_metrics.get('overall_micro_recall', 0):.4f}")

    # 5. Test Inference
    print("\n[Step 5/5] Generating Test Predictions & Validating...")
    df_test_s1, df_test_s2, df_test_s3, _ = load_dataset_split(test_dir, "test")
    test_cands, final_matches = run_test_inference(
        df_test_s1, df_test_s2, df_test_s3,
        classifier=clf, blocker=blocker, threshold=best_thresh
    )

    valid = save_and_validate_submission(test_cands, final_matches, test_dir)
    if valid:
        print("\n🎉 SUCCESS: All submission files validated perfectly against official rules!")
        package_submission_zip(team_name)
    else:
        print("\n⚠️ Validation warnings or errors detected.")

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Runner")
    parser.add_argument("--mode", choices=["eda", "blocking", "full", "package"], default="full")
    parser.add_argument("--train-dir", type=Path, default=DATA_DIR_TRAIN)
    parser.add_argument("--test-dir", type=Path, default=DATA_DIR_TEST)
    parser.add_argument("--model", choices=["lightgbm", "random_forest", "logistic_regression"], default="lightgbm")
    parser.add_argument("--team-name", default="amazon_ml_team")
    args = parser.parse_args()

    if args.mode == "eda":
        cmd_eda(args.train_dir)
    elif args.mode == "blocking":
        cmd_blocking(args.train_dir)
    elif args.mode == "package":
        package_submission_zip(args.team_name)
    else:
        cmd_full(args.train_dir, args.test_dir, model_type=args.model, team_name=args.team_name)

if __name__ == "__main__":
    main()
