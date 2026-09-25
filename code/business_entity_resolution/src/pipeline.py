#!/usr/bin/env python3
"""
pipeline.py
End-to-End Execution Pipeline for Amazon ML Challenge 2026.
Runs:
1. Data loading and text cleaning
2. High-recall candidate blocking
3. Pairwise feature extraction
4. LightGBM ensemble training & GroupKFold validation
5. F_0.5 threshold calibration
6. Test set inference
7. Exporting candidate_pairs.tsv and matching_results.tsv
8. Automatic submission format validation
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np

# Adjust python path
current_dir = os.path.dirname(os.path.abspath(__file__))
ber_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
for p in [ber_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.preprocessing import (
    clean_business_name, clean_address, clean_country, extract_digits
)
from src.blocking import CandidateBlocker
from src.features import build_feature_dataframe
from src.models import EntityMatchingModel
from src.postprocessing import (
    find_optimal_threshold, generate_predictions, export_submission_tsv
)

def load_source_tsv(path: str) -> pd.DataFrame:
    """Loads a source TSV file with explicit tab separator."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    # Ensure expected columns
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""
    # Add preprocessed columns
    df["cleaned_name"] = df["business_name"].apply(clean_business_name)
    df["cleaned_address"] = df["business_address"].apply(clean_address)
    df["clean_country"] = df["country"].apply(clean_country)
    df["digits"] = df["business_address"].apply(extract_digits)
    return df

def load_ground_truth(path: str) -> dict:
    """Loads train_ground_truth.tsv into {s1_id: set(matched_ids)}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth not found: {path}")
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == 0 and "source1_entity_id" in line:
                continue
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0].strip()
            matches = parts[1].strip() if len(parts) > 1 else ""
            gt[s1_id] = set(m.strip() for m in matches.split(",") if m.strip())
    return gt

def run_pipeline(
    train_dir: str = "dataset/train",
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_save_path: str = "models/lgb_model.joblib"
):
    print("=" * 60)
    print("🚀 AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION PIPELINE")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 1. Load Training Data
    print("\n[Step 1/6] Loading Training Datasets...")
    train_s1 = load_source_tsv(os.path.join(train_dir, "train_source1.tsv"))
    train_s2 = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"))
    train_s3 = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"))
    train_gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    print(f"Loaded {len(train_s1)} S1 entities, {len(train_s2)} S2, {len(train_s3)} S3.")

    # 2. Candidate Blocking on Train
    print("\n[Step 2/6] Running Candidate Blocking on Training Data...")
    blocker = CandidateBlocker(top_k_per_source=20)
    train_candidates = blocker.generate_candidates(train_s1, train_s2, train_s3)
    
    # Calculate blocking recall ceiling
    total_matches = sum(len(m) for m in train_gt.values())
    found_matches = sum(len(train_gt[sid] & set(train_candidates.get(sid, []))) for sid in train_gt)
    recall_ceiling = (found_matches / total_matches) if total_matches > 0 else 1.0
    print(f"Blocking Recall Ceiling on Train: {recall_ceiling * 100:.2f}% ({found_matches}/{total_matches} true links)")

    # 3. Pairwise Feature Extraction on Train
    print("\n[Step 3/6] Extracting Pairwise Features for Training Pairs...")
    train_target = pd.concat([train_s2, train_s3], ignore_index=True)
    df_train_pairs = build_feature_dataframe(train_candidates, train_s1, train_target)
    
    # Assign labels
    labels = []
    for _, row in df_train_pairs.iterrows():
        s1 = row["s1_id"]
        t = row["target_id"]
        labels.append(1 if t in train_gt.get(s1, set()) else 0)
    df_train_pairs["label"] = labels
    print(f"Created {len(df_train_pairs)} pairs ({df_train_pairs['label'].sum()} positives).")

    # 4. Train Model & Optimize F_0.5 Threshold
    print("\n[Step 4/6] Training LightGBM Model with GroupKFold...")
    model = EntityMatchingModel(n_estimators=350, learning_rate=0.04)
    oof_probs, y_true = model.train_cv(df_train_pairs, n_splits=5)
    model.save(model_save_path)

    print("\nTuning Decision Threshold for Macro F_0.5...")
    best_thresh, best_f05, metrics = find_optimal_threshold(df_train_pairs, oof_probs, train_gt)
    print(f"CV Macro F_0.5 Score: {best_f05:.4f}")
    print(f"Singleton Accuracy: {metrics.get('singleton_accuracy', 0.0):.4f}")

    # 5. Process Test Set
    print("\n[Step 5/6] Processing Test Set...")
    test_s1 = load_source_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s2 = load_source_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s3 = load_source_tsv(os.path.join(test_dir, "test_source3.tsv"))
    test_target = pd.concat([test_s2, test_s3], ignore_index=True)
    print(f"Loaded Test: {len(test_s1)} S1 entities, {len(test_s2)} S2, {len(test_s3)} S3.")

    # Candidate Blocking on Test
    print("Generating candidate pairs for Test set...")
    test_candidates = blocker.generate_candidates(test_s1, test_s2, test_s3)
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    export_submission_tsv(test_candidates, cand_pairs_path, column_name="candidate_entity_ids")

    # Feature Extraction on Test
    print("Extracting features for Test candidate pairs...")
    df_test_pairs = build_feature_dataframe(test_candidates, test_s1, test_target)
    
    if len(df_test_pairs) > 0:
        test_probs = model.predict_proba(df_test_pairs)
        test_preds = generate_predictions(df_test_pairs, test_probs, test_s1["entity_id"].tolist(), threshold=best_thresh)
    else:
        test_preds = {sid: [] for sid in test_s1["entity_id"]}

    # Export matching_results.tsv
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    export_submission_tsv(test_preds, matching_path, column_name="matched_entity_ids")

    # 6. Validate Outputs
    print("\n[Step 6/6] Validating Submission Files with validate_submission.py...")
    val_script = os.path.join(root_dir, "utils/validate_submission.py")
    cmd = [
        sys.executable, val_script,
        "--matching", matching_path,
        "--candidate", cand_pairs_path,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    
    if res.returncode == 0:
        print("🎉 SUCCESS: All submission files generated and validated successfully!")
    else:
        print("⚠️ Validation issues detected. Check output above.")

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train dataset")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset")
    parser.add_argument("--output-dir", default="output", help="Path to save outputs")
    parser.add_argument("--model-path", default="models/lgb_model.joblib", help="Model path")
    args = parser.parse_args()

    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_save_path=args.model_path
    )

if __name__ == "__main__":
    main()
