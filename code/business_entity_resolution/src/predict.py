#!/usr/bin/env python3
"""
predict.py
Inference engine: generates blocking candidates, extracts features, and produces final entity links.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .blocking import MultiIndexBlocker
from .features import generate_features_dataset
from .train import EntityClassifier
from .threshold import apply_threshold_to_candidates

def run_test_inference(
    df_test_s1: pd.DataFrame,
    df_test_s2: pd.DataFrame,
    df_test_s3: pd.DataFrame,
    classifier: EntityClassifier,
    blocker: MultiIndexBlocker,
    threshold: float
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Runs end-to-end inference over test datasets:
    Returns (candidate_pairs_dict, final_matches_dict).
    """
    print(f"Generating candidate pairs for {len(df_test_s1)} Test Source 1 entities...")
    candidate_pairs = blocker.generate_candidates(df_test_s1, df_test_s2, df_test_s3)

    df_test_target = pd.concat([df_test_s2, df_test_s3], ignore_index=True)
    print("Extracting features for candidate pairs...")
    df_test_features = generate_features_dataset(candidate_pairs, df_test_s1, df_test_target)

    all_s1_ids = df_test_s1["entity_id"].tolist()

    if len(df_test_features) > 0:
        print(f"Scoring {len(df_test_features)} candidate pairs with {classifier.model_type}...")
        probs = classifier.predict_proba(df_test_features)
        final_matches = apply_threshold_to_candidates(
            df_test_features, probs, all_s1_ids, threshold=threshold
        )
    else:
        print("Warning: No candidate pairs generated. Predicting all singletons.")
        final_matches = {sid: [] for sid in all_s1_ids}

    # Verify every match is a subset of candidates
    for sid, matches in final_matches.items():
        cands = set(candidate_pairs.get(sid, []))
        for m in matches:
            if m not in cands:
                candidate_pairs[sid].append(m)

    return candidate_pairs, final_matches
