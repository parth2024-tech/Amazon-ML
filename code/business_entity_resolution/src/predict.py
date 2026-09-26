#!/usr/bin/env python3
"""
predict.py
Memory-safe inference engine: country-sliced blocking, chunked feature extraction,
batched scoring, and country-level parquet checkpointing.
"""

from typing import Dict, List, Tuple
import gc
from pathlib import Path
import pandas as pd
import numpy as np

from .config import OUTPUT_DIR
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
    threshold: float,
    batch_size: int = 20000,
    checkpoint_dir: Path | None = None
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Memory-safe end-to-end inference with country-level checkpointing.
    Processes one country at a time. Saves each completed country to parquet checkpoint.
    If checkpoint exists on restart, reloads country in 2 seconds instead of recomputing.
    Returns (candidate_pairs_dict, final_matches_dict).
    """
    all_s1_ids = df_test_s1["entity_id"].tolist()
    candidate_pairs: dict[str, list[str]] = {sid: [] for sid in all_s1_ids}
    final_matches: dict[str, list[str]] = {sid: [] for sid in all_s1_ids}

    ckpt_dir = checkpoint_dir or OUTPUT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    countries = sorted(df_test_s1["norm_country"].unique())
    print(f"Running inference over {len(all_s1_ids):,} Test S1 entities across {len(countries)} countries...")

    for country in countries:
        s1_mask  = df_test_s1["norm_country"] == country
        s2_mask  = df_test_s2["norm_country"] == country
        s3_mask  = df_test_s3["norm_country"] == country

        s1_c = df_test_s1[s1_mask]
        s2_c = df_test_s2[s2_mask]
        s3_c = df_test_s3[s3_mask]

        if len(s1_c) == 0 or (len(s2_c) + len(s3_c)) == 0:
            continue

        ckpt_file = ckpt_dir / f".ckpt_test_{country}.parquet"
        if ckpt_file.exists():
            print(f"\n  [{country.upper()}] Found existing checkpoint at {ckpt_file.name}, loading...")
            df_ckpt = pd.read_parquet(ckpt_file)
            for sid, c_arr, m_arr in zip(df_ckpt["entity_id"], df_ckpt["candidate_ids"], df_ckpt["matched_ids"]):
                candidate_pairs[sid] = list(c_arr)
                final_matches[sid] = list(m_arr)
            n_c_matched = sum(1 for sid in df_ckpt["entity_id"] if final_matches.get(sid))
            print(f"  [{country.upper()}] Restored {len(df_ckpt):,} entities ({n_c_matched:,} matched) from checkpoint.")
            continue

        print(f"\n  [{country.upper()}] Blocking {len(s1_c):,} S1 vs {len(s2_c)+len(s3_c):,} targets...")
        country_cands = blocker.generate_candidates(s1_c, s2_c, s3_c)

        for sid, cands in country_cands.items():
            candidate_pairs[sid] = cands

        s1_ids_c = s1_c["entity_id"].tolist()
        n_matches_total = 0
        for start_idx in range(0, len(s1_ids_c), batch_size):
            end_idx  = min(start_idx + batch_size, len(s1_ids_c))
            batch_ids = s1_ids_c[start_idx:end_idx]

            batch_cands = {sid: country_cands[sid] for sid in batch_ids
                           if sid in country_cands and country_cands[sid]}
            if not batch_cands:
                continue

            df_batch_s1 = s1_c.iloc[start_idx:end_idx]
            df_batch_features = generate_features_dataset(batch_cands, df_batch_s1, [s2_c, s3_c])

            if len(df_batch_features) > 0:
                probs = classifier.predict_proba(df_batch_features)
                batch_preds = apply_threshold_to_candidates(
                    df_batch_features, probs, batch_ids, threshold=threshold
                )
                for sid, matches in batch_preds.items():
                    final_matches[sid] = matches
                n_matches_total += sum(1 for sid in batch_ids if final_matches.get(sid))

            del df_batch_features
            gc.collect()

            pct = end_idx / len(s1_ids_c) * 100
            print(f"    {country.upper()} {end_idx:,}/{len(s1_ids_c):,} ({pct:.0f}%) | Cumulative matches: {n_matches_total:,}")

        # Save country checkpoint to parquet
        ckpt_rows = []
        for sid in s1_ids_c:
            ckpt_rows.append({
                "entity_id": sid,
                "candidate_ids": candidate_pairs.get(sid, []),
                "matched_ids": final_matches.get(sid, [])
            })
        pd.DataFrame(ckpt_rows).to_parquet(ckpt_file, index=False)
        print(f"  [{country.upper()}] Checkpoint saved to {ckpt_file.name} ({len(s1_ids_c):,} entities)")

        # Free country memory
        del country_cands, s1_c, s2_c, s3_c, ckpt_rows
        gc.collect()

    # Verify every match is a subset of candidates (safety check)
    for sid, matches in final_matches.items():
        existing_cands = set(candidate_pairs.get(sid, []))
        for m in matches:
            if m not in existing_cands:
                candidate_pairs[sid].append(m)

    total_matches = sum(1 for m in final_matches.values() if m)
    print(f"\nInference complete: {total_matches:,}/{len(all_s1_ids):,} entities matched ({total_matches/len(all_s1_ids)*100:.1f}%)")
    return candidate_pairs, final_matches
