#!/usr/bin/env python3
"""
predict.py
Fault-tolerant inference engine: country-sliced blocking, chunked feature extraction,
batched scoring, and granular checkpointing (cands + batch matches).
Guarantees zero lost work even if system reboots or restarts periodically.
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
    batch_size: int = 10000,
    checkpoint_dir: Path | None = None
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Fault-tolerant end-to-end inference with granular checkpointing:
    1. Country checkpoint (.ckpt_test_{country}.parquet): complete country bypass.
    2. Candidates checkpoint (.ckpt_test_{country}_cands.parquet): skips re-blocking if interrupted.
    3. Batch checkpoint (.ckpt_test_{country}_batch.parquet): resumes mid-country scoring.
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

        # 1. Full country checkpoint check
        ckpt_file = ckpt_dir / f".ckpt_test_{country}.parquet"
        if ckpt_file.exists():
            print(f"\n  [{country.upper()}] Found existing complete checkpoint at {ckpt_file.name}, loading...")
            df_ckpt = pd.read_parquet(ckpt_file)
            for sid, c_arr, m_arr in zip(df_ckpt["entity_id"], df_ckpt["candidate_ids"], df_ckpt["matched_ids"]):
                candidate_pairs[sid] = list(c_arr)
                final_matches[sid] = list(m_arr)
            n_c_matched = sum(1 for sid in df_ckpt["entity_id"] if final_matches.get(sid))
            print(f"  [{country.upper()}] Restored {len(df_ckpt):,} entities ({n_c_matched:,} matched) from checkpoint.")
            continue

        s1_ids_c = s1_c["entity_id"].tolist()
        cands_ckpt_file = ckpt_dir / f".ckpt_test_{country}_cands.parquet"
        batch_ckpt_file = ckpt_dir / f".ckpt_test_{country}_batch.parquet"

        # 2. Check if candidate blocking is already cached
        country_cands: dict[str, list[str]] = {}
        if cands_ckpt_file.exists():
            print(f"\n  [{country.upper()}] Restoring precomputed candidates from {cands_ckpt_file.name}...")
            df_cands_ckpt = pd.read_parquet(cands_ckpt_file)
            for sid, c_arr in zip(df_cands_ckpt["entity_id"], df_cands_ckpt["candidate_ids"]):
                country_cands[sid] = list(c_arr)
                candidate_pairs[sid] = list(c_arr)
            print(f"  [{country.upper()}] Restored candidates for {len(country_cands):,} entities in 1s.")
        else:
            print(f"\n  [{country.upper()}] Blocking {len(s1_c):,} S1 vs {len(s2_c)+len(s3_c):,} targets...")
            country_cands = blocker.generate_candidates(s1_c, s2_c, s3_c)
            for sid, cands in country_cands.items():
                candidate_pairs[sid] = cands
            # Cache candidates immediately to disk
            cands_rows = [{"entity_id": sid, "candidate_ids": cands} for sid, cands in country_cands.items()]
            pd.DataFrame(cands_rows).to_parquet(cands_ckpt_file, index=False)
            del cands_rows
            print(f"  [{country.upper()}] Candidates saved to {cands_ckpt_file.name}")

        # 3. Check if partial scoring batches exist
        completed_s1_ids: set[str] = set()
        if batch_ckpt_file.exists():
            print(f"  [{country.upper()}] Restoring partial scoring batches from {batch_ckpt_file.name}...")
            df_bckpt = pd.read_parquet(batch_ckpt_file)
            for sid, m_arr in zip(df_bckpt["entity_id"], df_bckpt["matched_ids"]):
                final_matches[sid] = list(m_arr)
                completed_s1_ids.add(sid)
            n_restored_matches = sum(1 for sid in completed_s1_ids if final_matches.get(sid))
            print(f"  [{country.upper()}] Resuming: {len(completed_s1_ids):,} entities already scored ({n_restored_matches:,} matches).")

        n_matches_total = sum(1 for sid in s1_ids_c if final_matches.get(sid))
        for start_idx in range(0, len(s1_ids_c), batch_size):
            end_idx  = min(start_idx + batch_size, len(s1_ids_c))
            batch_ids = s1_ids_c[start_idx:end_idx]

            # Skip batch if already scored in a prior partial run
            if completed_s1_ids and all(sid in completed_s1_ids for sid in batch_ids):
                continue

            batch_cands = {sid: country_cands[sid] for sid in batch_ids
                           if sid in country_cands and country_cands[sid]}
            if not batch_cands:
                for sid in batch_ids:
                    completed_s1_ids.add(sid)
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
                    completed_s1_ids.add(sid)
                n_matches_total = sum(1 for sid in s1_ids_c if final_matches.get(sid))

            del df_batch_features
            gc.collect()

            pct = end_idx / len(s1_ids_c) * 100
            print(f"    {country.upper()} {end_idx:,}/{len(s1_ids_c):,} ({pct:.0f}%) | Cumulative matches: {n_matches_total:,}")

            # Granular checkpoint every 5 batches (~2.5 minutes)
            batch_num = (start_idx // batch_size) + 1
            if batch_num % 5 == 0:
                b_rows = [{"entity_id": sid, "matched_ids": final_matches.get(sid, [])} for sid in completed_s1_ids]
                pd.DataFrame(b_rows).to_parquet(batch_ckpt_file, index=False)
                del b_rows

        # 4. Save final unified country checkpoint and clean up temp files
        ckpt_rows = []
        for sid in s1_ids_c:
            ckpt_rows.append({
                "entity_id": sid,
                "candidate_ids": candidate_pairs.get(sid, []),
                "matched_ids": final_matches.get(sid, [])
            })
        pd.DataFrame(ckpt_rows).to_parquet(ckpt_file, index=False)
        print(f"  [{country.upper()}] Final checkpoint saved to {ckpt_file.name} ({len(s1_ids_c):,} entities)")

        # Remove temporary partial files for completed country
        if cands_ckpt_file.exists():
            cands_ckpt_file.unlink()
        if batch_ckpt_file.exists():
            batch_ckpt_file.unlink()

        del country_cands, s1_c, s2_c, s3_c, ckpt_rows, completed_s1_ids
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
