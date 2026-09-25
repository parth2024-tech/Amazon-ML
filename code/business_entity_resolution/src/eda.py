#!/usr/bin/env python3
"""
eda.py
Comprehensive Exploratory Data Analysis module for Entity Resolution.
Computes token frequencies, missing rates, duplicate rates, length distributions,
and ground-truth match count / singleton statistics.
"""

from collections import Counter
from typing import Dict, List, Set, Optional
import pandas as pd
import numpy as np

def run_source_eda(df: pd.DataFrame, source_name: str) -> dict:
    """Computes distributions and statistics for a single source file."""
    total = len(df)
    print(f"\n{'='*20} {source_name} (Total rows: {total}) {'='*20}")

    # Missing fields
    missing = {
        "missing_name": (df["business_name"].str.strip() == "").sum(),
        "missing_address": (df["business_address"].str.strip() == "").sum(),
        "missing_country": (df["country"].str.strip() == "").sum(),
    }
    for k, v in missing.items():
        print(f"  - {k}: {v} ({v/total*100:.2f}%)")

    # Name and Address length stats
    name_chars = df["business_name"].str.len()
    name_words = df["business_name"].str.split().str.len()
    addr_chars = df["business_address"].str.len()
    addr_words = df["business_address"].str.split().str.len()

    print(f"  - Name Char Length: mean={name_chars.mean():.1f}, min={name_chars.min()}, max={name_chars.max()}")
    print(f"  - Name Word Count: mean={name_words.mean():.1f}, min={name_words.min()}, max={name_words.max()}")
    print(f"  - Addr Char Length: mean={addr_chars.mean():.1f}, min={addr_chars.min()}, max={addr_chars.max()}")
    print(f"  - Addr Word Count: mean={addr_words.mean():.1f}, min={addr_words.min()}, max={addr_words.max()}")

    # Country distribution
    country_counts = df["country"].value_counts().to_dict()
    print(f"  - Countries: {country_counts}")

    # Duplicates within source
    dup_names = df["norm_name"].duplicated().sum()
    dup_pairs = df.duplicated(subset=["norm_name", "norm_address"]).sum()
    print(f"  - Duplicate normalized names: {dup_names} ({dup_names/total*100:.2f}%)")
    print(f"  - Duplicate (name, address) pairs: {dup_pairs} ({dup_pairs/total*100:.2f}%)")

    # Top name tokens
    all_tokens = [t for name in df["norm_name"] for t in name.split()]
    top_tokens = Counter(all_tokens).most_common(10)
    print(f"  - Top 10 Name Tokens: {top_tokens}")

    return {
        "source": source_name,
        "total_rows": total,
        "missing": missing,
        "country_counts": country_counts,
        "dup_names": int(dup_names),
        "dup_pairs": int(dup_pairs)
    }

def run_ground_truth_eda(gt: dict[str, set[str]], df_s1: pd.DataFrame) -> dict:
    """Analyzes the ground-truth match distribution and singleton statistics."""
    total_s1 = len(df_s1)
    match_counts = [len(gt.get(sid, set())) for sid in df_s1["entity_id"]]
    count_dist = Counter(match_counts)

    singletons = count_dist.get(0, 0)
    total_matches = sum(match_counts)

    print(f"\n{'='*20} Ground Truth Analysis {'='*20}")
    print(f"  - Total S1 Entities: {total_s1}")
    print(f"  - Singletons (0 matches): {singletons} ({singletons/total_s1*100:.2f}%)")
    print(f"  - Total Linked Targets: {total_matches}")
    print("  - Match Count Distribution:")
    for count in sorted(count_dist.keys()):
        print(f"      * {count} matches: {count_dist[count]} entities ({count_dist[count]/total_s1*100:.2f}%)")

    # Source breakdown for linked entities
    s2_matches = 0
    s3_matches = 0
    for matches in gt.values():
        for m in matches:
            if m.startswith("S2-"):
                s2_matches += 1
            elif m.startswith("S3-"):
                s3_matches += 1
    print(f"  - Matched Source 2 Records: {s2_matches} ({s2_matches/total_matches*100:.2f}%)" if total_matches else "  - Matched Source 2: 0")
    print(f"  - Matched Source 3 Records: {s3_matches} ({s3_matches/total_matches*100:.2f}%)" if total_matches else "  - Matched Source 3: 0")

    return {
        "total_s1": total_s1,
        "singletons": singletons,
        "singleton_rate": singletons / total_s1 if total_s1 else 0.0,
        "match_count_distribution": dict(count_dist),
        "s2_matches": s2_matches,
        "s3_matches": s3_matches
    }

def run_cross_source_overlap(df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame):
    """Computes exact string overlap across sources."""
    print(f"\n{'='*20} Exact String Overlap Across Sources {'='*20}")
    s1_names = set(df_s1["norm_name"].loc[lambda s: s != ""])
    s2_names = set(df_s2["norm_name"].loc[lambda s: s != ""])
    s3_names = set(df_s3["norm_name"].loc[lambda s: s != ""])

    print(f"  - S1 exact name in S2: {len(s1_names & s2_names)} ({len(s1_names & s2_names)/len(s1_names)*100:.2f}%)")
    print(f"  - S1 exact name in S3: {len(s1_names & s3_names)} ({len(s1_names & s3_names)/len(s1_names)*100:.2f}%)")
    print(f"  - S1 exact name in (S2 or S3): {len(s1_names & (s2_names | s3_names))} ({len(s1_names & (s2_names | s3_names))/len(s1_names)*100:.2f}%)")

def analyze_full_dataset(df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame, gt: dict[str, set[str]] | None = None):
    run_source_eda(df_s1, "Source 1 (Reference)")
    run_source_eda(df_s2, "Source 2 (Noisy)")
    run_source_eda(df_s3, "Source 3 (Noisy)")
    run_cross_source_overlap(df_s1, df_s2, df_s3)
    if gt is not None:
        run_ground_truth_eda(gt, df_s1)
