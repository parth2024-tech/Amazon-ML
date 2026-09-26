#!/usr/bin/env python3
"""
load_data.py
Robust TSV data loading, schema verification, and precomputation of normalized representations.
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
import pandas as pd

from .normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_geographic_tokens
)

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

def load_tsv_file(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Loads a single TSV file with strict tab delimiter and string dtypes.
    Uses automatic parquet caching for precomputed normalized representations.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path.resolve()}")

    cache_path = path.parent / f".cache_{path.stem}.parquet" if nrows is None else None
    if cache_path and cache_path.exists():
        if cache_path.stat().st_mtime >= path.stat().st_mtime:
            print(f"Loading cached precomputed representations from {cache_path.name}...")
            return pd.read_parquet(cache_path)

    print(f"Reading {path.name}...")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows).fillna("")
    
    # Check for empty columns or bad single-line parsing
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        raise ValueError(f"File {path.name} was read incorrectly with only 1 column. Ensure tab delimiter is used.")

    for col in REQUIRED_SOURCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    print(f"Precomputing normalized representations for {len(df):,} records in {path.name}...")
    df["norm_name"] = df["business_name"].apply(normalize_business_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    df["norm_country"] = df["country"].apply(normalize_country)

    if cache_path:
        try:
            print(f"Saving precomputed representations cache to {cache_path.name}...")
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: could not write cache: {e}")

    return df

def load_ground_truth(file_path: str | Path, s1_filter: set[str] | None = None) -> dict[str, set[str]]:
    """
    Loads train_ground_truth.tsv into {source1_entity_id: set(matched_entity_ids)}.
    Singletons map to empty sets: {s1_id: set()}.
    Uses memory-efficient line streaming.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path.resolve()}")

    print(f"Reading ground truth from {path.name}...")
    gt_map: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            return gt_map
        
        # If the first line doesn't have the header name, process it
        if "source1_entity_id" not in first_line:
            parts = first_line.rstrip("\r\n").split("\t")
            if parts and parts[0].strip():
                s1 = parts[0].strip()
                if s1_filter is None or s1 in s1_filter:
                    match_str = parts[1].strip() if len(parts) > 1 else ""
                    gt_map[s1] = set(x.strip() for x in match_str.split(",") if x.strip()) if match_str else set()

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            s1_id = parts[0].strip()
            if s1_filter is not None and s1_id not in s1_filter:
                continue
            match_str = parts[1].strip() if len(parts) > 1 else ""
            if match_str:
                matches = set(x.strip() for x in match_str.split(",") if x.strip())
            else:
                matches = set()
            gt_map[s1_id] = matches

    print(f"Loaded ground truth for {len(gt_map):,} entities.")
    return gt_map


def load_dataset_split(
    dir_path: str | Path,
    split_prefix: str = "train",
    sample_s1: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]] | None]:
    """
    Loads source1, source2, source3 (and ground truth if train) from a directory.
    If sample_s1 is provided, loads only sample_s1 rows for source1.
    """
    d = Path(dir_path)
    s1_path = d / f"{split_prefix}_source1.tsv"
    s2_path = d / f"{split_prefix}_source2.tsv"
    s3_path = d / f"{split_prefix}_source3.tsv"

    df_s1 = load_tsv_file(s1_path, nrows=sample_s1)
    df_s2 = load_tsv_file(s2_path)
    df_s3 = load_tsv_file(s3_path)

    gt = None
    if split_prefix == "train":
        gt_path = d / "train_ground_truth.tsv"
        if gt_path.exists():
            s1_set = set(df_s1["entity_id"]) if sample_s1 else None
            gt = load_ground_truth(gt_path, s1_filter=s1_set)

    return df_s1, df_s2, df_s3, gt
