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

def load_tsv_file(file_path: str | Path) -> pd.DataFrame:
    """
    Loads a single TSV file with strict tab delimiter and string dtypes.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path.resolve()}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).fillna("")
    
    # Check for empty columns or bad single-line parsing
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        raise ValueError(f"File {path.name} was read incorrectly with only 1 column. Ensure tab delimiter is used.")

    for col in REQUIRED_SOURCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Precompute normalized columns
    df["norm_name"] = df["business_name"].apply(normalize_business_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    df["norm_country"] = df["country"].apply(normalize_country)
    df["geo_tokens"] = df["business_address"].apply(extract_geographic_tokens)

    return df

def load_ground_truth(file_path: str | Path) -> dict[str, set[str]]:
    """
    Loads train_ground_truth.tsv into {source1_entity_id: set(matched_entity_ids)}.
    Singletons map to empty sets: {s1_id: set()}.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path.resolve()}")

    gt_map: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return gt_map
        
        # Check header
        header = lines[0].strip().split("\t")
        start_idx = 1 if "source1_entity_id" in header[0] else 0

        for line in lines[start_idx:]:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            s1_id = parts[0].strip()
            match_str = parts[1].strip() if len(parts) > 1 else ""
            if match_str:
                matches = set(x.strip() for x in match_str.split(",") if x.strip())
            else:
                matches = set()
            gt_map[s1_id] = matches

    return gt_map

def load_dataset_split(
    dir_path: str | Path,
    split_prefix: str = "train"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]] | None]:
    """
    Loads source1, source2, source3 (and ground truth if train) from a directory.
    """
    d = Path(dir_path)
    s1_path = d / f"{split_prefix}_source1.tsv"
    s2_path = d / f"{split_prefix}_source2.tsv"
    s3_path = d / f"{split_prefix}_source3.tsv"

    df_s1 = load_tsv_file(s1_path)
    df_s2 = load_tsv_file(s2_path)
    df_s3 = load_tsv_file(s3_path)

    gt = None
    if split_prefix == "train":
        gt_path = d / "train_ground_truth.tsv"
        if gt_path.exists():
            gt = load_ground_truth(gt_path)

    return df_s1, df_s2, df_s3, gt
