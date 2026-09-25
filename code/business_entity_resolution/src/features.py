#!/usr/bin/env python3
"""
features.py
Rich pairwise feature extraction between Source 1 and candidate target records.
Implements lexical, phonetic, token, numerical, and cross-field interaction features.
"""

from typing import Dict, List, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

def get_char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def extract_pairwise_features(s1: dict, target: dict) -> dict:
    """Extracts pairwise feature vector for a candidate pair."""
    s1_name = s1["norm_name"]
    t_name = target["norm_name"]
    s1_addr = s1["norm_address"]
    t_addr = target["norm_address"]

    # 1. Name Features
    name_exact = 1.0 if s1_name == t_name and s1_name else 0.0
    name_ratio = fuzz.ratio(s1_name, t_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, t_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, t_name) / 100.0
    name_jw = distance.JaroWinkler.similarity(s1_name, t_name)

    s1_words = set(s1_name.split())
    t_words = set(t_name.split())
    name_word_jaccard = jaccard_similarity(s1_words, t_words)
    name_token_overlap_count = len(s1_words & t_words)

    s1_ngrams = get_char_ngrams(s1_name, 3)
    t_ngrams = get_char_ngrams(t_name, 3)
    name_ngram_jaccard = jaccard_similarity(s1_ngrams, t_ngrams)

    first_word_match = 1.0 if (s1_name.split()[:1] == t_name.split()[:1] and s1_name) else 0.0

    len1 = len(s1_name)
    len2 = len(t_name)
    name_len_diff = abs(len1 - len2)
    name_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

    # 2. Address Features
    addr_exact = 1.0 if s1_addr == t_addr and s1_addr else 0.0
    addr_ratio = fuzz.ratio(s1_addr, t_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, t_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
    addr_jw = distance.JaroWinkler.similarity(s1_addr, t_addr)

    s1_addr_words = set(s1_addr.split())
    t_addr_words = set(t_addr.split())
    addr_word_jaccard = jaccard_similarity(s1_addr_words, t_addr_words)

    # Numeric & Postal / Geographic overlap
    s1_geos = set(s1.get("geo_tokens", []))
    t_geos = set(target.get("geo_tokens", []))
    geo_jaccard = jaccard_similarity(s1_geos, t_geos)
    has_common_geo = 1.0 if (s1_geos & t_geos) else 0.0
    # Contradictory address: both have postal codes/numbers but zero overlap
    contradictory_geo = 1.0 if (len(s1_geos) > 0 and len(t_geos) > 0 and len(s1_geos & t_geos) == 0) else 0.0

    # 3. Cross-Field Interaction Features
    strong_name_weak_addr = 1.0 if (name_token_set >= 0.85 and addr_token_set < 0.50) else 0.0
    weak_name_strong_addr = 1.0 if (name_token_set < 0.60 and addr_token_set >= 0.80) else 0.0
    strong_both = 1.0 if (name_token_set >= 0.80 and addr_token_set >= 0.80) else 0.0

    # Holistic full record text
    s1_full = s1_name + " " + s1_addr
    t_full = t_name + " " + t_addr
    full_token_set = fuzz.token_set_ratio(s1_full, t_full) / 100.0
    full_jw = distance.JaroWinkler.similarity(s1_full, t_full)

    # Missing field flags
    s1_missing_addr = 1.0 if not s1_addr else 0.0
    t_missing_addr = 1.0 if not t_addr else 0.0

    # Source & Country Metadata
    country_match = 1.0 if s1["norm_country"] == target["norm_country"] else 0.0
    is_source2 = 1.0 if target["entity_id"].startswith("S2-") else 0.0
    is_source3 = 1.0 if target["entity_id"].startswith("S3-") else 0.0

    return {
        "name_exact": name_exact,
        "name_ratio": name_ratio,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jw": name_jw,
        "name_word_jaccard": name_word_jaccard,
        "name_token_overlap_count": name_token_overlap_count,
        "name_ngram_jaccard": name_ngram_jaccard,
        "first_word_match": first_word_match,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_exact": addr_exact,
        "addr_ratio": addr_ratio,
        "addr_partial": addr_partial,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jw": addr_jw,
        "addr_word_jaccard": addr_word_jaccard,
        "geo_jaccard": geo_jaccard,
        "has_common_geo": has_common_geo,
        "contradictory_geo": contradictory_geo,
        "strong_name_weak_addr": strong_name_weak_addr,
        "weak_name_strong_addr": weak_name_strong_addr,
        "strong_both": strong_both,
        "full_token_set": full_token_set,
        "full_jw": full_jw,
        "s1_missing_addr": s1_missing_addr,
        "t_missing_addr": t_missing_addr,
        "country_match": country_match,
        "is_source2": is_source2,
        "is_source3": is_source3
    }

FEATURE_COLUMNS = [
    "name_exact", "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jw",
    "name_word_jaccard", "name_token_overlap_count", "name_ngram_jaccard", "first_word_match",
    "name_len_diff", "name_len_ratio",
    "addr_exact", "addr_ratio", "addr_partial", "addr_token_sort", "addr_token_set", "addr_jw",
    "addr_word_jaccard", "geo_jaccard", "has_common_geo", "contradictory_geo",
    "strong_name_weak_addr", "weak_name_strong_addr", "strong_both",
    "full_token_set", "full_jw", "s1_missing_addr", "t_missing_addr",
    "country_match", "is_source2", "is_source3"
]

def generate_features_dataset(
    candidate_map: dict[str, list[str]],
    df_s1: pd.DataFrame,
    df_target: pd.DataFrame
) -> pd.DataFrame:
    """Constructs the pairwise feature dataframe for all candidate pairs."""
    s1_dict = df_s1.set_index("entity_id", drop=False).to_dict(orient="index")
    target_dict = df_target.set_index("entity_id", drop=False).to_dict(orient="index")

    rows = []
    for s1_id, cand_ids in candidate_map.items():
        if s1_id not in s1_dict:
            continue
        s1_row = s1_dict[s1_id]
        for tid in cand_ids:
            if tid not in target_dict:
                continue
            t_row = target_dict[tid]
            feats = extract_pairwise_features(s1_row, t_row)
            feats["s1_id"] = s1_id
            feats["target_id"] = tid
            rows.append(feats)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
