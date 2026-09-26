#!/usr/bin/env python3
"""
features.py
Rich pairwise feature extraction between Source 1 and candidate target records.
Optimized with UnaryProfile precomputation: computes lexical, phonetic, and numeric
profiles ONCE per entity, yielding 3-4x faster pairwise feature extraction.
"""

from typing import Dict, List, Set
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from .normalize import extract_geographic_tokens

# ── Phonetic key ─────────────────────────────────────────────────────────────
def _metaphone_key(text: str) -> str:
    if not text or len(text) < 2:
        return text
    s = text.lower()
    result = s[0]
    for ch in s[1:]:
        if ch != result[-1]:
            result += ch
    if result.endswith('e') and len(result) > 2:
        result = result[:-1]
    result = re.sub(r'ck', 'k', result)
    result = re.sub(r'ph', 'f', result)
    result = re.sub(r'sch', 'sk', result)
    result = re.sub(r'th', '0', result)
    result = re.sub(r'qu', 'k', result)
    first = result[0]
    rest = re.sub(r'[aeiou]', '', result[1:])
    return first + rest

_LEGAL_TOKENS_F = {
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp',
    'corporation', 'llc', 'llp', 'co', 'company', 'services', 'service',
    'solutions', 'solution', 'holdings', 'group', 'enterprises', 'center',
    'centre', 'associates', 'consultants', 'consultant', 'consulting',
    'international', 'intl', 'gmbh', 'sarl', 'sas', 'sa'
}

def _make_pkey(name: str) -> str:
    words = [w for w in name.split() if w not in _LEGAL_TOKENS_F and len(w) >= 2][:3]
    return " ".join(_metaphone_key(w) for w in words) if words else ""

def get_char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


class UnaryProfile:
    """Precomputed unary profile for an entity to avoid recomputing in pairwise loop."""
    __slots__ = (
        'name', 'addr', 'country', 'entity_id',
        'words', 'ngrams3', 'ngrams4',
        'first_word', 'last_word', 'len_name', 'wc',
        'pkey', 'addr_words', 'geos',
        'first_num', 'num_set',
        'is_s2', 'is_s3', 'country_id', 'full'
    )

    def __init__(self, r: dict):
        name = str(r.get("norm_name", "") or "")
        addr = str(r.get("norm_address", "") or "")
        country = str(r.get("norm_country", "") or "")
        eid = str(r.get("entity_id", "") or "")

        toks = name.split()
        words = set(toks)
        content = [w for w in toks if w not in _LEGAL_TOKENS_F]

        s_nums = re.findall(r"\d+", addr)
        first_num = s_nums[0].lstrip("0") if (s_nums and s_nums[0].lstrip("0")) else ""
        num_set = {n.lstrip("0") for n in s_nums if n.lstrip("0")}

        raw_geos = r.get("geo_tokens")
        if raw_geos:
            geos = set(raw_geos)
        else:
            geos = set(extract_geographic_tokens(r.get("business_address", "")))

        self.name = name
        self.addr = addr
        self.country = country
        self.entity_id = eid
        self.words = words
        self.ngrams3 = get_char_ngrams(name, 3)
        self.ngrams4 = get_char_ngrams(name, 4)
        self.first_word = toks[0] if toks else ""
        self.last_word = content[-1] if content else ""
        self.len_name = len(name)
        self.wc = len(words)
        self.pkey = _make_pkey(name)
        self.addr_words = set(addr.split())
        self.geos = geos
        self.first_num = first_num
        self.num_set = num_set
        self.is_s2 = 1.0 if eid.startswith("S2-") else 0.0
        self.is_s3 = 1.0 if eid.startswith("S3-") else 0.0
        self.country_id = {"india": 0.0, "us": 1.0, "france": 2.0}.get(country, -1.0)
        self.full = name + " " + addr


def extract_pairwise_features(s1: UnaryProfile | dict, target: UnaryProfile | dict) -> dict:
    """Extracts 53-feature pairwise vector for a candidate pair."""
    p1: UnaryProfile = s1 if isinstance(s1, UnaryProfile) else UnaryProfile(s1)
    p2: UnaryProfile = target if isinstance(target, UnaryProfile) else UnaryProfile(target)

    s1_name, t_name = p1.name, p2.name
    s1_addr, t_addr = p1.addr, p2.addr

    # ── 1. Name Lexical Features ──────────────────────────────────────────
    name_exact      = 1.0 if s1_name == t_name and s1_name else 0.0
    name_ratio      = fuzz.ratio(s1_name, t_name) / 100.0
    name_partial    = fuzz.partial_ratio(s1_name, t_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
    name_token_set  = fuzz.token_set_ratio(s1_name, t_name) / 100.0
    name_jw         = distance.JaroWinkler.similarity(s1_name, t_name)

    common_words = p1.words & p2.words
    name_word_jaccard        = jaccard_similarity(p1.words, p2.words)
    name_token_overlap_count = len(common_words)

    name_ngram_jaccard = jaccard_similarity(p1.ngrams3, p2.ngrams3)
    name_4gram_jaccard = jaccard_similarity(p1.ngrams4, p2.ngrams4)

    first_word_match = 1.0 if (p1.first_word == p2.first_word and p1.first_word) else 0.0
    last_word_match  = 1.0 if (p1.last_word == p2.last_word and p1.last_word) else 0.0

    len1 = p1.len_name
    len2 = p2.len_name
    name_len_diff  = abs(len1 - len2)
    name_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

    name_containment_s1 = len(common_words) / max(p1.wc, 1)
    name_containment_t  = len(common_words) / max(p2.wc, 1)
    name_word_count_ratio = min(p1.wc, p2.wc) / max(p1.wc, p2.wc) if max(p1.wc, p2.wc) > 0 else 1.0

    # ── 2. Phonetic Feature ──────────────────────────────────────────────
    if p1.pkey and p2.pkey:
        name_phonetic_sim = fuzz.token_set_ratio(p1.pkey, p2.pkey) / 100.0
    else:
        name_phonetic_sim = 0.0

    # ── 3. Address Features ───────────────────────────────────────────────
    addr_exact      = 1.0 if s1_addr == t_addr and s1_addr else 0.0
    addr_ratio      = fuzz.ratio(s1_addr, t_addr) / 100.0
    addr_partial    = fuzz.partial_ratio(s1_addr, t_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
    addr_token_set  = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
    addr_jw         = distance.JaroWinkler.similarity(s1_addr, t_addr)

    addr_word_jaccard = jaccard_similarity(p1.addr_words, p2.addr_words)

    common_geos = p1.geos & p2.geos
    geo_jaccard      = jaccard_similarity(p1.geos, p2.geos)
    has_common_geo   = 1.0 if common_geos else 0.0
    contradictory_geo = 1.0 if (len(p1.geos) > 0 and len(p2.geos) > 0 and len(common_geos) == 0) else 0.0
    geo_overlap_count = len(common_geos)
    geo_overlap_ratio = geo_overlap_count / max(len(p1.geos | p2.geos), 1)

    addr_first_num_match = 1.0 if (p1.first_num and p2.first_num and p1.first_num == p2.first_num) else 0.0
    addr_num_jaccard     = jaccard_similarity(p1.num_set, p2.num_set)
    addr_num_overlap_count = len(p1.num_set & p2.num_set)

    # ── 4. Cross-Field Interaction Features ──────────────────────────────
    strong_name_weak_addr = 1.0 if (name_token_set >= 0.85 and addr_token_set < 0.50) else 0.0
    weak_name_strong_addr = 1.0 if (name_token_set < 0.60 and addr_token_set >= 0.80) else 0.0
    strong_both           = 1.0 if (name_token_set >= 0.80 and addr_token_set >= 0.80) else 0.0

    phonetic_strong_lex_weak = 1.0 if (name_phonetic_sim >= 0.75 and name_token_set < 0.60) else 0.0

    name_max_sim = max(name_ratio, name_token_sort, name_token_set, name_jw, name_phonetic_sim)
    addr_max_sim = max(addr_ratio, addr_token_sort, addr_token_set, addr_jw)
    combined_max = name_max_sim * 0.6 + addr_max_sim * 0.4

    full_token_set = fuzz.token_set_ratio(p1.full, p2.full) / 100.0
    full_jw        = distance.JaroWinkler.similarity(p1.full, p2.full)

    # ── 5. Structural / Meta Features ────────────────────────────────────
    s1_missing_addr   = 1.0 if not s1_addr else 0.0
    t_missing_addr    = 1.0 if not t_addr  else 0.0
    both_missing_addr = 1.0 if (not s1_addr and not t_addr) else 0.0

    country_match = 1.0 if p1.country == p2.country else 0.0

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
        "name_4gram_jaccard": name_4gram_jaccard,
        "first_word_match": first_word_match,
        "last_word_match": last_word_match,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "name_containment_s1": name_containment_s1,
        "name_containment_t": name_containment_t,
        "name_word_count_ratio": name_word_count_ratio,
        "name_phonetic_sim": name_phonetic_sim,
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
        "geo_overlap_count": geo_overlap_count,
        "geo_overlap_ratio": geo_overlap_ratio,
        "addr_first_num_match": addr_first_num_match,
        "addr_num_jaccard": addr_num_jaccard,
        "addr_num_overlap_count": addr_num_overlap_count,
        "strong_name_weak_addr": strong_name_weak_addr,
        "weak_name_strong_addr": weak_name_strong_addr,
        "strong_both": strong_both,
        "phonetic_strong_lex_weak": phonetic_strong_lex_weak,
        "name_max_sim": name_max_sim,
        "addr_max_sim": addr_max_sim,
        "combined_max": combined_max,
        "full_token_set": full_token_set,
        "full_jw": full_jw,
        "s1_missing_addr": s1_missing_addr,
        "t_missing_addr": t_missing_addr,
        "both_missing_addr": both_missing_addr,
        "country_match": country_match,
        "is_source2": p2.is_s2,
        "is_source3": p2.is_s3,
        "country_id": p1.country_id,
    }


FEATURE_COLUMNS = [
    # Name lexical
    "name_exact", "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jw",
    "name_word_jaccard", "name_token_overlap_count", "name_ngram_jaccard", "name_4gram_jaccard",
    "first_word_match", "last_word_match", "name_len_diff", "name_len_ratio",
    "name_containment_s1", "name_containment_t", "name_word_count_ratio",
    # Phonetic
    "name_phonetic_sim",
    # Address
    "addr_exact", "addr_ratio", "addr_partial", "addr_token_sort", "addr_token_set", "addr_jw",
    "addr_word_jaccard", "geo_jaccard", "has_common_geo", "contradictory_geo",
    "geo_overlap_count", "geo_overlap_ratio", "addr_first_num_match",
    "addr_num_jaccard", "addr_num_overlap_count",
    # Cross-field
    "strong_name_weak_addr", "weak_name_strong_addr", "strong_both",
    "phonetic_strong_lex_weak",
    "name_max_sim", "addr_max_sim", "combined_max",
    "full_token_set", "full_jw",
    # Structural
    "s1_missing_addr", "t_missing_addr", "both_missing_addr",
    "country_match", "is_source2", "is_source3", "country_id",
]


def generate_features_dataset(
    candidate_map: dict[str, list[str]],
    df_s1: pd.DataFrame,
    df_target: pd.DataFrame | list[pd.DataFrame]
) -> pd.DataFrame:
    """Constructs pairwise feature dataframe with UnaryProfile precomputation."""
    import time
    s1_dict = df_s1.set_index("entity_id", drop=False).to_dict(orient="index")

    needed_tids = {tid for cands in candidate_map.values() for tid in cands}
    target_dict: dict[str, dict] = {}
    if isinstance(df_target, list):
        for df in df_target:
            sub = df[df["entity_id"].isin(needed_tids)]
            target_dict.update(sub.set_index("entity_id", drop=False).to_dict(orient="index"))
    else:
        df_target_sub = df_target[df_target["entity_id"].isin(needed_tids)]
        target_dict = df_target_sub.set_index("entity_id", drop=False).to_dict(orient="index")

    # Precompute UnaryProfiles for S1 entities and needed target entities
    s1_profiles: dict[str, UnaryProfile] = {
        sid: UnaryProfile(r) for sid, r in s1_dict.items()
    }
    target_profiles: dict[str, UnaryProfile] = {
        tid: UnaryProfile(r) for tid, r in target_dict.items()
    }
    del s1_dict, target_dict

    total_pairs = sum(len(v) for v in candidate_map.values())
    items = [(k, v) for k, v in candidate_map.items() if v]

    rows: list[dict] = []
    t0 = time.time()
    for s1_id, cand_ids in items:
        if s1_id not in s1_profiles:
            continue
        p1 = s1_profiles[s1_id]
        for tid in cand_ids:
            if tid not in target_profiles:
                continue
            p2 = target_profiles[tid]
            feats = extract_pairwise_features(p1, p2)
            feats["s1_id"] = s1_id
            feats["target_id"] = tid
            rows.append(feats)
        if len(rows) % 300000 == 0 and len(rows) > 0:
            elapsed = time.time() - t0
            rate = len(rows) / max(elapsed, 0.01)
            print(f"    Features: {len(rows):,}/{total_pairs:,} pairs ({rate:.0f} pairs/s, ETA {(total_pairs - len(rows)) / rate / 60:.1f}m)")

    elapsed = time.time() - t0
    if rows:
        print(f"    Extracted {len(rows):,} pairs in {elapsed:.1f}s ({len(rows)/max(elapsed,0.01):.0f} pairs/s)")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
