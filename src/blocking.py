#!/usr/bin/env python3
"""
blocking.py
Ultra-fast, memory-safe multi-index blocking engine with chunked target processing
and precomputed S1 query structures.
Peak RAM: ~4-6 GB even for millions of targets.
"""

import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

LEGAL_TOKENS = {
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'services', 'service', 'solutions', 'solution',
    'holdings', 'group', 'enterprises', 'center', 'centre', 'associates',
    'consultants', 'consultant', 'consulting', 'international', 'intl', 'gmbh',
    'sarl', 'sas', 'sa'
}

COMMON_ADDR_STOPWORDS = {
    'road', 'street', 'avenue', 'lane', 'drive', 'nagar', 'near',
    'opposite', 'behind', 'floor', 'shop', 'plot', 'cross', 'main',
    'north', 'south', 'east', 'west', 'blvd', 'court',
    'circle', 'highway', 'parkway', 'suite', 'block', 'sector', 'village',
    'fl', 'st', 'rd', 'ave', 'dr', 'ln', 'colony', 'city', 'state', 'bazaar',
    'marg', 'bhavan', 'complex', 'towers', 'tower', 'building', 'bldg', 'plaza',
    'enclave', 'layout', 'dist', 'district', 'post', 'po', 'pin', 'zip', 'india', 'usa'
}

STOPWORDS = LEGAL_TOKENS | {'the', 'and', 'for', 'with', 'all', 'india', 'state', 'department'}


def _mkey(word: str) -> str:
    """Fast metaphone-style phonetic key."""
    s = word.lower()
    r = s[0]
    for c in s[1:]:
        if c != r[-1]:
            r += c
    if r.endswith('e') and len(r) > 2:
        r = r[:-1]
    r = re.sub(r'ck|qu', 'k', r)
    r = re.sub(r'ph', 'f', r)
    r = re.sub(r'th', '0', r)
    return r[0] + re.sub(r'[aeiou]', '', r[1:])


def _phonetic_key(name: str, n: int = 3) -> str:
    words = [w for w in name.split() if w not in LEGAL_TOKENS and len(w) >= 2][:n]
    return "_".join(_mkey(w) for w in words) if words else ""


def get_core_name(norm_name: str) -> str:
    words = [w for w in norm_name.split() if w not in LEGAL_TOKENS]
    return " ".join(words) if words else norm_name


def get_char_ngrams(text: str, n: int = 4) -> list[str]:
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def extract_addr_blocks(raw_addr: str) -> tuple[list[str], list[str]]:
    if not raw_addr:
        return [], []
    text = str(raw_addr).lower()
    text = re.sub(r"[.,\-–—_/\\()\"''`#@:;!*\[\]{}~?]", " ", text)
    raw_nums = re.findall(r"\d+", text)
    nums: list[str] = []
    for n in raw_nums:
        c = n.lstrip("0")
        if c and len(c) <= 8 and c not in nums:
            nums.append(c)
    words = re.findall(r"\b[a-z]{3,}\b", text)
    return nums, [w for w in words if w not in COMMON_ADDR_STOPWORDS]


def _prepare_s1_query(name: str, addr: str):
    """Precompute all query tokens and subkeys for an S1 entity once."""
    core = get_core_name(name)
    nums, words = extract_addr_blocks(addr)
    tokens = core.split()
    rare = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

    st = " ".join(sorted(tokens)) if len(tokens) > 1 else None
    k2 = " ".join(tokens[:2]) if len(tokens) > 1 else None
    unsp = core.replace(" ", "") if len(core.replace(" ", "")) >= 3 else None
    fw = tokens[0] if (tokens and len(tokens[0]) >= 3 and tokens[0] not in STOPWORDS) else None

    ct = [t for t in tokens if t not in LEGAL_TOKENS]
    lw = ct[-1] if (ct and len(ct[-1]) >= 3 and ct[-1] not in STOPWORDS) else None

    valid_toks = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    bg = f"{rare[0]}_{rare[1]}" if len(rare) >= 2 else None
    pkey = _phonetic_key(name) if name else None
    ng4 = get_char_ngrams(core, 4)[:6] if core else []

    pfx3 = core[:3] if len(core) >= 3 else core
    pgeo = [f"{pfx3}_{num}" for num in nums]
    pword = [f"{pfx3}_{w}" for w in words[:3] if len(w) >= 4]

    twonum = f"{nums[0]}_{nums[1]}" if len(nums) >= 2 else None
    numwords = [f"{num}_{w}" for num in nums[:2] for w in words[:3]]
    pin_nums = [num for num in nums if len(num) >= 5]

    doorpin = None
    if nums:
        door = nums[0]
        for n in nums[1:]:
            if len(n) in (5, 6):
                doorpin = f"{door}_{n}"
                break

    return (name, core, st, k2, unsp, fw, lw, valid_toks, bg, pkey, ng4,
            pgeo, pword, twonum, numwords, pin_nums, doorpin)


def _build_indexes(t_names, t_addrs, chunk_offset: int,
                   MN: int, MFW: int, MT: int, MNW: int, MNG: int, MPH: int):
    """Build 16 inverted indexes for a target chunk. Returns tuple of dicts."""
    name_idx: dict[str, list[int]] = defaultdict(list)
    core_idx: dict[str, list[int]] = defaultdict(list)
    sort_idx: dict[str, list[int]] = defaultdict(list)
    unsp_idx: dict[str, list[int]] = defaultdict(list)
    ngram4_idx: dict[str, list[int]] = defaultdict(list)
    bigram_idx: dict[str, list[int]] = defaultdict(list)
    fw_idx: dict[str, list[int]] = defaultdict(list)
    lw_idx: dict[str, list[int]] = defaultdict(list)
    tok_idx: dict[str, list[int]] = defaultdict(list)
    rbig_idx: dict[str, list[int]] = defaultdict(list)
    phon_idx: dict[str, list[int]] = defaultdict(list)
    pgeo_idx: dict[str, list[int]] = defaultdict(list)
    pword_idx: dict[str, list[int]] = defaultdict(list)
    twonum_idx: dict[str, list[int]] = defaultdict(list)
    numword_idx: dict[str, list[int]] = defaultdict(list)
    addrnum_idx: dict[str, list[int]] = defaultdict(list)
    doorpin_idx: dict[str, list[int]] = defaultdict(list)

    for local_i, (name, addr) in enumerate(zip(t_names, t_addrs)):
        gi = chunk_offset + local_i   # global index into full target array
        core = get_core_name(name)
        nums, words = extract_addr_blocks(addr)
        tokens = core.split()
        rare = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

        if name:
            if len(name_idx[name]) < MN:
                name_idx[name].append(gi)
            if core != name and len(core) >= 3 and len(core_idx[core]) < MN:
                core_idx[core].append(gi)

            if len(tokens) > 1:
                st = " ".join(sorted(tokens))
                if len(sort_idx[st]) < 20:
                    sort_idx[st].append(gi)
                k2 = " ".join(tokens[:2])
                if len(bigram_idx[k2]) < 20:
                    bigram_idx[k2].append(gi)

            unsp = core.replace(" ", "")
            if len(unsp) >= 3 and len(unsp_idx[unsp]) < 20:
                unsp_idx[unsp].append(gi)

            if tokens:
                fw = tokens[0]
                if len(fw) >= 3 and fw not in STOPWORDS and len(fw_idx[fw]) <= MFW:
                    fw_idx[fw].append(gi)

            ct = [t for t in tokens if t not in LEGAL_TOKENS]
            if ct:
                lw = ct[-1]
                if len(lw) >= 3 and lw not in STOPWORDS and len(lw_idx[lw]) <= MFW:
                    lw_idx[lw].append(gi)

            for tok in tokens:
                if len(tok) >= 3 and tok not in STOPWORDS and len(tok_idx[tok]) <= MT:
                    tok_idx[tok].append(gi)

            if len(rare) >= 2:
                bg = f"{rare[0]}_{rare[1]}"
                if len(rbig_idx[bg]) < 20:
                    rbig_idx[bg].append(gi)

            pkey = _phonetic_key(name)
            if pkey and len(phon_idx[pkey]) < MPH:
                phon_idx[pkey].append(gi)

            for ng in get_char_ngrams(core, 4)[:6]:
                if len(ngram4_idx[ng]) < MNG:
                    ngram4_idx[ng].append(gi)

            pfx3 = core[:3] if len(core) >= 3 else core
            for num in nums:
                k = f"{pfx3}_{num}"
                if len(pgeo_idx[k]) < 15:
                    pgeo_idx[k].append(gi)
            for w in words[:3]:
                if len(w) >= 4:
                    kw = f"{pfx3}_{w}"
                    if len(pword_idx[kw]) < 15:
                        pword_idx[kw].append(gi)

        if len(nums) >= 2:
            k2n = f"{nums[0]}_{nums[1]}"
            if len(twonum_idx[k2n]) <= 25:
                twonum_idx[k2n].append(gi)

        if nums and words:
            for num in nums[:2]:
                for w in words[:3]:
                    k = f"{num}_{w}"
                    if len(numword_idx[k]) <= MNW:
                        numword_idx[k].append(gi)

        for num in nums:
            if len(num) >= 5 and len(addrnum_idx[num]) <= 30:
                addrnum_idx[num].append(gi)

        if nums:
            door = nums[0]
            for n in nums[1:]:
                if len(n) in (5, 6):
                    kdp = f"{door}_{n}"
                    if len(doorpin_idx[kdp]) < 25:
                        doorpin_idx[kdp].append(gi)

    return (name_idx, core_idx, sort_idx, unsp_idx, ngram4_idx, bigram_idx,
            fw_idx, lw_idx, tok_idx, rbig_idx, phon_idx,
            pgeo_idx, pword_idx, twonum_idx, numword_idx, addrnum_idx, doorpin_idx)


def _query_indexes(indexes, q, votes: Counter):
    """Ultra-fast index query using precomputed query tuple."""
    (name, core, st, k2, unsp, fw, lw, valid_toks, bg, pkey, ng4,
     pgeo, pword, twonum, numwords, pin_nums, doorpin) = q
    (name_idx, core_idx, sort_idx, unsp_idx, ngram4_idx, bigram_idx,
     fw_idx, lw_idx, tok_idx, rbig_idx, phon_idx,
     pgeo_idx, pword_idx, twonum_idx, numword_idx, addrnum_idx, doorpin_idx) = indexes

    if name:
        if name in name_idx:
            for gi in name_idx[name][:30]:
                votes[gi] += 8
        if core and core in core_idx:
            for gi in core_idx[core][:25]:
                votes[gi] += 6
        if st and st in sort_idx:
            for gi in sort_idx[st][:15]:
                votes[gi] += 4
        if k2 and k2 in bigram_idx:
            for gi in bigram_idx[k2][:15]:
                votes[gi] += 3
        if unsp and unsp in unsp_idx:
            for gi in unsp_idx[unsp][:12]:
                votes[gi] += 3
        if fw and fw in fw_idx:
            for gi in fw_idx[fw][:10]:
                votes[gi] += 2
        if lw and lw in lw_idx:
            for gi in lw_idx[lw][:10]:
                votes[gi] += 2
        for tok in valid_toks:
            if tok in tok_idx:
                for gi in tok_idx[tok][:8]:
                    votes[gi] += 1
        if bg and bg in rbig_idx:
            for gi in rbig_idx[bg][:12]:
                votes[gi] += 3
        if pkey and pkey in phon_idx:
            for gi in phon_idx[pkey][:12]:
                votes[gi] += 3
        if ng4:
            ng4_v: Counter[int] = Counter()
            for ng in ng4:
                if ng in ngram4_idx:
                    for gi in ngram4_idx[ng][:8]:
                        ng4_v[gi] += 1
            for gi, cnt in ng4_v.items():
                if cnt >= 2:
                    votes[gi] += min(cnt, 3)
        for k in pgeo:
            if k in pgeo_idx:
                for gi in pgeo_idx[k][:8]:
                    votes[gi] += 2
        for k in pword:
            if k in pword_idx:
                for gi in pword_idx[k][:8]:
                    votes[gi] += 2

    if twonum and twonum in twonum_idx:
        for gi in twonum_idx[twonum][:10]:
            votes[gi] += 4
    for k in numwords:
        if k in numword_idx:
            for gi in numword_idx[k][:6]:
                votes[gi] += 2
    for num in pin_nums:
        if num in addrnum_idx:
            for gi in addrnum_idx[num][:10]:
                votes[gi] += 2
    if doorpin and doorpin in doorpin_idx:
        for gi in doorpin_idx[doorpin][:8]:
            votes[gi] += 5


class MultiIndexBlocker:
    """
    Memory-safe blocking engine with chunked target processing
    and precomputed S1 query structures.
    Peak RAM: ~4-6 GB even for 4.1M India targets.
    """
    TARGET_CHUNK_SIZE = 1_500_000

    def __init__(
        self,
        max_total_candidates: int = 100,
        max_token_doc_freq: int = 200,
        max_num_word_freq: int = 80,
        max_first_word_freq: int = 120,
        max_name_freq: int = 200,
        max_ngram_freq: int = 30,
        max_phonetic_freq: int = 40,
    ):
        self.max_total_candidates = max_total_candidates
        self.max_token_doc_freq   = max_token_doc_freq
        self.max_num_word_freq    = max_num_word_freq
        self.max_first_word_freq  = max_first_word_freq
        self.max_name_freq        = max_name_freq
        self.max_ngram_freq       = max_ngram_freq
        self.max_phonetic_freq    = max_phonetic_freq

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> dict[str, list[str]]:
        import gc
        all_countries = set(df_s1["norm_country"].unique())
        candidate_map: dict[str, list[str]] = {sid: [] for sid in df_s1["entity_id"]}

        for country in sorted(all_countries):
            s1_mask = df_s1["norm_country"] == country
            s1_slice = df_s1[s1_mask]
            if len(s1_slice) == 0:
                continue

            t_s2 = df_s2[df_s2["norm_country"] == country]
            t_s3 = df_s3[df_s3["norm_country"] == country]
            target_len = len(t_s2) + len(t_s3)
            if target_len == 0:
                continue

            print(f"  [Blocking] Country: {country.upper()} | S1: {len(s1_slice):,} | Targets: {target_len:,}")

            t_ids   = np.concatenate([t_s2["entity_id"].values,   t_s3["entity_id"].values])
            t_names = np.concatenate([t_s2["norm_name"].values,    t_s3["norm_name"].values])
            t_addrs = np.concatenate([t_s2["norm_address"].values, t_s3["norm_address"].values])

            s1_ids   = s1_slice["entity_id"].values
            s1_names = s1_slice["norm_name"].values
            s1_addrs = s1_slice["norm_address"].values

            # Precompute S1 queries ONCE per country (massive speedup across chunks)
            s1_queries = [_prepare_s1_query(name, addr) for name, addr in zip(s1_names, s1_addrs)]
            s1_votes: dict[str, Counter] = {sid: Counter() for sid in s1_ids}

            chunk_size = self.TARGET_CHUNK_SIZE
            n_chunks = (target_len + chunk_size - 1) // chunk_size

            for chunk_idx in range(n_chunks):
                cs = chunk_idx * chunk_size
                ce = min(cs + chunk_size, target_len)
                chunk_names = t_names[cs:ce]
                chunk_addrs = t_addrs[cs:ce]

                if n_chunks > 1:
                    print(f"    Chunk {chunk_idx+1}/{n_chunks}: targets [{cs:,} – {ce:,}]")

                indexes = _build_indexes(
                    chunk_names, chunk_addrs, chunk_offset=cs,
                    MN=self.max_name_freq, MFW=self.max_first_word_freq,
                    MT=self.max_token_doc_freq, MNW=self.max_num_word_freq,
                    MNG=self.max_ngram_freq, MPH=self.max_phonetic_freq
                )

                # Query with precomputed S1 query structures
                for sid, q in zip(s1_ids, s1_queries):
                    _query_indexes(indexes, q, s1_votes[sid])

                del indexes, chunk_names, chunk_addrs
                gc.collect()

            for sid in s1_ids:
                top_gi = [gi for gi, _ in s1_votes[sid].most_common(self.max_total_candidates)]
                candidate_map[sid] = [t_ids[gi] for gi in top_gi]

            del t_ids, t_names, t_addrs, s1_queries, s1_votes, s1_slice, t_s2, t_s3
            gc.collect()

        return candidate_map
