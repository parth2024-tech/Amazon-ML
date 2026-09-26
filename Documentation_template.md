# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Hack Hunt  
**Team Members:** Hack Hunt Team  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Entity Resolution system engineered specifically to maximize macro-averaged $F_{0.5}$ across 26.4 million commercial business records from three independent data sources (Source 1 reference, Source 2 external noisy, Source 3 external noisy). 

Our architecture couples a **multi-channel vote-ranked inverted index blocker** with a **5-fold GroupKFold LightGBM gradient boosted ensemble** and an **entity-level decision threshold calibrated directly for macro $F_{0.5}$**:
- **Blocking Recall Ceiling:** **84.91%** across 10.3 million candidate targets with an average of only **33.2 candidates per entity** (> 99.999% reduction ratio).
- **Out-of-Fold Macro $F_{0.5}$:** **0.8444** (with **93.89% precision** and **85.12% non-singleton $F_{0.5}$**).
- **Singleton Guard:** Reaches **73.36% singleton accuracy**, preventing false merges that would otherwise drop entity scores from 1.0 to 0.0.
- **Throughput & Efficiency:** Processes > 7,000 candidate pairs/second via vectorized RapidFuzz feature extraction while maintaining a strict < 3.5 GB RAM footprint through chunked batch inference.
- **Fair Play & Compliance:** 100% compliant with zero external lookups, zero external APIs, fully MIT-licensed lightweight model (< 200 MB memory footprint, 0 parameters above limits), and open-set country support (US, India, and France).

---

## 2. Methodology

### 2.1 Problem Analysis
Through comprehensive exploratory data analysis across 2.2 million reference entities in `train_source1.tsv` and 10.3 million target records in `train_source2.tsv` and `train_source3.tsv`, we identified key structural characteristics:
- **Strict Country Segmentation:** Ground-truth links exhibit exactly **0.0000% cross-country links** (US records strictly match US, India matches India, France matches France). Country partitioning prunes cross-country Cartesian comparisons without losing any true links.
- **Transliteration & Multi-Script Noise:** Indian records frequently have business names rendered in regional Indic scripts (Devanagari, Tamil, Telugu, Kannada) matching English reference entities. In these cases, English numeric tokens in the address (plot numbers, survey numbers, postal PIN codes) serve as cross-script invariants.
- **Asymmetric Field Reliability:** Target records frequently feature missing addresses (3.36% in Source 2, 3.42% in Source 3) or domain names masquerading as company names.
- **Asymmetric Penalty in Macro $F_{0.5}$:** The competition evaluation metric:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  weights precision $2\times$ over recall. For singletons (entities with 0 true matches), outputting an empty string scores 1.0, while a single false merge drops the entity score to 0.0. A conservative, high-confidence decision boundary is essential to maximize overall macro $F_{0.5}$.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Vote-Ranked Inverted Index Blocker + Pairwise GBDT Classification + Entity-Level $F_{0.5}$ Threshold Search  
**Core Innovation:** Cross-field feature coupling (address numeric invariants compensating for transliterated names, and lexical partial similarity compensating for missing addresses) combined with a singleton-guarded decision threshold optimized directly for the asymmetric $2\times$ precision penalty of $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To compress the Cartesian search space from $O(N \times M)$ ($2.2\text{M} \times 10.3\text{M} \approx 2.27 \times 10^{13}$ pairs) down to $\approx 33$ candidates per entity:

### 3.1 Multi-Channel Vote-Ranked Indexing
Instead of relying on single-field blocking or sparse TF-IDF matrices (which suffer from dictionary explosion and OOM on 10M rows), we construct an inverted index across 9 independent, complementary channels:
1. **Exact Normalized Name (4 votes, max 15 hits):** Direct match after legal suffix standardization (`pvt ltd`, `corp`, `llc`).
2. **Core Brand Name (3 votes, max 12 hits):** Exact match of brand tokens stripped of legal forms.
3. **Token-Sorted Name & 2-Grams (2 votes, max 8 hits):** Recalls permutations (e.g., "Tata Steel Limited" vs "Steel Tata").
4. **Unspaced Name (2 votes, max 6 hits):** Recalls domain/handle concatenations (e.g., `amazonweb` vs `amazon web`).
5. **Compound Two-Number Physical Address (3 votes, max 8 hits):** Key formed by `(num1, num2)` (e.g. plot number + sector number, or door number + pin code). This key is 100% language- and script-invariant.
6. **Number + Street Word (2 votes, max 6 hits):** Combines building number with significant street word (excluding high-frequency words like `road`, `street`, `marg`, `colony`).
7. **3-Char Name Prefix + Address Number (2 votes, max 6 hits):** Anchors brand root to building number.
8. **3-Char Name Prefix + Street Word (2 votes, max 6 hits):** Anchors brand root to street name.
9. **First Word & Informative Name Tokens (1-2 votes):** Rare token matches with posting lists capped to prevent high-frequency token bloat.

### 3.2 Candidate Selection & Vote Weighting
- Each candidate accumulates vote weights across channels.
- Final candidate lists are sorted using `votes.most_common(max_total_candidates=35)`.
- Eliminates arbitrary alphanumeric truncation traps (which previously dropped `S3-9*` IDs).
- **Result:** **84.91% Recall Ceiling** capturing 14,742 of 17,362 links with an average of only 33.2 candidates per entity.

---

## 4. Matching Model

### 4.1 Feature Engineering (32 Pairwise Signals)
For every candidate pair `(s1, target)`, we extract 32 pairwise features using high-speed RapidFuzz implementations:
- **Name Similarity Features:**
  - `name_exact`: Exact binary match.
  - `name_ratio`: Normalized Levenshtein similarity ratio.
  - `name_partial`: Optimal substring alignment similarity.
  - `name_token_sort`: Token-sorted similarity ratio.
  - `name_token_set`: Set intersection/union similarity ratio.
  - `name_jw`: Jaro-Winkler string similarity.
  - `name_word_jaccard`: Word token Jaccard overlap.
  - `name_token_overlap_count`: Absolute count of shared name tokens.
  - `name_ngram_jaccard`: Character 3-gram Jaccard similarity.
  - `first_word_match`: Binary match on initial token.
  - `name_len_diff` & `name_len_ratio`: Length discrepancy metrics.
- **Address & Geographic Features:**
  - `addr_exact`, `addr_ratio`, `addr_partial`, `addr_token_sort`, `addr_token_set`, `addr_jw`, `addr_word_jaccard`.
  - `geo_jaccard`: Jaccard similarity on numeric address tokens (PIN codes, plot/door numbers).
  - `has_common_geo`: Binary indicator for at least one shared numeric address anchor.
  - `contradictory_geo`: Binary indicator when both records contain numbers but share 0 numbers (strong signal against false merges).
- **Cross-Field Interactions & Metadata:**
  - `strong_name_weak_addr`: Name similarity $\ge 0.85$ while address similarity $< 0.40$ (handles missing/relocated addresses).
  - `weak_name_strong_addr`: Name similarity $< 0.50$ while address similarity $\ge 0.85$ (handles cross-script Indic records).
  - `strong_both`: Name $\ge 0.80$ and address $\ge 0.80$.
  - `full_token_set` & `full_jw`: Full combined record string metrics.
  - `s1_missing_addr` & `t_missing_addr`: Missing field flags.
  - `country_match`: Binary country alignment flag.
  - `is_source2` & `is_source3`: Target source origin indicators.

### 4.2 Model Architecture & Training
- **Model Type:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`) with `learning_rate=0.04`, `max_depth=6`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Leakage Prevention:** 5-fold `GroupKFold` grouped strictly by `source1_entity_id` to ensure that all candidate pairs for any given reference entity remain strictly in train or validation fold.
- **Class Imbalance:** Scaled positive class weight (`scale_pos_weight=min(n_neg/n_pos, 10.0)`) to compensate for candidate negative-to-positive ratio.

### 4.3 Entity-Level $F_{0.5}$ Decision Threshold Calibration
- Standard 0.50 thresholding over-predicts links, causing singleton false merges.
- We calibrate the optimal decision threshold $\theta^*$ via grid search $\theta \in [0.20, 0.85]$ directly maximizing macro-averaged $F_{0.5}$ over out-of-fold validation predictions.
- **Optimal Threshold:** $\theta^* = 0.850$, achieving:
  - **Macro $F_{0.5}$:** **0.8444**
  - **Micro Precision:** **93.89%**
  - **Micro Recall:** **77.27%**
  - **Singleton Accuracy:** **73.36%**

---

## 5. Results & Error Analysis

| Metric | Validation Score | Description |
| :--- | :--- | :--- |
| **Macro $F_{0.5}$** | **0.8444** | Primary competition evaluation metric |
| **Precision** | **0.9389** | Weighted $2\times$ over recall by $F_{0.5}$ |
| **Recall** | **0.7727** | Recall on candidate links |
| **Non-Singleton $F_{0.5}$** | **0.8512** | Score on entities with true matches |
| **Singleton Accuracy** | **73.36%** | Correctly predicted empty link sets |
| **Blocking Recall Ceiling** | **84.91%** | Upper bound established by candidate generation |
| **Reduction Ratio** | **99.9997%** | Candidate space pruning efficiency |

### Error Analysis:
1. **False Positives (Precision drops):** Businesses operating within identical commercial complexes, industrial estates, or tech parks sharing identical address numbers and generic brand prefixes (e.g. "Apex Healthcare" vs "Apex Diagnostics"). The `name_jw` and `name_token_sort` features, coupled with the high threshold $\theta^* = 0.850$, mitigate the vast majority of these merges.
2. **False Negatives (Missed links):** Extreme transliteration noise where Indian business records rendered in regional Indic scripts lack both recognizable English brand tokens and numeric building/PIN numbers in the address field.

---

## 6. Conclusion

By combining multi-channel vote-ranked candidate blocking with compound physical address invariant keys, vectorized RapidFuzz feature extraction, and singleton-guarded $F_{0.5}$ threshold calibration, our solution achieves high-precision entity resolution across millions of multilingual commercial records. The solution strictly obeys all fair-play constraints with zero external lookups, runs in < 3.5 GB RAM, and provides reproducible, validated predictions.

---

## Appendix

### A. Code Artefacts
All reproducible code is organized inside `code/business_entity_resolution/`:
- `src/normalize.py`: Unicode normalization, legal token stripping, and address cleaning.
- `src/blocking.py`: Multi-channel vote-ranked candidate blocking engine.
- `src/features.py`: Vectorized 32-feature pairwise extraction pipeline.
- `src/train.py`: 5-fold `GroupKFold` LightGBM model training.
- `src/threshold.py`: Macro $F_{0.5}$ threshold calibration and singleton guard.
- `src/predict.py`: Chunked batch test inference engine.
- `src/submission.py`: TSV export and competition zip packaging.
- `run.py`: Unified CLI entrypoint.

### B. Reproduction Instructions
```bash
# 1. Activate environment
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run full end-to-end pipeline
python3 run.py --mode full --team-name hack_hunt

# 3. Validate generated outputs
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### C. Validation Check
The generated submission files in `output/` passed the official competition validator (`utils/validate_submission.py`) with 0 errors:
`PASS — no blocking issues found. Safe to submit.`
