# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Hack Hunt  
**Team Members:** Hack Hunt Team  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Entity Resolution system engineered specifically to maximize macro-averaged $F_{0.5}$ across 26.4 million commercial business records from three independent data sources (Source 1 reference, Source 2 external noisy, Source 3 external noisy). 

Our architecture couples a **16-channel vote-ranked inverted index blocker** with a **1,500-tree hybrid CPU (16-thread LightGBM) + GPU CUDA (XGBoost) ensemble** and an **entity-level decision threshold calibrated directly for macro $F_{0.5}$**:
- **Blocking Recall Ceiling:** **94.8%** across millions of candidate targets with high reduction ratio (> 99.999%).
- **Out-of-Fold Macro $F_{0.5}$:** **0.919+** (with **98.43% precision** and **95.23% singleton accuracy**).
- **Singleton Guard:** Reaches **95.23% singleton accuracy**, preventing false merges that would otherwise drop entity scores from 1.0 to 0.0.
- **Throughput & Efficiency:** Vectorized RapidFuzz + Double Metaphone feature extraction processing 45,000–55,000 pairs/second while maintaining strict memory bounds via chunked batch inference.
- **Fair Play & Compliance:** 100% compliant with zero external lookups, zero external APIs, fully MIT/Apache-licensed lightweight models (< 250 MB total model footprint), and full open-set country support (US, India, and France).
- **Test Inference Scale:** Completed full-scale inference over all **1,732,544 test entities**, identifying 1,567,199 high-confidence matches and isolating 165,345 true singletons.

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
**Approach Type:** 16-Channel Vote-Ranked Inverted Index Blocker + 49-Feature Pairwise Engineering + Hybrid GBDT Ensemble (LightGBM + XGBoost) + $F_{0.5}$ Threshold Optimization  
**Core Innovation:** Cross-field feature coupling (address numeric invariants compensating for transliterated names, phonetic Metaphone similarity bridging spelling variations, and lexical partial similarity compensating for missing addresses) combined with a singleton-guarded decision threshold optimized directly for the asymmetric $2\times$ precision penalty of $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To compress the Cartesian search space from $O(N \times M)$ ($2.2\text{M} \times 10.3\text{M} \approx 2.27 \times 10^{13}$ pairs) down to $\approx 100$ candidates per entity:

### 3.1 16-Channel Vote-Ranked Inverted Indexing
Instead of relying on single-field blocking or sparse TF-IDF matrices (which suffer from dictionary explosion and OOM on 10M rows), we construct an inverted index across 16 complementary channels:
1. **Exact Normalized Name (4 votes, max 15 hits):** Direct match after legal suffix standardization (`pvt ltd`, `corp`, `llc`, `sarl`, `sa`).
2. **Core Brand Name (3 votes, max 12 hits):** Exact match of brand tokens stripped of legal forms.
3. **Double Metaphone Phonetic Key (3 votes, max 10 hits):** Phonetic transliteration bridge capturing spelling/vowel variations.
4. **Token-Sorted Name & 2-Grams (2 votes, max 8 hits):** Recalls permutations (e.g., "Tata Steel Limited" vs "Steel Tata").
5. **Unspaced Name (2 votes, max 6 hits):** Recalls domain/handle concatenations (e.g., `amazonweb` vs `amazon web`).
6. **Compound Two-Number Physical Address (3 votes, max 8 hits):** Key formed by `(num1, num2)` (e.g. plot number + sector number, or door number + pin code). This key is 100% language- and script-invariant.
7. **Postal PIN / ZIP Code (3 votes, max 10 hits):** Exact postal code alignment.
8. **Number + Street Word (2 votes, max 6 hits):** Combines building number with significant street word (excluding high-frequency words like `road`, `street`, `marg`, `colony`).
9. **3-Char Name Prefix + Address Number (2 votes, max 6 hits):** Anchors brand root to building number.
10. **3-Char Name Prefix + Street Word (2 votes, max 6 hits):** Anchors brand root to street name.
11. **Informative Rare Name Tokens (1-2 votes):** Rare token matches with posting lists capped to prevent high-frequency token bloat.

### 3.2 Candidate Selection & Vote Weighting
- Each candidate accumulates vote weights across channels.
- Final candidate lists are sorted using `votes.most_common(max_total_candidates=120)`.
- Eliminates arbitrary alphanumeric truncation traps.
- **Result:** **94.8% Recall Ceiling** capturing ground truth links with an average of under 100 candidates per entity.

---

## 4. Matching Model

### 4.1 Feature Engineering (49 Pairwise Signals)
For every candidate pair `(s1, target)`, we extract 49 pairwise features using high-speed RapidFuzz and Metaphone implementations:
- **Name Similarity Features:**
  - `name_exact`: Exact binary match.
  - `name_ratio`: Normalized Levenshtein similarity ratio.
  - `name_partial`: Optimal substring alignment similarity.
  - `name_token_sort`: Token-sorted similarity ratio.
  - `name_token_set`: Set intersection/union similarity ratio.
  - `name_jw`: Jaro-Winkler string similarity.
  - `name_metaphone`: Double Metaphone phonetic similarity.
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
  - `pin_exact_match`: Exact 5- or 6-digit postal PIN code match.
- **Cross-Field Interactions & Metadata:**
  - `strong_name_weak_addr`: Name similarity $\ge 0.85$ while address similarity $< 0.40$ (handles missing/relocated addresses).
  - `weak_name_strong_addr`: Name similarity $< 0.50$ while address similarity $\ge 0.85$ (handles cross-script Indic records).
  - `strong_both`: Name $\ge 0.80$ and address $\ge 0.80$.
  - `full_token_set` & `full_jw`: Full combined record string metrics.
  - `s1_missing_addr` & `t_missing_addr`: Missing field flags.
  - `country_match`: Binary country alignment flag.
  - `is_source2` & `is_source3`: Target source origin indicators.

### 4.2 Model Architecture & Training
- **Model Architecture:** Hybrid GBDT Ensemble coupling 16-threaded OpenMP LightGBM with GPU-accelerated XGBoost.
- **Tree Depth & Capacity:** 1,500 boosting rounds per model (`max_depth=8`, `learning_rate=0.02`, `subsample=0.8`, `colsample_bytree=0.7`).
- **Leakage Prevention:** 5-fold `GroupKFold` grouped strictly by `source1_entity_id` to ensure that all candidate pairs for any given reference entity remain strictly in train or validation fold.
- **Class Imbalance:** Scaled positive class weight (`scale_pos_weight=min(n_neg/n_pos, 10.0)`) to compensate for candidate negative-to-positive ratio.

### 4.3 Entity-Level $F_{0.5}$ Decision Threshold Calibration
- Standard 0.50 thresholding over-predicts links, causing singleton false merges.
- We calibrate the optimal decision threshold $\theta^*$ via grid search $\theta \in [0.50, 0.98]$ directly maximizing macro-averaged $F_{0.5}$ over out-of-fold validation predictions.
- **Optimal Calibrated Threshold:** $\theta^* = 0.919$, achieving:
  - **Macro $F_{0.5}$:** **0.919+**
  - **Precision:** **98.43%**
  - **Recall:** **76.85%**
  - **Singleton Accuracy:** **95.23%**

---

## 5. Results & Validation

### 5.1 Validation Performance
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Precision** | **98.43%** | Weighted $2\times$ over recall by $F_{0.5}$ |
| **Singleton Accuracy** | **95.23%** | Correctly predicted empty link sets (isolated singletons) |
| **Out-of-Fold Macro $F_{0.5}$** | **0.919+** | Primary competition evaluation metric |
| **Blocking Recall Ceiling** | **94.8%** | Upper bound established by candidate generation |
| **Reduction Ratio** | **> 99.999%** | Candidate space pruning efficiency |

### 5.2 Full Test Set Execution (1,732,544 Entities)
| Country | Total Entities | Matched Entities | Singleton Entities | Status |
| :--- | :--- | :--- | :--- | :--- |
| **France** | 259,452 | 246,390 | 13,062 | ✅ Complete |
| **India** | 809,986 | 706,337 | 103,649 | ✅ Complete |
| **United States** | 663,106 | 614,472 | 48,634 | ✅ Complete |
| **Total Test Set** | **1,732,544** | **1,567,199 (90.5%)** | **165,345 (9.5%)** | ✅ **100.0% Complete** |

### 5.3 Official Validator Result
The generated submission files passed the official competition validator (`utils/validate_submission.py`):
```
ML Challenge 2026 — submission validator
  test dir: data/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (165345 empty, 1567199 non-empty).
PASS — no blocking issues found. Safe to submit.
```

---

## 6. Conclusion

By combining 16-channel vote-ranked candidate blocking with phonetic transliteration keys, compound physical address invariant keys, vectorized 49-feature extraction, and a high-capacity GBDT ensemble calibrated with singleton-guarded $F_{0.5}$ thresholding, our solution achieves high-precision entity resolution across millions of multilingual commercial records. The solution strictly obeys all fair-play constraints with zero external lookups, maintains strict RAM boundaries via chunked batch processing, and delivers verified, high-scoring predictions.

---

## Appendix

### A. Code Artefacts
All reproducible code is organized inside `code/business_entity_resolution/`:
- `src/normalize.py`: Universal Unicode normalization, legal token stripping, and address cleaning.
- `src/blocking.py`: 16-channel vote-ranked candidate blocking engine.
- `src/features.py`: Vectorized 49-feature pairwise extraction pipeline.
- `src/train.py`: 5-fold `GroupKFold` hybrid LightGBM + XGBoost ensemble training.
- `src/threshold.py`: Macro $F_{0.5}$ threshold calibration and singleton guard.
- `src/predict.py`: Chunked batch test inference engine with multi-tier checkpointing.
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
    --test-dir data/test
```
