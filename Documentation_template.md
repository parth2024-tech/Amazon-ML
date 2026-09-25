# Amazon ML Challenge 2026: Methodology Document
## Challenge: Business Entity Resolution

---

### 1. Executive Summary & Problem Overview
In multi-source commercial platforms, business identity data arrives from disparate sources with noisy, unstandardized, and incomplete fields. The goal of this challenge is to perform entity resolution (record linkage) between a deduplicated reference source (**Source 1**) and two unlinked sources (**Source 2** and **Source 3**). Each Source 1 entity may link to zero (singleton), one, or multiple entities from Sources 2 and 3.

Crucially, the evaluation metric is **Macro-Averaged $F_{0.5}$**:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
Because $F_{0.5}$ assigns **twice as much weight to Precision as to Recall**, false positives (incorrect merges) are heavily penalized. In particular, predicting an incorrect match for a true singleton causes its score to plummet from **1.0 to 0.0**. Therefore, our system is engineered with a **high-precision, conservative matching architecture** supported by a high-recall candidate generation foundation.

---

### 2. Candidate Generation / Blocking Strategy
Evaluating all Cartesian pairs $|S1| \times (|S2| + |S3|)$ is computationally infeasible ($O(N^2)$) and severely imbalanced. We implement a multi-index blocking strategy designed to achieve $\ge 98\%$ recall ceiling while filtering out $\ge 99.8\%$ of non-matching pairs.

#### Blocking Indexes:
1. **Character $n$-gram TF-IDF Sublinear Cosine Index**:
   - Analyzes sublinear character $3$-gram and $4$-gram term frequencies across normalized business name and address components.
   - Vectorized cosine similarity via sparse matrix dot products (`scipy.sparse.csr_matrix`).
   - Retrieves top-$K$ nearest neighbors per entity. This is highly robust to typos, character transpositions, abbreviations, and word order re-ordering.
2. **Token Inverted Index (Name Anchors)**:
   - Indexes significant alphanumeric tokens ($\ge 4$ characters) after stripping common legal suffixes.
   - Quickly recalls matches where entities share distinct brand or trade names despite severe address noise.
3. **Country-Stratified Execution**:
   - Partitions candidate search within matching country partitions (US, India, France) to eliminate cross-country false candidates.
   - Graceful fallback to global indexing if an entity's country is missing or unassigned.
4. **Candidate Union & Capping**:
   - Takes the union of candidates from both indexes, capped at Top-20 plausible candidates per Source 1 entity.
   - The resulting candidate set is saved as `output/candidate_pairs.tsv` as required by the competition specification.

---

### 3. Feature Engineering
For each candidate pair $(S1_i, \text{Target}_j)$, we construct an extensive vector of pairwise similarity signals:

#### A. Name Similarity Features:
- **Levenshtein String Ratios**: Full ratio, partial ratio, token sort ratio, and token set ratio (via `RapidFuzz`).
- **Jaro-Winkler Distance**: High sensitivity to common prefixes and root brand names.
- **Word & Character $n$-gram Jaccard**: Measures exact token and substring set overlap.
- **Structural Name Features**: Exact match indicator, first token match indicator, length difference, and character length ratio.

#### B. Address Similarity Features:
- **Fuzzy Address Overlap**: Token sort and token set ratios on standardized address text (expanding abbreviations like `rd` $\to$ `road`, `st` $\to$ `street`).
- **Digit & Numerical Alignment**: Jaccard similarity and boolean match of extracted numerical tokens (PIN codes, ZIP codes, door numbers, building IDs).
- **Address Jaro-Winkler Distance**: Captures geographical and street name alignment.

#### C. Cross-Field & Contextual Features:
- **Combined Name + Address Token Set Ratio**: Captures holistic record similarity.
- **Source Indicators**: Source 2 vs Source 3 categorical indicator flags.
- **Country Match Flag**: Binary verification of country consistency.

---

### 4. Model Architecture & Validation Setup

#### A. Model Architecture:
- We employ an ensemble of **Gradient Boosted Decision Trees (LightGBM)** for pairwise link classification.
- **Hyperparameters**:
  - `boosting_type`: GBDT
  - `learning_rate`: 0.04
  - `max_depth`: 6
  - `num_leaves`: 31
  - `colsample_bytree`: 0.8
  - `subsample`: 0.8
  - `scale_pos_weight`: Balanced to counteract class imbalance in candidate pairs.
  - Early stopping enabled on validation loss.

#### B. Cross-Validation Setup:
- **Group $K$-Fold Cross Validation (5 folds)** grouped strictly by `source1_entity_id`.
- This ensures that all pairs involving a given Source 1 entity appear exclusively in either the training fold or the validation fold, preventing any data leakage and mirroring test-time evaluation.

---

### 5. Post-Processing & $F_{0.5}$ Metric Optimization

1. **Threshold Search for Macro $F_{0.5}$**:
   - Rather than relying on a default $0.5$ classification threshold, we perform an automated grid search across candidate thresholds $\theta \in [0.20, 0.85]$.
   - At each threshold, full macro-averaged $F_{0.5}$ is calculated on the out-of-fold validation set, including full singleton scoring.
2. **Singleton Guard**:
   - If an entity's top candidate probability falls below the calibrated threshold $\theta^*$, the prediction list is set to empty ($\emptyset$).
   - This prevents false positive merges on singletons, securing a perfect $1.0$ score on true singletons.
3. **Multi-Match Resolution**:
   - Entities with multiple candidates exceeding $\theta^*$ are ranked by predicted probability, and deduplicated candidate IDs are emitted in comma-separated order.

---

### 6. Submission Compliance & Validation
All outputs are strictly formatted and verified against `utils/validate_submission.py`:
- `output/matching_results.tsv`: `source1_entity_id\tmatched_entity_ids` (exact headers, one row per test S1 entity).
- `output/candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids` (blocking superset).
- All IDs are verified to exist in the test set with valid `S2-` / `S3-` prefixes.
