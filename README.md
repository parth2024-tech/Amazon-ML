# Amazon ML Challenge 2026: Business Entity Resolution

Complete, production-ready machine learning solution for cross-source noisy business entity matching targeting macro-averaged **$F_{0.5}$**.

---

## 📁 Project Structure

```
amazon-ml-challenge/
├── data/
│   ├── train/                 # Place train_source1/2/3.tsv and train_ground_truth.tsv here
│   └── test/                  # Place test_source1/2/3.tsv here
├── src/
│   ├── config.py              # Centralized configuration, hyperparams, and paths
│   ├── normalize.py           # Universal Unicode, legal suffix, & address cleaning
│   ├── load_data.py           # Robust TSV loaders & schema verification
│   ├── eda.py                 # Comprehensive distribution & overlap analysis
│   ├── blocking.py            # High-recall multi-index candidate blocker
│   ├── features.py            # RapidFuzz pairwise & cross-field feature extraction
│   ├── train.py               # Supervised LightGBM classifier with GroupKFold
│   ├── evaluate.py            # Macro F_0.5 evaluation metric & diagnostics
│   ├── threshold.py           # F_0.5 threshold calibration & singleton guard
│   ├── predict.py             # Test inference pipeline
│   └── submission.py          # Output formatting, validation, & zip packager
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis notebook
│   ├── 02_blocking.ipynb      # Blocking recall benchmarking notebook
│   └── 03_modeling.ipynb      # Supervised modeling & threshold optimization notebook
├── outputs/
│   ├── candidate_pairs.tsv    # Blocking candidate set
│   └── matching_results.tsv   # Final entity matches (leaderboard submission)
├── utils/
│   ├── validate_submission.py # Official submission format validator
│   ├── metrics.py             # Standalone F_0.5 score calculator
│   └── generate_dummy_data.py # Synthetic data generator for smoke testing
├── run.py                     # Master CLI runner
├── methodology.md             # Detailed engineering & methodology report
├── requirements.txt           # Pinned environment dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Place Your Data

Download the competition files and place them in `data/train/` and `data/test/`:
- `data/train/train_source1.tsv`
- `data/train/train_source2.tsv`
- `data/train/train_source3.tsv`
- `data/train/train_ground_truth.tsv`
- `data/test/test_source1.tsv`
- `data/test/test_source2.tsv`
- `data/test/test_source3.tsv`

*(Note: `dataset/train` and `dataset/test` are also automatically recognized).*

---

## 🚀 Execution Commands

### Run Exploratory Data Analysis
```bash
python run.py --mode eda
```

### Benchmark Candidate Blocking Recall
```bash
python run.py --mode blocking
```

### Run End-to-End Pipeline (Train + Tune + Predict + Validate + Package)
```bash
python run.py --mode full --team-name <your_team_name>
```

### Validate Outputs Manually
```bash
python utils/validate_submission.py \
    --matching outputs/matching_results.tsv \
    --candidate outputs/candidate_pairs.tsv \
    --test-dir data/test
```

### Package Official Submission Zip
```bash
python run.py --mode package --team-name <your_team_name>
```

---

## 🎯 Key Design Highlights

1. **Precision-First Architecture ($F_{0.5}$)**:
   - False positive merges are penalized twice as heavily as false negatives.
   - True singletons score **1.0** when correctly predicted empty, but drop to **0.0** on any false merge.
   - The threshold optimizer explicitly guards singletons by rejecting ambiguous candidates.

2. **Zero-Leakage GroupKFold**:
   - Validation folds are partitioned strictly at the Source 1 entity level. No pairs from the same business entity are split across train and validation.

3. **Open Country Generalization**:
   - Universal unicode NFKD diacritic removal handles France, India, US, and any international character set without closed-set hardcoding.
