import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    extract_geographic_tokens
)
from src.evaluate import compute_entity_f05, evaluate_predictions

def test_normalization():
    # Legal suffix removal
    assert normalize_business_name("Acme Corp Pvt Ltd") == "acme"
    assert normalize_business_name("Global Tech Solutions LLC") == "global tech"
    
    # Address abbreviations
    assert normalize_address("123 MG Rd, Opp SBI, 5th Fl") == "123 mg road opposite sbi 5th floor"
    
    # Digits extraction
    assert extract_geographic_tokens("124 MG Road, Bangalore 560038") == ["124", "560038"]
    
    # Country normalization
    assert normalize_country("United States of America") == "us"
    assert normalize_country("FRANCE") == "france"
    assert normalize_country("India") == "india"

def test_metric_f05_official_example():
    # Official example from challenge document:
    # S1-00001 predicts [S2-00047, S2-00193, S3-00812]
    # Ground truth is [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    gt = {"S2-00047", "S3-00812"}
    score = compute_entity_f05(pred, gt)
    assert round(score, 3) == 0.714

def test_singleton_scoring():
    # Empty pred for empty gt gives 1.0
    assert compute_entity_f05(set(), set()) == 1.0
    # Any pred for empty gt gives 0.0
    assert compute_entity_f05({"S2-00001"}, set()) == 0.0
    # Empty pred for non-empty gt gives 0.0
    assert compute_entity_f05(set(), {"S2-00001"}) == 0.0

def test_evaluate_predictions_macro():
    gt = {
        "S1-001": {"S2-010"},
        "S1-002": set() # Singleton
    }
    pred_perfect = {
        "S1-001": {"S2-010"},
        "S1-002": set()
    }
    metrics = evaluate_predictions(pred_perfect, gt)
    assert metrics["macro_f05"] == 1.0
    assert metrics["singleton_accuracy"] == 1.0
