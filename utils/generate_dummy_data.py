#!/usr/bin/env python3
"""
generate_dummy_data.py
Creates realistic synthetic TSV files for pipeline smoke-testing.
"""

import os
import pandas as pd

def generate(base_dir: str):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Train S1
    s1_train = [
        {"entity_id": "S1-00001", "business_name": "Apex Logistics Pvt Ltd", "business_address": "124 MG Road, Indiranagar, Bangalore 560038", "country": "India"},
        {"entity_id": "S1-00002", "business_name": "Blue Horizon Tech Solutions Corp", "business_address": "450 Lexington Ave, Suite 1200, New York, NY 10017", "country": "US"},
        {"entity_id": "S1-00003", "business_name": "Singleton Bakery", "business_address": "12 Pine Street, Seattle, WA 98101", "country": "US"},
        {"entity_id": "S1-00004", "business_name": "Kaveri Sweets & Snacks", "business_address": "Near Bus Stand, Mysore Road, Bangalore", "country": "India"}
    ]
    pd.DataFrame(s1_train).to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)

    # Train S2
    s2_train = [
        {"entity_id": "S2-00010", "business_name": "Apex Logistics Private Limited", "business_address": "124 M.G. Rd, Indira Nagar, Bengaluru", "country": "India"},
        {"entity_id": "S2-00020", "business_name": "Blue Horizon Technologies", "business_address": "450 Lexington Avenue, Ste 1200, NYC", "country": "US"},
        {"entity_id": "S2-00030", "business_name": "Random Unrelated Shop", "business_address": "77 Market St, San Francisco, CA", "country": "US"}
    ]
    pd.DataFrame(s2_train).to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)

    # Train S3
    s3_train = [
        {"entity_id": "S3-00100", "business_name": "Apex Logistics", "business_address": "MG Road, Indiranagar, Bangalore 560038", "country": "India"},
        {"entity_id": "S3-00200", "business_name": "Kaveri Sweets", "business_address": "Opposite KSRTC Bus Stand, Mysore Rd, Bangalore", "country": "India"}
    ]
    pd.DataFrame(s3_train).to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)

    # Train Ground Truth
    gt_train = [
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00010,S3-00100"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S2-00020"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": ""}, # Singleton!
        {"source1_entity_id": "S1-00004", "matched_entity_ids": "S3-00200"}
    ]
    pd.DataFrame(gt_train).to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    # Test S1 (includes France!)
    s1_test = [
        {"entity_id": "S1-10001", "business_name": "Le Bistro Parisien SAS", "business_address": "15 Rue de Rivoli, 75001 Paris", "country": "France"},
        {"entity_id": "S1-10002", "business_name": "Quantum AI Innovations Inc", "business_address": "100 Innovation Way, Boston, MA 02110", "country": "US"},
        {"entity_id": "S1-10003", "business_name": "Delhi Chai House", "business_address": "Connaught Place, Block B, New Delhi 110001", "country": "India"},
        {"entity_id": "S1-10004", "business_name": "Lone Star Hardware", "business_address": "880 South Congress Ave, Austin, TX 78704", "country": "US"}
    ]
    pd.DataFrame(s1_test).to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)

    # Test S2
    s2_test = [
        {"entity_id": "S2-10010", "business_name": "Le Bistro Parisien", "business_address": "15 Rue de Rivoli, Paris", "country": "France"},
        {"entity_id": "S2-10020", "business_name": "Quantum AI Innovations", "business_address": "100 Innovation Way, Boston, Massachusetts", "country": "US"},
        {"entity_id": "S2-10030", "business_name": "Completely Unrelated Cafe", "business_address": "Paris", "country": "France"}
    ]
    pd.DataFrame(s2_test).to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)

    # Test S3
    s3_test = [
        {"entity_id": "S3-10100", "business_name": "Delhi Chai", "business_address": "B-Block, Connaught Place, New Delhi", "country": "India"},
        {"entity_id": "S3-10200", "business_name": "Random Tech Labs", "business_address": "Bangalore", "country": "India"}
    ]
    pd.DataFrame(s3_test).to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)

    print(f"Generated synthetic dataset in {base_dir}")

if __name__ == "__main__":
    generate("/tmp/dummy_dataset")
