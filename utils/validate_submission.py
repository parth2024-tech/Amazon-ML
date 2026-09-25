#!/usr/bin/env python3
"""
validate_submission.py
Amazon ML Challenge 2026: Business Entity Resolution Submission Validator
Stdlib only, zero dependencies.
"""
import sys
import os
import argparse
import csv

def load_test_ids(test_dir):
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")
    
    if not os.path.exists(s1_file):
        raise FileNotFoundError(f"Missing {s1_file}")
    if not os.path.exists(s2_file):
        raise FileNotFoundError(f"Missing {s2_file}")
    if not os.path.exists(s3_file):
        raise FileNotFoundError(f"Missing {s3_file}")
        
    s1_ids = []
    with open(s1_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_ids.append(row["entity_id"].strip())
            
    s2_ids = set()
    with open(s2_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s2_ids.add(row["entity_id"].strip())

    s3_ids = set()
    with open(s3_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s3_ids.add(row["entity_id"].strip())

    return s1_ids, s2_ids, s3_ids

def validate(matching_path, candidate_path, test_dir):
    issues = []
    warnings = []
    
    # 1. Load test IDs
    try:
        s1_expected_list, s2_valid, s3_valid = load_test_ids(test_dir)
        s1_expected_set = set(s1_expected_list)
        valid_targets = s2_valid | s3_valid
    except Exception as e:
        issues.append(f"Could not load test source files from {test_dir}: {e!s}")
        return issues, warnings

    # 2. Check matching file
    if not os.path.exists(matching_path):
        issues.append(f"Matching file not found: {matching_path}")
        return issues, warnings

    matching_map = {}
    with open(matching_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            issues.append(f"Matching file is empty: {matching_path}")
            return issues, warnings
        
        header = lines[0].strip().split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            issues.append(f"Invalid header in matching file: {header}. Expected: ['source1_entity_id', 'matched_entity_ids']")

        for line_no, line in enumerate(lines[1:], start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 2:
                issues.append(f"Line {line_no} in matching file has {len(parts)} columns (expected 1 or 2 tab-separated).")
                continue
            s1_id = parts[0].strip()
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            
            if s1_id in matching_map:
                issues.append(f"Duplicate source1_entity_id in matching file: {s1_id} at line {line_no}")
            
            matched_list = [x.strip() for x in matched_str.split(",") if x.strip()] if matched_str else []
            # Check duplicate IDs in list
            if len(matched_list) != len(set(matched_list)):
                issues.append(f"Duplicate entity IDs in matching list for {s1_id} at line {line_no}")
            
            # Check source prefix & existence
            for mid in matched_list:
                if mid.startswith("S1-"):
                    issues.append(f"Self-match error: Source 1 entity {mid} found in matched_entity_ids for {s1_id}")
                elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                    issues.append(f"Invalid entity prefix for {mid} in matched_entity_ids for {s1_id}")
                elif mid not in valid_targets:
                    issues.append(f"Entity {mid} in matched_entity_ids does not exist in test set for {s1_id}")

            matching_map[s1_id] = set(matched_list)

    # Check S1 coverage
    missing_s1_in_matching = s1_expected_set - set(matching_map.keys())
    if missing_s1_in_matching:
        issues.append(f"{len(missing_s1_in_matching)} Source 1 test entities are missing from matching file. Example: {list(missing_s1_in_matching)[:3]}")
    extra_s1_in_matching = set(matching_map.keys()) - s1_expected_set
    if extra_s1_in_matching:
        issues.append(f"{len(extra_s1_in_matching)} unknown Source 1 entities found in matching file. Example: {list(extra_s1_in_matching)[:3]}")

    # 3. Check candidate file
    if not os.path.exists(candidate_path):
        issues.append(f"Candidate file not found: {candidate_path}")
        return issues, warnings

    candidate_map = {}
    with open(candidate_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            issues.append(f"Candidate file is empty: {candidate_path}")
            return issues, warnings

        header = lines[0].strip().split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            issues.append(f"Invalid header in candidate file: {header}. Expected: ['source1_entity_id', 'candidate_entity_ids']")

        for line_no, line in enumerate(lines[1:], start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > 2:
                issues.append(f"Line {line_no} in candidate file has {len(parts)} columns (expected 1 or 2 tab-separated).")
                continue
            s1_id = parts[0].strip()
            cand_str = parts[1].strip() if len(parts) > 1 else ""

            if s1_id in candidate_map:
                issues.append(f"Duplicate source1_entity_id in candidate file: {s1_id} at line {line_no}")

            cand_list = [x.strip() for x in cand_str.split(",") if x.strip()] if cand_str else []
            if len(cand_list) != len(set(cand_list)):
                issues.append(f"Duplicate entity IDs in candidate list for {s1_id} at line {line_no}")

            for cid in cand_list:
                if cid.startswith("S1-"):
                    issues.append(f"Source 1 entity {cid} found in candidate_entity_ids for {s1_id}")
                elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                    issues.append(f"Invalid entity prefix for {cid} in candidate_entity_ids for {s1_id}")
                elif cid not in valid_targets:
                    issues.append(f"Entity {cid} in candidate_entity_ids does not exist in test set for {s1_id}")

            candidate_map[s1_id] = set(cand_list)

    missing_s1_in_cand = s1_expected_set - set(candidate_map.keys())
    if missing_s1_in_cand:
        issues.append(f"{len(missing_s1_in_cand)} Source 1 test entities are missing from candidate file.")

    # 4. Check that matches are a subset of candidates
    for s1_id, matches in matching_map.items():
        candidates = candidate_map.get(s1_id, set())
        not_in_cand = matches - candidates
        if not_in_cand:
            warnings.append(f"Pipeline warning: {len(not_in_cand)} matched entities for {s1_id} were not in candidate_pairs: {list(not_in_cand)[:3]}")

    return issues, warnings

def main():
    parser = argparse.ArgumentParser(description="Validate submission files for Amazon ML Challenge 2026")
    parser.add_argument("--matching", required=True, help="Path to output/matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()

    issues, warnings = validate(args.matching, args.candidate, args.test_dir)

    for w in warnings:
        print(f"⚠️  WARNING: {w}")

    if issues:
        print(f"\n❌ Validation FAILED with {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)
    else:
        print("\n✅ PASS: Submission files are fully compliant and ready to submit!")
        sys.exit(0)

if __name__ == "__main__":
    main()
