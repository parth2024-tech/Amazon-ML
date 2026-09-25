#!/usr/bin/env python3
"""
submission.py
Exports formatted TSV files, validates against official competition rules,
and builds the submission zip archive.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

from .config import PROJECT_ROOT, OUTPUT_DIR, LEGACY_OUTPUT_DIR

def export_tsv(
    data_dict: dict[str, list[str]],
    file_path: Path,
    column_name: str
):
    """Writes TSV with exact column header and comma-separated IDs."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{column_name}\n")
        for s1_id in sorted(data_dict.keys()):
            matches = data_dict[s1_id]
            match_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{match_str}\n")

    print(f"Exported {len(data_dict)} rows to {file_path.resolve()}")

def save_and_validate_submission(
    candidate_pairs: dict[str, list[str]],
    final_matches: dict[str, list[str]],
    test_dir: Path
) -> bool:
    """
    Saves outputs to both outputs/ and output/, then runs utils/validate_submission.py.
    """
    # Write to outputs/
    cand_out = OUTPUT_DIR / "candidate_pairs.tsv"
    match_out = OUTPUT_DIR / "matching_results.tsv"
    export_tsv(candidate_pairs, cand_out, "candidate_entity_ids")
    export_tsv(final_matches, match_out, "matched_entity_ids")

    # Mirror to output/ for official competition zip structure
    legacy_cand = LEGACY_OUTPUT_DIR / "candidate_pairs.tsv"
    legacy_match = LEGACY_OUTPUT_DIR / "matching_results.tsv"
    export_tsv(candidate_pairs, legacy_cand, "candidate_entity_ids")
    export_tsv(final_matches, legacy_match, "matched_entity_ids")

    # Run validator
    validator_path = PROJECT_ROOT / "utils" / "validate_submission.py"
    if validator_path.exists():
        print("\nRunning submission validator...")
        cmd = [
            sys.executable, str(validator_path),
            "--matching", str(match_out),
            "--candidate", str(cand_out),
            "--test-dir", str(test_dir)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        return res.returncode == 0
    return True

def package_submission_zip(team_name: str, output_zip_path: Path | None = None) -> Path:
    """Packages the official challenge zip archive."""
    zip_path = output_zip_path or (PROJECT_ROOT / f"{team_name}_submission.zip")
    
    required = [
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "Documentation_template.md",
        "code/business_entity_resolution/README.md",
        "code/business_entity_resolution/requirements.txt"
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rf in required:
            p = PROJECT_ROOT / rf
            if p.exists():
                z.write(p, arcname=rf)
            else:
                print(f"Warning: {rf} not found during packaging.")

        src_dir = PROJECT_ROOT / "src"
        if src_dir.exists():
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.endswith((".py", ".md", ".json")):
                        full_p = Path(root) / f
                        rel_p = Path("code/business_entity_resolution/src") / full_p.relative_to(src_dir)
                        z.write(full_p, arcname=str(rel_p))

    print(f"Created submission archive: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return zip_path
