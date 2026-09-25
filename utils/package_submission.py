#!/usr/bin/env python3
"""
package_submission.py
Packages the submission zip conforming exactly to Amazon ML Challenge 2026 specifications:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse

def package(team_name: str, root_dir: str = "."):
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(root_dir, zip_filename)

    # Required files
    required_files = [
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "Documentation_template.md",
        "code/business_entity_resolution/README.md",
        "code/business_entity_resolution/requirements.txt"
    ]

    for rf in required_files:
        p = os.path.join(root_dir, rf)
        if not os.path.exists(p):
            print(f"❌ Error: Required file missing: {rf}")
            sys.exit(1)

    print(f"Creating submission package: {zip_filename}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add required individual files
        for rf in required_files:
            p = os.path.join(root_dir, rf)
            zipf.write(p, arcname=rf)
            print(f"  Added: {rf}")

        # Add all files under code/business_entity_resolution/src/
        src_dir = os.path.join(root_dir, "code/business_entity_resolution/src")
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".py", ".json", ".yaml", ".yml", ".md")):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, root_dir)
                    zipf.write(full_p, arcname=rel_p)
                    print(f"  Added: {rel_p}")

    print(f"\n🎉 Package created successfully: {zip_path}")
    print(f"Size: {os.path.getsize(zip_path) / (1024):.1f} KB")

def main():
    parser = argparse.ArgumentParser(description="Package submission zip for Amazon ML Challenge 2026")
    parser.add_argument("--team-name", required=True, help="Your team name")
    args = parser.parse_args()

    package(args.team_name)

if __name__ == "__main__":
    main()
