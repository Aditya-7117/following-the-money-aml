"""
Link every record in HI-Small_Patterns.txt to its transaction_id.

Run from the repo folder:
    python scripts/match_patterns.py

Reads:  data/raw/HI-Small_Patterns.txt and data/raw/HI-Small_Trans.csv
Saves:  artifacts/evaluation/pattern_map.csv       (matched records)
        artifacts/evaluation/pattern_coverage.json (counts for the report)
"""

import json
import sys
from pathlib import Path

# let Python find the src folder
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.patterns import coverage_summary, match_patterns, read_patterns

RAW = ROOT / "data" / "raw"
OUT = ROOT / "artifacts" / "evaluation"

patterns = read_patterns(RAW / "HI-Small_Patterns.txt")
print(f"Read {len(patterns)} records from {patterns['attempt_id'].nunique()} attempts")

print("Matching to the transactions CSV (this takes a minute or two)...")
result, n_laundering = match_patterns(patterns, RAW / "HI-Small_Trans.csv")
summary = coverage_summary(result, n_laundering)

# save the matched records and the counts
OUT.mkdir(parents=True, exist_ok=True)
matched = result[result["status"] == "matched"]
matched[["transaction_id", "attempt_id", "pattern_type"]].to_csv(OUT / "pattern_map.csv", index=False)

with open(OUT / "pattern_coverage.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
