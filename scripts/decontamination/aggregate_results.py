#!/usr/bin/env python3
"""Aggregate decontamination results from per-pair JSONL files into a summary table."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("analysis/decontamination")

    # Collect all JSONL files
    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return

    # Parse results: {eval_dataset: {train_index: mean_score}}
    results = defaultdict(dict)
    for f in jsonl_files:
        # Filename format: <train_index>_<eval_dataset>.jsonl
        # e.g., allenai_dolci-think-sft-7b_text_gsm8k.jsonl
        name = f.stem  # without .jsonl
        # The train index ends with _text, so split on that
        parts = name.split("_text_")
        if len(parts) != 2:
            print(f"  Skipping {f.name}: unexpected format")
            continue
        train_index = parts[0] + "_text"
        eval_dataset = parts[1]

        # Read JSONL and compute mean match score
        scores = []
        with open(f) as fh:
            for line in fh:
                data = json.loads(line)
                if "score" in data:
                    scores.append(data["score"])
                elif "match_score" in data:
                    scores.append(data["match_score"])

        if scores:
            mean_score = sum(scores) / len(scores)
            results[eval_dataset][train_index] = (mean_score, len(scores))

    if not results:
        print("No results found")
        return

    # Get all train indices
    all_train = sorted(set(t for d in results.values() for t in d))
    all_eval = sorted(results.keys())

    # Print summary table
    print("=" * 100)
    print("C₁₃ Decontamination Results: Mean 13-gram Overlap Scores")
    print("=" * 100)
    print()

    # Header
    header = f"{'Eval Dataset':<25}"
    for t in all_train:
        short = t.replace("allenai_dolci-", "").replace("_text", "")
        header += f"  {short:>18}"
    print(header)
    print("-" * len(header))

    # Data rows
    for eval_ds in all_eval:
        row = f"{eval_ds:<25}"
        for t in all_train:
            if t in results[eval_ds]:
                score, n = results[eval_ds][t]
                row += f"  {score:>18.6f}"
            else:
                row += f"  {'—':>18}"
        print(row)

    print()
    print("Interpretation: scores < 0.05 indicate negligible contamination")
    print()

    # Details
    print("Details (N = number of eval instances):")
    for eval_ds in all_eval:
        for t in all_train:
            if t in results[eval_ds]:
                score, n = results[eval_ds][t]
                print(f"  {eval_ds} vs {t}: mean={score:.6f}, N={n}")


if __name__ == "__main__":
    main()
