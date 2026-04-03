#!/usr/bin/env python3
"""
Wrapper for C₁₃ 13-gram overlap measurement using the open-instruct
decontamination scripts (https://github.com/allenai/open-instruct).

Usage:
    # Index a training dataset
    python scripts/decontamination/measure_c13_wrapper.py index \
        --dataset allenai/Dolci-Think-SFT-7B \
        --messages_field messages \
        --query_filter role:user

    # Search for contamination
    python scripts/decontamination/measure_c13_wrapper.py search \
        --train_dataset_names allenai/Dolci-Think-SFT-7B \
        --dataset openai/gsm8k \
        --subset main \
        --field question \
        --ngram_size 13 \
        --output_dir results/contamination
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_index(args):
    """Run the official index.py script."""
    script_dir = Path(__file__).parent
    index_script = script_dir / "index.py"

    cmd = [
        sys.executable,
        str(index_script),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--index_type",
        args.index_type,
    ]

    if args.messages_field:
        cmd.extend(["--messages_field", args.messages_field])
    if args.query_filter:
        cmd.extend(["--query_filter", args.query_filter])
    if args.query_field:
        cmd.extend(["--query_field", args.query_field])
    if args.es_url:
        cmd.extend(["--es_url", args.es_url])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_search(args):
    """Run the official search.py script."""
    script_dir = Path(__file__).parent
    search_script = script_dir / "search.py"

    cmd = (
        [sys.executable, str(search_script), "--train_dataset_names"]
        + args.train_dataset_names
        + [
            "--output_dir",
            args.output_dir,
            "--index_type",
            args.index_type,
        ]
    )

    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.subset:
        cmd.extend(["--subset", args.subset])
    if args.split:
        cmd.extend(["--split", args.split])
    if args.field:
        cmd.extend(["--field"] + args.field)
    if args.ngram_size is not None:
        cmd.extend(["--ngram_size", str(args.ngram_size)])
    if args.search_size is not None:
        cmd.extend(["--search_size", str(args.search_size)])
    if args.es_url:
        cmd.extend(["--es_url", args.es_url])
    if args.decontaminate:
        cmd.append("--decontaminate")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Wrapper for AllenAI decontamination scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index subcommand
    index_parser = subparsers.add_parser("index", help="Index a training dataset")
    index_parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    index_parser.add_argument("--split", default="train", help="Dataset split")
    index_parser.add_argument("--messages_field", help="Field containing messages")
    index_parser.add_argument("--query_filter", default="role:user", help="Filter for messages")
    index_parser.add_argument("--query_field", default="content", help="Field to extract")
    index_parser.add_argument("--index_type", choices=["text", "vector"], default="text")
    index_parser.add_argument("--es_url", default="http://localhost:9200")

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search for contamination")
    search_parser.add_argument("--train_dataset_names", nargs="+", required=True)
    search_parser.add_argument("--dataset", help="Eval dataset name")
    search_parser.add_argument("--subset", help="Eval dataset subset")
    search_parser.add_argument("--split", default="test", help="Eval dataset split")
    search_parser.add_argument("--field", nargs="+", help="Fields to search")
    search_parser.add_argument("--ngram_size", type=int, help="N-gram size (e.g., 13 for C₁₃)")
    search_parser.add_argument("--search_size", type=int, default=100)
    search_parser.add_argument("--output_dir", required=True)
    search_parser.add_argument("--index_type", choices=["text", "vector"], default="text")
    search_parser.add_argument("--es_url", default="http://localhost:9200")
    search_parser.add_argument("--decontaminate", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Check ELASTIC_PASSWORD
    if "ELASTIC_PASSWORD" not in os.environ:
        print("Error: ELASTIC_PASSWORD environment variable not set")
        print("Set it with: export ELASTIC_PASSWORD=your_password")
        return 1

    if args.command == "index":
        return run_index(args)
    elif args.command == "search":
        return run_search(args)


if __name__ == "__main__":
    sys.exit(main())
