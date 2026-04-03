#!/usr/bin/env python3
"""
Compute pass@k quality metrics from HumanEval/MBPP diversity generations.

Reads diversity_generations.jsonl files, executes each of K=16 generations
against test cases from HuggingFace, and computes unbiased pass@k estimates.

Uses the Codex estimator: pass@k = 1 - C(n-c, k) / C(n, k)

Output CSV columns:
    model, task, pass_at_1, pass_at_5, pass_at_10, pass_at_16,
    n_prompts, n_correct_total

Usage:
    # All models
    python scripts/compute_code_quality.py \\
        --input-dir runs/diversity/ --output-dir analysis/code_quality/

    # Single model
    python scripts/compute_code_quality.py \\
        --generations runs/diversity/OLMo-3-7B/diversity_generations.jsonl \\
        --model OLMo-3-7B --output-dir analysis/code_quality/

    # Dry run
    python scripts/compute_code_quality.py --input-dir runs/diversity/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# Add src to path for local imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evaluation.code_execution import (
    execute_code_with_tests,
    prepare_humaneval_execution,
    prepare_mbpp_execution,
)
from evaluation.lighteval_diversity_metrics import _strip_cot

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "humaneval_diversity": {
        "hf_repo": "openai_humaneval",
        "hf_subset": "openai_humaneval",
        "split": "test",
        "prompt_field": "prompt",
        "test_field": "test",
        "entry_point_field": "entry_point",
    },
    "mbpp_diversity": {
        "hf_repo": "mbpp",
        "hf_subset": "full",
        "split": "test",
        "text_field": "text",
        "test_list_field": "test_list",
    },
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_generations(path: Path) -> Dict[str, List[Dict]]:
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    duplicate_counts: Dict[str, int] = defaultdict(int)
    for record in _iter_jsonl(path):
        task = record.get("task", "unknown").split("|")[0]
        input_id = str(record.get("input_id", "")).strip() or str(record.get("prompt", ""))[:200]
        if input_id in seen[task]:
            duplicate_counts[task] += 1
            continue
        seen[task].add(input_id)
        by_task[task].append(record)
    for task, count in sorted(duplicate_counts.items()):
        print(
            f"  Deduplicated {count} duplicate generation record(s) for task {task}",
            file=sys.stderr,
        )
    return dict(by_task)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex paper.

    pass@k = 1 - C(n-c, k) / C(n, k)

    where n = total samples, c = correct samples, k = selection size.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _build_humaneval_index(config: dict) -> Dict[str, dict]:
    """Build prompt prefix -> test case mapping for HumanEval."""
    from datasets import load_dataset

    ds = load_dataset(config["hf_repo"], config["hf_subset"], split=config["split"])
    index = {}
    for row in ds:
        prompt = str(row[config["prompt_field"]])
        key = prompt[:200].strip()
        index[key] = {
            "prompt": prompt,
            "test": str(row[config["test_field"]]),
            "entry_point": str(row[config["entry_point_field"]]),
        }
    return index


def _build_mbpp_index(config: dict) -> Dict[str, dict]:
    """Build prompt prefix -> test case mapping for MBPP."""
    from datasets import load_dataset

    ds = load_dataset(config["hf_repo"], config["hf_subset"], split=config["split"])
    index = {}
    for row in ds:
        text = str(row[config["text_field"]])
        # MBPP prompt format: '"""\n{text}\n"""\n'
        prompt_key = f'"""\n{text}\n"""'[:200].strip()
        index[prompt_key] = {
            "text": text,
            "test_list": list(row[config["test_list_field"]]),
        }
    return index


def _match_prompt(prompt: str, index: Dict[str, dict]) -> Optional[dict]:
    """Match a generation prompt to a test case via prefix."""
    key = prompt[:200].strip()
    return index.get(key)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_humaneval(
    records: List[Dict],
    config: dict,
    timeout: float = 5.0,
) -> dict:
    """Evaluate HumanEval pass@k from generation records."""
    print("  Loading HumanEval test cases...")
    index = _build_humaneval_index(config)
    print(f"  Loaded {len(index)} test case(s)")

    k_values = [1, 5, 10, 16]
    all_results = []
    n_matched = 0
    n_missing = 0

    for record in records:
        prompt = record.get("prompt", "")
        outputs = record.get("outputs", [])
        if not outputs:
            continue
        # Strip CoT reasoning from Think/RL models before execution
        outputs = [_strip_cot(o) for o in outputs]

        test_case = _match_prompt(prompt, index)
        if test_case is None:
            n_missing += 1
            continue
        n_matched += 1

        n = len(outputs)
        c = 0
        for completion in outputs:
            code = prepare_humaneval_execution(
                prompt=test_case["prompt"],
                completion=completion,
                test_code=test_case["test"],
                entry_point=test_case["entry_point"],
            )
            passed, _ = execute_code_with_tests(code, timeout=timeout)
            if passed:
                c += 1

        all_results.append({"n": n, "c": c})

    if n_missing:
        print(f"  Warning: {n_missing} prompt(s) could not be matched")

    # Aggregate pass@k
    result = {"n_prompts": n_matched, "n_correct_total": sum(r["c"] for r in all_results)}
    for k in k_values:
        if all_results:
            scores = [compute_pass_at_k(r["n"], r["c"], min(k, r["n"])) for r in all_results]
            result[f"pass_at_{k}"] = sum(scores) / len(scores)
        else:
            result[f"pass_at_{k}"] = 0.0

    return result


def evaluate_mbpp(
    records: List[Dict],
    config: dict,
    timeout: float = 5.0,
) -> dict:
    """Evaluate MBPP pass@k from generation records."""
    print("  Loading MBPP test cases...")
    index = _build_mbpp_index(config)
    print(f"  Loaded {len(index)} test case(s)")

    k_values = [1, 5, 10, 16]
    all_results = []
    n_matched = 0
    n_missing = 0

    for record in records:
        prompt = record.get("prompt", "")
        outputs = record.get("outputs", [])
        if not outputs:
            continue
        # Strip CoT reasoning from Think/RL models before execution
        outputs = [_strip_cot(o) for o in outputs]

        test_case = _match_prompt(prompt, index)
        if test_case is None:
            n_missing += 1
            continue
        n_matched += 1

        n = len(outputs)
        c = 0
        for generated_code in outputs:
            code = prepare_mbpp_execution(generated_code, test_case["test_list"])
            passed, _ = execute_code_with_tests(code, timeout=timeout)
            if passed:
                c += 1

        all_results.append({"n": n, "c": c})

    if n_missing:
        print(f"  Warning: {n_missing} prompt(s) could not be matched")

    result = {"n_prompts": n_matched, "n_correct_total": sum(r["c"] for r in all_results)}
    for k in k_values:
        if all_results:
            scores = [compute_pass_at_k(r["n"], r["c"], min(k, r["n"])) for r in all_results]
            result[f"pass_at_{k}"] = sum(scores) / len(scores)
        else:
            result[f"pass_at_{k}"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_model(
    gen_path: Path,
    model_name: str,
    output_dir: Path,
    dry_run: bool = False,
    timeout: float = 5.0,
) -> None:
    """Process code generation tasks for one model."""
    print(f"\nProcessing model: {model_name}")
    by_task = _load_generations(gen_path)
    code_tasks = [t for t in sorted(by_task) if t in TASK_CONFIG]

    if not code_tasks:
        print(f"  No code tasks found (available: {list(by_task.keys())})")
        return

    print(f"  Code tasks: {code_tasks}")

    if dry_run:
        for task_name in code_tasks:
            n = len(by_task[task_name])
            print(f"  [dry-run] {task_name}: {n} prompts")
        return

    rows = []
    for task_name in code_tasks:
        records = by_task[task_name]
        config = TASK_CONFIG[task_name]
        print(f"  Evaluating: {task_name} ({len(records)} prompts)")

        if task_name == "humaneval_diversity":
            result = evaluate_humaneval(records, config, timeout=timeout)
        elif task_name == "mbpp_diversity":
            result = evaluate_mbpp(records, config, timeout=timeout)
        else:
            continue

        result["model"] = model_name
        result["task"] = task_name
        rows.append(result)

        print(
            f"  Results: pass@1={result['pass_at_1']:.4f}, "
            f"pass@10={result['pass_at_10']:.4f}, "
            f"pass@16={result['pass_at_16']:.4f}"
        )

    if rows:
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            col_order = [
                "model", "task", "pass_at_1", "pass_at_5", "pass_at_10",
                "pass_at_16", "n_prompts", "n_correct_total",
            ]
            df = df[[c for c in col_order if c in df.columns]]

            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "code_quality_results.csv"

            # Append if file exists
            if out_path.exists():
                existing = pd.read_csv(out_path)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["model", "task"], keep="last")

            df.to_csv(out_path, index=False, float_format="%.6f")
            print(f"  Saved to {out_path}")
        except ImportError:
            # Fallback: write as JSON
            out_path = output_dir / "code_quality_results.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  Saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pass@k quality from code generation diversity outputs.",
    )
    parser.add_argument(
        "--generations", type=Path, help="Path to diversity_generations.jsonl for one model"
    )
    parser.add_argument("--model", help="Model name (required with --generations)")
    parser.add_argument(
        "--input-dir", type=Path, help="Root diversity runs directory (e.g. runs/diversity/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/code_quality"),
        help="Output directory (default: analysis/code_quality/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would be evaluated")
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="Execution timeout per test (default: 5s)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generations:
        if not args.model:
            raise SystemExit("--model is required when using --generations")
        if not args.generations.exists():
            raise SystemExit(f"File not found: {args.generations}")
        process_model(
            args.generations, args.model, args.output_dir,
            args.dry_run, args.timeout,
        )

    elif args.input_dir:
        if not args.input_dir.is_dir():
            raise SystemExit(f"Directory not found: {args.input_dir}")

        gen_files = sorted(args.input_dir.glob("*/diversity_generations.jsonl"))
        if not gen_files:
            raise SystemExit(f"No diversity_generations.jsonl files found in {args.input_dir}")

        print(f"Found {len(gen_files)} model(s) with generation files")
        for gf in gen_files:
            model_name = gf.parent.name
            process_model(gf, model_name, args.output_dir, args.dry_run, args.timeout)
    else:
        raise SystemExit("Provide either --generations or --input-dir")


if __name__ == "__main__":
    main()
