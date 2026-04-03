#!/usr/bin/env python3
"""
Compute IFEval constraint satisfaction quality from diversity generations.

Reads diversity_generations.jsonl files, loads IFEval constraint definitions
from HuggingFace, and verifies constraints on each of K=16 generations.

Computes:
  - strict_accuracy_at_1: first gen satisfies ALL constraints
  - loose_accuracy_at_1: fraction of constraints satisfied by first gen
  - strict_pass_at_16: any gen satisfies ALL constraints
  - consistency_at_16: fraction of 16 gens satisfying ALL constraints

Output CSV columns:
    model, task, strict_accuracy_at_1, loose_accuracy_at_1,
    strict_pass_at_16, consistency_at_16, n_prompts

Usage:
    python scripts/compute_ifeval_quality.py \\
        --input-dir runs/diversity/ --output-dir analysis/ifeval_quality/

    python scripts/compute_ifeval_quality.py \\
        --generations runs/extended/OLMo-3-7B/diversity_generations.jsonl \\
        --model OLMo-3-7B --output-dir analysis/ifeval_quality/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evaluation.ifeval_constraints import verify_constraints
from evaluation.lighteval_diversity_metrics import _strip_cot

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "ifeval_diversity": {
        "hf_repo": "google/IFEval",
        "hf_subset": "default",
        "split": "train",
        "prompt_field": "prompt",
        "instruction_id_list_field": "instruction_id_list",
        "kwargs_field": "kwargs",
    },
}


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


def _build_constraint_index(config: dict) -> Dict[str, dict]:
    """Build prompt prefix -> constraint definitions mapping."""
    from datasets import load_dataset

    ds = load_dataset(config["hf_repo"], config["hf_subset"], split=config["split"])
    index = {}
    for row in ds:
        prompt = str(row[config["prompt_field"]])
        key = prompt[:200].strip()
        instruction_id_list = row[config["instruction_id_list_field"]]
        kwargs_list = row[config["kwargs_field"]]
        # kwargs may be stored as JSON strings in some versions
        parsed_kwargs = []
        for kw in kwargs_list:
            if isinstance(kw, str):
                try:
                    parsed_kwargs.append(json.loads(kw))
                except (json.JSONDecodeError, ValueError):
                    parsed_kwargs.append({})
            elif isinstance(kw, dict):
                parsed_kwargs.append(kw)
            else:
                parsed_kwargs.append({})
        index[key] = {
            "instruction_id_list": list(instruction_id_list),
            "kwargs_list": parsed_kwargs,
        }
    return index


def _match_constraints(
    prompt: str, constraint_index: Dict[str, dict]
) -> Optional[dict]:
    """Match a generation prompt to its constraint definitions."""
    key = prompt[:200].strip()
    return constraint_index.get(key)


def evaluate_ifeval(
    records: List[Dict],
    config: dict,
) -> dict:
    """Evaluate IFEval constraint satisfaction from generation records."""
    print("  Loading IFEval constraint definitions...")
    constraint_index = _build_constraint_index(config)
    print(f"  Loaded {len(constraint_index)} prompt(s) with constraints")

    n_prompts = 0
    n_strict_first = 0
    sum_loose_first = 0.0
    n_strict_any = 0
    sum_consistency = 0.0

    for record in records:
        prompt = record.get("prompt", "")
        outputs = record.get("outputs", [])
        if not outputs:
            continue
        # Strip CoT reasoning from Think/RL models
        outputs = [_strip_cot(o) for o in outputs]

        constraints = _match_constraints(prompt, constraint_index)
        if constraints is None:
            continue
        n_prompts += 1

        instruction_id_list = constraints["instruction_id_list"]
        kwargs_list = constraints["kwargs_list"]

        # Evaluate first generation
        strict_first, loose_first = verify_constraints(
            outputs[0], instruction_id_list, kwargs_list
        )
        if strict_first:
            n_strict_first += 1
        sum_loose_first += loose_first

        # Evaluate all K generations
        n_strict_this = 0
        any_strict = False
        for out in outputs:
            strict, _ = verify_constraints(out, instruction_id_list, kwargs_list)
            if strict:
                n_strict_this += 1
                any_strict = True

        if any_strict:
            n_strict_any += 1
        sum_consistency += n_strict_this / len(outputs)

    result = {
        "n_prompts": n_prompts,
        "strict_accuracy_at_1": n_strict_first / max(n_prompts, 1),
        "loose_accuracy_at_1": sum_loose_first / max(n_prompts, 1),
        "strict_pass_at_16": n_strict_any / max(n_prompts, 1),
        "consistency_at_16": sum_consistency / max(n_prompts, 1),
    }

    return result


def process_model(
    gen_path: Path,
    model_name: str,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Process IFEval tasks for one model."""
    print(f"\nProcessing model: {model_name}")
    by_task = _load_generations(gen_path)
    ifeval_tasks = [t for t in sorted(by_task) if t in TASK_CONFIG]

    if not ifeval_tasks:
        print(f"  No IFEval tasks found (available: {list(by_task.keys())})")
        return

    print(f"  IFEval tasks: {ifeval_tasks}")

    if dry_run:
        for task_name in ifeval_tasks:
            n = len(by_task[task_name])
            print(f"  [dry-run] {task_name}: {n} prompts")
        return

    rows = []
    for task_name in ifeval_tasks:
        records = by_task[task_name]
        config = TASK_CONFIG[task_name]
        print(f"  Evaluating: {task_name} ({len(records)} prompts)")

        result = evaluate_ifeval(records, config)
        result["model"] = model_name
        result["task"] = task_name
        rows.append(result)

        print(
            f"  Results: strict@1={result['strict_accuracy_at_1']:.4f}, "
            f"loose@1={result['loose_accuracy_at_1']:.4f}, "
            f"strict_pass@16={result['strict_pass_at_16']:.4f}, "
            f"consistency@16={result['consistency_at_16']:.4f}"
        )

    if rows:
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            col_order = [
                "model", "task", "strict_accuracy_at_1", "loose_accuracy_at_1",
                "strict_pass_at_16", "consistency_at_16", "n_prompts",
            ]
            df = df[[c for c in col_order if c in df.columns]]

            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "ifeval_quality_results.csv"

            if out_path.exists():
                existing = pd.read_csv(out_path)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["model", "task"], keep="last")

            df.to_csv(out_path, index=False, float_format="%.6f")
            print(f"  Saved to {out_path}")
        except ImportError:
            out_path = output_dir / "ifeval_quality_results.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  Saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute IFEval constraint satisfaction from diversity outputs.",
    )
    parser.add_argument(
        "--generations", type=Path, help="Path to diversity_generations.jsonl for one model"
    )
    parser.add_argument("--model", help="Model name (required with --generations)")
    parser.add_argument(
        "--input-dir", type=Path, help="Root diversity runs directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ifeval_quality"),
        help="Output directory (default: analysis/ifeval_quality/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would be evaluated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generations:
        if not args.model:
            raise SystemExit("--model is required when using --generations")
        if not args.generations.exists():
            raise SystemExit(f"File not found: {args.generations}")
        process_model(args.generations, args.model, args.output_dir, args.dry_run)

    elif args.input_dir:
        if not args.input_dir.is_dir():
            raise SystemExit(f"Directory not found: {args.input_dir}")

        # Search in multiple run directories (diversity, extended, benchmark_diversity)
        gen_files = []
        for subdir in ["diversity", "extended", "benchmark_diversity"]:
            search_dir = args.input_dir / subdir if (args.input_dir / subdir).is_dir() else args.input_dir
            gen_files.extend(sorted(search_dir.glob("*/diversity_generations.jsonl")))

        if not gen_files:
            gen_files = sorted(args.input_dir.glob("*/diversity_generations.jsonl"))

        if not gen_files:
            raise SystemExit(f"No diversity_generations.jsonl found in {args.input_dir}")

        # Deduplicate by path
        gen_files = list(dict.fromkeys(gen_files))
        print(f"Found {len(gen_files)} model(s) with generation files")
        for gf in gen_files:
            model_name = gf.parent.name
            process_model(gf, model_name, args.output_dir, args.dry_run)
    else:
        raise SystemExit("Provide either --generations or --input-dir")


if __name__ == "__main__":
    main()
