#!/usr/bin/env python3
"""
Compute accuracy and majority vote quality from GSM8K diversity generations.

Reads diversity_generations.jsonl files, extracts numeric answers from each
of K=16 chain-of-thought generations, and computes:
  - accuracy_at_1: first generation correct
  - majority_vote_at_16: self-consistency (Wang et al., 2023)
  - pass_at_16: any generation correct

Output CSV columns:
    model, task, accuracy_at_1, majority_vote_at_16, pass_at_16,
    n_prompts, extraction_rate

Usage:
    python scripts/compute_math_quality.py \\
        --input-dir runs/diversity/ --output-dir analysis/math_quality/

    python scripts/compute_math_quality.py \\
        --generations runs/diversity/OLMo-3-7B/diversity_generations.jsonl \\
        --model OLMo-3-7B --output-dir analysis/math_quality/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# Add src to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evaluation.math_answer_extraction import (
    answers_match,
    extract_generated_answer,
    extract_gsm8k_gold_answer,
    normalize_number,
    strip_think_tags,
)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "bench_gsm8k_diversity": {
        "hf_repo": "gsm8k",
        "hf_subset": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "prompt_prefix": "",
    },
    "gsm8k_diversity": {
        "hf_repo": "gsm8k",
        "hf_subset": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "prompt_prefix": "",
    },
    "math_algebra_diversity": {
        "hf_repo": "EleutherAI/hendrycks_math",
        "hf_subset": "algebra",
        "split": "test",
        "question_field": "problem",
        "answer_field": "solution",
        "prompt_prefix": "",
    },
    "math_geometry_diversity": {
        "hf_repo": "EleutherAI/hendrycks_math",
        "hf_subset": "geometry",
        "split": "test",
        "question_field": "problem",
        "answer_field": "solution",
        "prompt_prefix": "",
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


def _extract_gold_answer(answer_text: str, task_name: str) -> Optional[str]:
    """Extract gold answer based on task type."""
    if "gsm8k" in task_name:
        return extract_gsm8k_gold_answer(answer_text)
    elif "math_algebra" in task_name or "math_geometry" in task_name:
        # MATH tasks use \boxed{} in solutions
        return extract_generated_answer(answer_text)
    elif "truthfulqa" in task_name:
        # TruthfulQA: gold is plain text best_answer
        return answer_text.strip() if answer_text.strip() else None
    return extract_gsm8k_gold_answer(answer_text)


def _build_gold_index(config: dict, task_name: str = "") -> Dict[str, str]:
    """Build question prefix -> gold answer mapping."""
    from datasets import load_dataset

    ds = load_dataset(config["hf_repo"], config["hf_subset"], split=config["split"])
    index = {}
    for row in ds:
        question = str(row[config["question_field"]])
        answer_text = str(row[config["answer_field"]])
        gold = _extract_gold_answer(answer_text, task_name)
        if gold is not None:
            key = question[:200].strip()
            index[key] = gold
    return index


def _extract_question_from_prompt(prompt: str) -> str:
    """Extract the question text from a formatted prompt."""
    # GSM8K prompts may have few-shot examples; take the last question
    # Common format: "Question: ... Answer:" or just the question text
    # Also handle MATH format: "Problem: ... Solution:"
    markers = ["Question:", "Q:", "Problem:"]
    last_pos = -1
    last_marker = ""
    for marker in markers:
        pos = prompt.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            last_marker = marker

    if last_pos >= 0:
        text = prompt[last_pos + len(last_marker):]
        # Remove trailing markers
        for end_marker in ["Answer:", "A:", "Solution:", "\n\n"]:
            pos = text.rfind(end_marker)
            if pos >= 0:
                text = text[:pos]
        return text.strip()

    return prompt.strip()


def _match_gold(prompt: str, gold_index: Dict[str, str]) -> Optional[str]:
    """Match a generation prompt to its gold answer."""
    question = _extract_question_from_prompt(prompt)
    key = question[:200].strip()
    return gold_index.get(key)


def evaluate_gsm8k(
    records: List[Dict],
    config: dict,
    task_name: str = "",
) -> dict:
    """Evaluate accuracy/majority-vote from generation records."""
    print(f"  Loading gold answers for {task_name or 'gsm8k'}...")
    gold_index = _build_gold_index(config, task_name=task_name)
    print(f"  Loaded {len(gold_index)} gold answer(s)")

    n_prompts = 0
    n_first_correct = 0
    n_majority_correct = 0
    n_any_correct = 0
    n_total_extracted = 0
    n_total_outputs = 0

    for record in records:
        prompt = record.get("prompt", "")
        outputs = record.get("outputs", [])
        if not outputs:
            continue

        gold = _match_gold(prompt, gold_index)
        if gold is None:
            continue
        n_prompts += 1
        n_total_outputs += len(outputs)

        # Extract answers from all generations
        extracted = []
        for out in outputs:
            ans = extract_generated_answer(out)
            if ans is not None:
                extracted.append(ans)
                n_total_extracted += 1
            else:
                extracted.append(None)

        # Filter None for quality metrics
        valid_extracted = [a for a in extracted if a is not None]

        # accuracy@1: first generation
        first_ans = extract_generated_answer(outputs[0])
        if first_ans is not None and answers_match(first_ans, gold):
            n_first_correct += 1

        # pass@16: any generation correct
        any_correct = any(answers_match(a, gold) for a in valid_extracted)
        if any_correct:
            n_any_correct += 1

        # majority vote (self-consistency): most frequent extracted answer
        # Use __UNPARSEABLE__ sentinel for None answers so they participate
        # in the vote (a model that can't parse most answers shouldn't get
        # credit if the parseable minority happens to be correct).
        vote_answers = [a if a is not None else "__UNPARSEABLE__" for a in extracted]
        if vote_answers:
            counter = Counter(vote_answers)
            top_two = counter.most_common(2)
            # Skip if top two are tied — no clear majority
            if len(top_two) >= 2 and top_two[0][1] == top_two[1][1]:
                pass  # no majority, don't count as correct
            else:
                majority_ans = top_two[0][0]
                if majority_ans != "__UNPARSEABLE__" and answers_match(majority_ans, gold):
                    n_majority_correct += 1

    return {
        "n_prompts": n_prompts,
        "accuracy_at_1": n_first_correct / max(n_prompts, 1),
        "majority_vote_at_16": n_majority_correct / max(n_prompts, 1),
        "pass_at_16": n_any_correct / max(n_prompts, 1),
        "extraction_rate": n_total_extracted / max(n_total_outputs, 1),
    }


def _evaluate_truthfulqa(records: List[Dict]) -> dict:
    """Evaluate TruthfulQA accuracy with multiple correct answers."""
    from datasets import load_dataset

    print("  Loading TruthfulQA correct answers...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    answer_index: Dict[str, List[str]] = {}
    for row in ds:
        question = str(row["question"])
        key = f"Q: {question}\nA:"[:200].strip()
        correct_answers = list(row["correct_answers"])
        best = str(row["best_answer"])
        if best not in correct_answers:
            correct_answers.append(best)
        answer_index[key] = correct_answers
    print(f"  Loaded {len(answer_index)} questions with correct answers")

    n_prompts = 0
    n_first_correct = 0
    n_majority_correct = 0
    n_any_correct = 0

    for record in records:
        outputs = record.get("outputs", [])
        if not outputs:
            continue
        prompt = record.get("prompt", "")
        key = prompt[:200].strip()
        correct_answers = answer_index.get(key)
        if correct_answers is None:
            continue
        n_prompts += 1

        def _match(gen: str) -> bool:
            gen_norm = strip_think_tags(gen).strip().lower()
            if not gen_norm:
                return False
            for ans in correct_answers:
                ans_norm = ans.strip().lower()
                if ans_norm and (ans_norm in gen_norm or gen_norm in ans_norm):
                    return True
            return False

        gen_correct = [_match(out) for out in outputs]
        if gen_correct[0]:
            n_first_correct += 1
        if any(gen_correct):
            n_any_correct += 1
        if sum(gen_correct) > len(outputs) / 2:
            n_majority_correct += 1

    return {
        "accuracy_at_1": n_first_correct / max(n_prompts, 1),
        "majority_vote_at_16": n_majority_correct / max(n_prompts, 1),
        "pass_at_16": n_any_correct / max(n_prompts, 1),
        "n_prompts": n_prompts,
        "extraction_rate": 1.0,
    }


def process_model(
    gen_path: Path,
    model_name: str,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Process math tasks for one model."""
    print(f"\nProcessing model: {model_name}")
    by_task = _load_generations(gen_path)
    # Accept both TASK_CONFIG tasks and truthfulqa_diversity
    supported = set(TASK_CONFIG) | {"truthfulqa_diversity"}
    math_tasks = [t for t in sorted(by_task) if t in supported]

    if not math_tasks:
        print(f"  No math/reasoning tasks found (available: {list(by_task.keys())})")
        return

    print(f"  Math/reasoning tasks: {math_tasks}")

    if dry_run:
        for task_name in math_tasks:
            n = len(by_task[task_name])
            print(f"  [dry-run] {task_name}: {n} prompts")
        return

    rows = []
    for task_name in math_tasks:
        records = by_task[task_name]
        print(f"  Evaluating: {task_name} ({len(records)} prompts)")

        if task_name == "truthfulqa_diversity":
            result = _evaluate_truthfulqa(records)
        else:
            config = TASK_CONFIG[task_name]
            result = evaluate_gsm8k(
                records, config,
                task_name=task_name,
            )
        result["model"] = model_name
        result["task"] = task_name
        rows.append(result)

        print(
            f"  Results: acc@1={result['accuracy_at_1']:.4f}, "
            f"majority@16={result['majority_vote_at_16']:.4f}, "
            f"pass@16={result['pass_at_16']:.4f}, "
            f"extraction_rate={result['extraction_rate']:.4f}"
        )

    if rows:
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            col_order = [
                "model", "task", "accuracy_at_1", "majority_vote_at_16",
                "pass_at_16", "n_prompts", "extraction_rate",
            ]
            df = df[[c for c in col_order if c in df.columns]]

            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "math_quality_results.csv"

            if out_path.exists():
                existing = pd.read_csv(out_path)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["model", "task"], keep="last")

            df.to_csv(out_path, index=False, float_format="%.6f")
            print(f"  Saved to {out_path}")
        except ImportError:
            out_path = output_dir / "math_quality_results.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  Saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy/majority-vote quality from GSM8K diversity outputs.",
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
        default=Path("analysis/math_quality"),
        help="Output directory (default: analysis/math_quality/)",
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

        gen_files = sorted(args.input_dir.glob("*/diversity_generations.jsonl"))
        if not gen_files:
            raise SystemExit(f"No diversity_generations.jsonl found in {args.input_dir}")

        print(f"Found {len(gen_files)} model(s) with generation files")
        for gf in gen_files:
            model_name = gf.parent.name
            process_model(gf, model_name, args.output_dir, args.dry_run)
    else:
        raise SystemExit("Provide either --generations or --input-dir")


if __name__ == "__main__":
    main()
