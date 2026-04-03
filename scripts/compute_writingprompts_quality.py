#!/usr/bin/env python3
"""
Compute WritingPrompts quality via pairwise LLM judging against the Base model.

Uses the Arena-Hard creative writing judging protocol: each model's gen-0 is
compared against the Base model's gen-0 using GPT as judge, with double
position-swap evaluation to debias position effects.

Workflow:
    1. --prepare  : Discover parquets/JSONLs, create OpenAI Batch API input files
    2. (manually) : Submit batch files to OpenAI and download results
    3. --score    : Parse results, compute position-debiased win rates, output CSV

The position-swap protocol works as follows:
  - Round 1: model=Assistant_A, Base=Assistant_B
  - Round 2: Base=Assistant_A, model=Assistant_B
  - Final score: average of both rounds (after flipping Round 2 perspective)

This ensures that any first-position bias cancels out in expectation.

Output: analysis/all_writingprompts_quality.csv with columns:
    model, task, win_rate, win_rate_r1, win_rate_r2, n_samples, n_unparseable

Usage:
    python scripts/compute_writingprompts_quality.py --prepare
    python scripts/compute_writingprompts_quality.py --prepare --dry-run
    python scripts/compute_writingprompts_quality.py --score --results-dir batch_results/
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_DIR_NAME = "allenai__OLMo-3-1025-7B"
TASK_NAME = "writingprompts_diversity"

# Arena-Hard creative writing judging prompt (verbatim from Arena-Hard-Auto)
ARENA_HARD_CREATIVE_SYSTEM = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""

USER_TEMPLATE = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""

# Score mapping: A wins -> 1.0, B wins -> 0.0 (from perspective of A)
SCORE_MAP = {
    "A>>B": 1.0,
    "A>B": 0.75,
    "A=B": 0.5,
    "B>A": 0.25,
    "B>>A": 0.0,
}

VERDICT_PATTERN = re.compile(r'\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]')

# Judge model
JUDGE_MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# CoT stripping
# ---------------------------------------------------------------------------

def _strip_cot(text: str) -> str:
    """Strip chain-of-thought <think> tags from model output.

    Handles paired <think>...</think> tags and orphaned </think> at the
    start of output (opening tag consumed by the chat template).
    """
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if start != -1 and end != -1:
            text = text[:start] + text[end + len("</think>"):]
        else:
            break
    if "</think>" in text and "<think>" not in text:
        end_idx = text.find("</think>")
        text = text[end_idx + len("</think>"):]
    return text.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet_records(path: Path) -> List[Tuple[str, str]]:
    """Load a parquet and return list of (query, gen-0-text) tuples."""
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        doc = row["doc"]
        mr = row["model_response"]
        query = doc.get("query", "")
        raw_outputs = mr.get("text", []) if isinstance(mr, dict) else []
        if hasattr(raw_outputs, "tolist"):
            raw_outputs = raw_outputs.tolist()
        if raw_outputs:
            gen0 = _strip_cot(str(raw_outputs[0]))
        else:
            gen0 = ""
        records.append((query.strip(), gen0))
    return records


def _load_jsonl_records(path: Path, task_filter: str = "") -> List[Tuple[str, str]]:
    """Load JSONL generation file and return list of (prompt, gen-0-text) tuples."""
    records = []
    seen: set = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            task = r.get("task", "").split("|")[0]
            if task_filter and task != task_filter:
                continue
            input_id = str(r.get("input_id", ""))
            dedup_key = f"{task}:{input_id}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            outputs = r.get("outputs", [])
            if outputs:
                gen0 = _strip_cot(str(outputs[0]))
            else:
                gen0 = ""
            records.append((r.get("prompt", "").strip(), gen0))
    return records


def load_records(path: Path, task_filter: str = "") -> List[Tuple[str, str]]:
    """Load records from parquet or JSONL. Returns list of (query, gen0_text)."""
    if path.suffix == ".parquet":
        return _load_parquet_records(path)
    elif path.suffix == ".jsonl":
        return _load_jsonl_records(path, task_filter=task_filter)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_writingprompts_files(
    runs_dir: Path,
) -> Dict[str, Path]:
    """Find WritingPrompts data files for all models.

    Searches runs/open_ended/ (parquets) and runs/no_cot_open_ended/ (JSONLs).

    Returns: {model_name: path_to_data_file}
    """
    result: Dict[str, Path] = {}

    # --- Standard runs (parquets in runs/open_ended/) ---
    open_ended_dir = runs_dir / "open_ended"
    if open_ended_dir.is_dir():
        for model_dir in sorted(open_ended_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            # Search for writingprompts parquet
            parquet_files = glob.glob(
                str(model_dir / "details" / "**" / "details_writingprompts_diversity*"),
                recursive=True,
            )
            parquet_files = [p for p in parquet_files if p.endswith(".parquet")]
            if parquet_files:
                result[model_dir.name] = Path(sorted(parquet_files)[-1])

    # --- No-CoT runs (JSONLs in runs/no_cot_open_ended/) ---
    no_cot_dir = runs_dir / "no_cot_open_ended"
    if no_cot_dir.is_dir():
        for model_dir in sorted(no_cot_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            # Check for task-specific JSONL
            jsonl_path = model_dir / f"diversity_generations_{TASK_NAME}.jsonl"
            if jsonl_path.exists():
                model_name = f"no_cot__{model_dir.name}"
                result[model_name] = jsonl_path
            else:
                # Check generic JSONL
                generic = model_dir / "diversity_generations.jsonl"
                if generic.exists():
                    # Verify it contains writingprompts data
                    with generic.open() as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            task = json.loads(line).get("task", "").split("|")[0]
                            if task == TASK_NAME:
                                model_name = f"no_cot__{model_dir.name}"
                                result[model_name] = generic
                                break

    return result


# ---------------------------------------------------------------------------
# Batch file preparation
# ---------------------------------------------------------------------------

def _make_batch_request(
    custom_id: str,
    question: str,
    answer_a: str,
    answer_b: str,
) -> dict:
    """Create a single OpenAI Batch API request object."""
    user_content = USER_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": ARENA_HARD_CREATIVE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": 1024,
        },
    }


def _match_records_by_prompt_prefix(
    model_records: List[Tuple[str, str]],
    base_records: List[Tuple[str, str]],
    prefix_len: int = 200,
) -> List[Tuple[str, str, str]]:
    """Match model and base records by prompt prefix.

    Returns list of (query, model_gen0, base_gen0) for matched pairs.
    Used when row-index alignment is unreliable (e.g., JSONLs from
    different generation runs).
    """
    # Build base index by prompt prefix
    base_index: Dict[str, str] = {}
    for query, gen0 in base_records:
        key = query[:prefix_len].strip()
        base_index[key] = gen0

    matched = []
    for query, model_gen0 in model_records:
        key = query[:prefix_len].strip()
        base_gen0 = base_index.get(key)
        if base_gen0 is not None:
            matched.append((query, model_gen0, base_gen0))

    return matched


def prepare(args: argparse.Namespace) -> None:
    """Prepare OpenAI Batch API input files for pairwise judging."""
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.batch_dir)

    print("Discovering WritingPrompts data files...")
    all_files = discover_writingprompts_files(runs_dir)
    print(f"Found {len(all_files)} models with WritingPrompts data")

    if BASE_MODEL_DIR_NAME not in all_files:
        raise SystemExit(
            f"Base model not found: {BASE_MODEL_DIR_NAME}\n"
            f"Available models: {sorted(all_files.keys())}"
        )

    # Load base model records
    base_path = all_files[BASE_MODEL_DIR_NAME]
    print(f"\nLoading Base model from {base_path}")
    base_records = load_records(base_path, task_filter=TASK_NAME)
    print(f"  Base model: {len(base_records)} samples")

    # Process each non-base model
    non_base_models = {k: v for k, v in all_files.items() if k != BASE_MODEL_DIR_NAME}
    print(f"\nPreparing batch files for {len(non_base_models)} non-base models...")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_name, data_path in sorted(non_base_models.items()):
        print(f"\n--- {model_name} ---")
        print(f"  Source: {data_path}")

        model_records = load_records(data_path, task_filter=TASK_NAME)
        print(f"  Model records: {len(model_records)}")

        # Determine matching strategy
        is_no_cot = model_name.startswith("no_cot__")

        if is_no_cot or data_path.suffix == ".jsonl":
            # Match by prompt prefix for JSONLs (different generation runs)
            matched = _match_records_by_prompt_prefix(model_records, base_records)
            print(f"  Matched by prompt prefix: {len(matched)} pairs")
        else:
            # Match by row index for parquets (same dataset ordering)
            n = min(len(model_records), len(base_records))
            matched = [
                (model_records[i][0], model_records[i][1], base_records[i][1])
                for i in range(n)
            ]
            print(f"  Matched by row index: {len(matched)} pairs")

        if not matched:
            print("  WARNING: No matched pairs, skipping")
            continue

        # Filter out empty generations
        matched = [
            (q, m, b) for q, m, b in matched
            if m.strip() and b.strip()
        ]
        print(f"  After filtering empty: {len(matched)} pairs")

        if args.dry_run:
            summary_rows.append({
                "model": model_name, "n_pairs": len(matched),
                "source": str(data_path),
            })
            continue

        # Create sanitized filename
        safe_name = model_name.replace("/", "_").replace(" ", "_")

        # Round 1: model=A, Base=B
        r1_path = output_dir / f"round1_{safe_name}.jsonl"
        r1_requests = []
        for idx, (query, model_gen0, base_gen0) in enumerate(matched):
            custom_id = f"r1||{safe_name}||{TASK_NAME}||{idx}"
            req = _make_batch_request(custom_id, query, model_gen0, base_gen0)
            r1_requests.append(req)

        with r1_path.open("w") as f:
            for req in r1_requests:
                f.write(json.dumps(req) + "\n")
        print(f"  Round 1: {len(r1_requests)} requests -> {r1_path}")

        # Round 2: Base=A, model=B (position swap)
        r2_path = output_dir / f"round2_{safe_name}.jsonl"
        r2_requests = []
        for idx, (query, model_gen0, base_gen0) in enumerate(matched):
            custom_id = f"r2||{safe_name}||{TASK_NAME}||{idx}"
            req = _make_batch_request(custom_id, query, base_gen0, model_gen0)
            r2_requests.append(req)

        with r2_path.open("w") as f:
            for req in r2_requests:
                f.write(json.dumps(req) + "\n")
        print(f"  Round 2: {len(r2_requests)} requests -> {r2_path}")

        summary_rows.append({
            "model": model_name,
            "n_pairs": len(matched),
            "source": str(data_path),
            "round1_file": str(r1_path),
            "round2_file": str(r2_path),
        })

    # Print summary
    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)
    for row in summary_rows:
        print(f"  {row['model']}: {row['n_pairs']} pairs")
    total_pairs = sum(r["n_pairs"] for r in summary_rows)
    total_requests = total_pairs * 2  # two rounds
    print(f"\nTotal: {total_pairs} pairs, {total_requests} API requests")
    if args.dry_run:
        print("(DRY RUN -- no files written)")


# ---------------------------------------------------------------------------
# Verdict parsing & scoring
# ---------------------------------------------------------------------------

def parse_verdict(text: str) -> Optional[str]:
    """Extract verdict label from judge response text.

    Returns the verdict string (e.g., "A>>B") or None if unparseable.
    """
    match = VERDICT_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


def score(args: argparse.Namespace) -> None:
    """Parse batch results and compute position-debiased win rates."""
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    # Collect all result files
    result_files = sorted(results_dir.glob("*.jsonl"))
    if not result_files:
        raise SystemExit(f"No JSONL result files found in {results_dir}")

    print(f"Found {len(result_files)} result files in {results_dir}")

    # Parse all results into {model -> {round -> {idx -> score}}}
    # Structure: scores[model][round_num][sample_idx] = score_from_model_perspective
    scores: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    n_unparseable: Dict[str, int] = defaultdict(int)

    for result_file in result_files:
        print(f"  Parsing {result_file.name}...")
        with result_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = json.loads(line)
                custom_id = result.get("custom_id", "")

                # Parse custom_id: r{round}||{model}||{task}||{idx}
                parts = custom_id.split("||")
                if len(parts) != 4:
                    continue
                round_str, model_name, task, idx_str = parts
                round_num = int(round_str[1])  # "r1" -> 1, "r2" -> 2
                idx = int(idx_str)

                # Extract the judge's response text
                response = result.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                if not choices:
                    n_unparseable[model_name] += 1
                    continue

                judge_text = choices[0].get("message", {}).get("content", "")
                verdict = parse_verdict(judge_text)

                if verdict is None:
                    n_unparseable[model_name] += 1
                    continue

                raw_score = SCORE_MAP[verdict]

                if round_num == 1:
                    # Round 1: model=A, base=B. Score is already from model perspective.
                    model_score = raw_score
                elif round_num == 2:
                    # Round 2: base=A, model=B. Need to flip perspective.
                    # A>>B means base wins -> model_score = 0.0
                    # B>>A means model wins -> model_score = 1.0
                    model_score = 1.0 - raw_score
                else:
                    continue

                scores[model_name][round_num][idx] = model_score

    # Compute per-model aggregates
    print("\n" + "=" * 60)
    print("WRITINGPROMPTS QUALITY (LLM-as-Judge Win Rate vs Base)")
    print("=" * 60)

    rows = []
    for model_name in sorted(scores.keys()):
        r1_scores = scores[model_name].get(1, {})
        r2_scores = scores[model_name].get(2, {})

        # Only count samples that have BOTH rounds for fair comparison
        common_indices = set(r1_scores.keys()) & set(r2_scores.keys())

        if not common_indices:
            print(f"  {model_name}: no samples with both rounds, skipping")
            continue

        r1_vals = [r1_scores[i] for i in sorted(common_indices)]
        r2_vals = [r2_scores[i] for i in sorted(common_indices)]

        win_rate_r1 = sum(r1_vals) / len(r1_vals)
        win_rate_r2 = sum(r2_vals) / len(r2_vals)
        win_rate = (win_rate_r1 + win_rate_r2) / 2.0

        n_unparsed = n_unparseable.get(model_name, 0)

        print(f"  {model_name}: win_rate={win_rate:.4f} "
              f"(r1={win_rate_r1:.4f}, r2={win_rate_r2:.4f}, "
              f"n={len(common_indices)}, unparsed={n_unparsed})")

        rows.append({
            "model": model_name,
            "task": TASK_NAME,
            "win_rate": win_rate,
            "win_rate_r1": win_rate_r1,
            "win_rate_r2": win_rate_r2,
            "n_samples": len(common_indices),
            "n_unparseable": n_unparsed,
        })

    if rows:
        df = pd.DataFrame(rows)
        col_order = [
            "model", "task", "win_rate", "win_rate_r1", "win_rate_r2",
            "n_samples", "n_unparseable",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "all_writingprompts_quality.csv"
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"\nSaved to {out_path}")
    else:
        print("\nNo results to save.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WritingPrompts quality via pairwise LLM judging.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- prepare --
    prep = subparsers.add_parser(
        "prepare",
        help="Create OpenAI Batch API input files",
    )
    prep.add_argument(
        "--runs-dir", type=str, default="runs",
        help="Root runs directory (default: runs/)",
    )
    prep.add_argument(
        "--batch-dir", type=str, default="batch_inputs/writingprompts",
        help="Output directory for batch JSONL files (default: batch_inputs/writingprompts/)",
    )
    prep.add_argument(
        "--dry-run", action="store_true",
        help="List what would be prepared without writing files",
    )

    # -- score --
    sc = subparsers.add_parser(
        "score",
        help="Parse batch results and compute win rates",
    )
    sc.add_argument(
        "--results-dir", type=str, default="batch_results/writingprompts",
        help="Directory containing batch result JSONL files",
    )
    sc.add_argument(
        "--output-dir", type=str, default="analysis",
        help="Output directory for CSV (default: analysis/)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    return args


def main() -> None:
    args = parse_args()

    if args.command == "prepare":
        prepare(args)
    elif args.command == "score":
        score(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")

    print("\nDone!")


if __name__ == "__main__":
    main()
