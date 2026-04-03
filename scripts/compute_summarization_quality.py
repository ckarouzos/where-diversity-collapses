#!/usr/bin/env python3
"""Summarization quality via LLM-judge (Kirk et al., ICLR 2024).

Produces batch JSONL files for the OpenAI Batch API, where a judge model
compares each model's first generation (gen-0) against the reference summary.
Position bias is mitigated via deterministic hashing to swap A/B order.

Two-phase workflow:
  1. --prepare: read parquets, build Kirk prompts, write batch JSONL + manifest
  2. --score:   read completed batch results, parse verdicts, aggregate to CSV

Usage:
    python scripts/compute_summarization_quality.py --prepare [--dry-run]
    python scripts/compute_summarization_quality.py --score --results-dir DIR
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_MODEL = "gpt-4.1-mini"

TASK_DOC_TYPE: Dict[str, str] = {
    "tldr": "forum post",
    "cnn_dailymail": "news article",
    "xsum": "news article",
}

# Task names as they appear in lighteval parquet filenames
TASK_PARQUET_NAMES: Dict[str, str] = {
    "tldr": "tldr_diversity",
    "cnn_dailymail": "cnn_dailymail_diversity",
    "xsum": "xsum_diversity",
}

SYSTEM_PROMPT = (
    "You are a helpful assistant, that ranks models by the quality of their answers."
)

# Kirk et al. (ICLR 2024) pairwise ranking prompt template.
# Placeholders: {doc_type}, {source}, {summary_a}, {summary_b}
_KIRK_TEMPLATE = """\
Which of the following summaries does a better job of summarizing the most \
important points in the given {doc_type}, without including unimportant or \
irrelevant details? A good summary is both precise and concise.

Post: \"\"\"{source}\"\"\"

Summary A:
{{"model": "model_1", "summary": \"\"\"{summary_a}\"\"\"}}

Summary B:
{{"model": "model_2", "summary": \"\"\"{summary_b}\"\"\"}}

Now please rank the models by the quality of their summaries, so that the \
model with rank 1 has the best summary. Then return a list of the model \
names and ranks, i.e., produce the following output:
[{{'model': <model-name>, 'rank': <model-rank>}},
 {{'model': <model-name>, 'rank': <model-rank>}}]

Your response must be a valid Python dictionary and should contain nothing \
else because we will directly execute it in Python. Please provide the \
ranking that the majority of humans would give."""


# ---------------------------------------------------------------------------
# Utility functions
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


def _should_swap(source: str, summary_a: str, summary_b: str) -> bool:
    """Deterministic position-bias debiasing via SHA-256 hash.

    Returns True when the candidate/reference order should be swapped so that
    across the dataset each position is equally likely.  Mirrors the pattern
    in ``lighteval_llm_judge_metrics.py``.
    """
    payload = f"{source}\n||\n{summary_a}\n||\n{summary_b}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(h[-1], 16) % 2 == 1


def _extract_source(query: str, task: str) -> str:
    """Extract the source document text from a lighteval-formatted prompt.

    TL;DR prompts:   ``SUBREDDIT: ...\\nTITLE: ...\\nPOST: {text}\\nTL;DR:``
    CNN / XSum:       ``ARTICLE: {text}\\nTL;DR:``

    We want the body text without the prompt scaffolding.
    """
    if task == "tldr":
        # Extract everything between "POST: " and the final "\nTL;DR:"
        post_start = query.find("POST: ")
        if post_start != -1:
            body = query[post_start + len("POST: "):]
        else:
            body = query
        # Remove trailing TL;DR marker
        tldr_pos = body.rfind("\nTL;DR:")
        if tldr_pos != -1:
            body = body[:tldr_pos]
        else:
            # Fallback: strip any trailing "TL;DR:" at the very end
            body = body.rstrip()
            if body.endswith("TL;DR:"):
                body = body[: -len("TL;DR:")].rstrip()
        return body.strip()
    else:
        # cnn_dailymail, xsum: "ARTICLE: {text}\nTL;DR:"
        art_start = query.find("ARTICLE: ")
        if art_start != -1:
            body = query[art_start + len("ARTICLE: "):]
        else:
            body = query
        tldr_pos = body.rfind("\nTL;DR:")
        if tldr_pos != -1:
            body = body[:tldr_pos]
        else:
            body = body.rstrip()
            if body.endswith("TL;DR:"):
                body = body[: -len("TL;DR:")].rstrip()
        return body.strip()


def _build_kirk_prompt(
    source: str, summary_a: str, summary_b: str, doc_type: str
) -> str:
    """Build the Kirk et al. pairwise ranking user prompt."""
    return _KIRK_TEMPLATE.format(
        doc_type=doc_type,
        source=source,
        summary_a=summary_a,
        summary_b=summary_b,
    )


def _parse_kirk_verdict(text: str) -> Optional[str]:
    """Parse the Kirk ranking response and return 'win', 'loss', or 'tie'.

    The expected output is a Python list of dicts:
        [{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}]

    Returns the verdict *from model_1's perspective* (i.e., ignoring swap):
        - 'win'  if model_1 has rank 1
        - 'loss' if model_2 has rank 1
        - 'tie'  if both have the same rank

    Returns None if parsing fails completely.
    """
    text = text.strip()

    # Strategy 1: try ast.literal_eval on the full text
    parsed = None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    # Strategy 2: extract the list with regex and try again
    if parsed is None:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            try:
                parsed = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                pass

    # Strategy 3: regex for individual rank assignments
    if parsed is None:
        ranks: Dict[str, int] = {}
        for match in re.finditer(
            r"['\"]model['\"]\s*:\s*['\"](\w+)['\"].*?['\"]rank['\"]\s*:\s*(\d+)",
            text,
            flags=re.DOTALL,
        ):
            ranks[match.group(1)] = int(match.group(2))
        if ranks:
            parsed = [{"model": k, "rank": v} for k, v in ranks.items()]

    if not parsed or not isinstance(parsed, list):
        return None

    # Build model -> rank mapping
    rank_map: Dict[str, int] = {}
    for entry in parsed:
        if isinstance(entry, dict) and "model" in entry and "rank" in entry:
            rank_map[str(entry["model"])] = int(entry["rank"])

    r1 = rank_map.get("model_1")
    r2 = rank_map.get("model_2")

    if r1 is None or r2 is None:
        return None
    if r1 < r2:
        return "win"
    if r1 > r2:
        return "loss"
    return "tie"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_models(runs_dir: Path) -> List[str]:
    """Find model directories under runs/diversity/ (and no_cot variants)."""
    models: List[str] = []
    for experiment in ("diversity", "no_cot_diversity"):
        exp_dir = runs_dir / experiment
        if not exp_dir.is_dir():
            continue
        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            prefix = "no_cot__" if experiment.startswith("no_cot_") else ""
            models.append(prefix + model_dir.name)
    return models


def _find_data_file(runs_dir: Path, model_name: str, task_parquet: str) -> Optional[Path]:
    """Locate the parquet or JSONL file for a given model and task."""
    is_no_cot = model_name.startswith("no_cot__")
    if is_no_cot:
        experiment = "no_cot_diversity"
        dir_name = model_name[len("no_cot__"):]
    else:
        experiment = "diversity"
        dir_name = model_name

    model_dir = runs_dir / experiment / dir_name
    if not model_dir.is_dir():
        return None

    import glob as glob_mod

    # Try parquet first
    pattern = str(model_dir / "details" / "**" / f"details_{task_parquet}|*.parquet")
    matches = glob_mod.glob(pattern, recursive=True)
    if matches:
        return Path(sorted(matches)[-1])

    # Fallback: task-specific JSONL
    jsonl_path = model_dir / f"diversity_generations_{task_parquet}.jsonl"
    if jsonl_path.exists():
        return jsonl_path

    return None


# Keep old name as alias
_find_parquet = _find_data_file


def load_parquet_records(
    path: Path, task_filter: str = ""
) -> List[Dict]:
    """Load parquet, return list of dicts with query, reference, and gen-0.

    Each record contains:
        - query: the raw prompt (with ARTICLE:/POST: scaffolding)
        - reference: the gold summary from choices[0]
        - gen_0: the first model generation (raw, before CoT stripping)
        - idx: integer index within the file
    """
    df = pd.read_parquet(path)
    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        doc = row["doc"]
        mr = row["model_response"]

        query = doc.get("query", "")
        choices = doc.get("choices", [])
        if hasattr(choices, "tolist"):
            choices = choices.tolist()
        reference = str(choices[0]).strip() if choices else ""

        raw_outputs = mr.get("text", []) if isinstance(mr, dict) else []
        if hasattr(raw_outputs, "tolist"):
            raw_outputs = raw_outputs.tolist()
        gen_0 = str(raw_outputs[0]).strip() if raw_outputs else ""

        records.append({
            "query": query,
            "reference": reference,
            "gen_0": gen_0,
            "idx": idx,
        })
    return records


def load_jsonl_records(
    path: Path, task_filter: str = ""
) -> List[Dict]:
    """Load a JSONL generation file and return the same record format as parquet.

    JSONL records have {task, prompt, outputs, input_id} but no gold/reference.
    For summarization, the reference must be fetched from the HF dataset.
    """
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
            if input_id in seen:
                continue
            seen.add(input_id)
            outputs = r.get("outputs", [])
            gen_0 = str(outputs[0]).strip() if outputs else ""
            records.append({
                "query": r.get("prompt", ""),
                "reference": "",  # Will be filled from HF dataset
                "gen_0": gen_0,
                "idx": len(records),
            })
    return records


_SUMM_GOLD_INDEXES: Dict[str, Dict[str, str]] = {}


def _get_summ_gold_index(task_short: str) -> Dict[str, str]:
    """Build prompt-prefix -> reference summary index from HF dataset."""
    if task_short in _SUMM_GOLD_INDEXES:
        return _SUMM_GOLD_INDEXES[task_short]
    from datasets import load_dataset
    if task_short == "tldr":
        ds = load_dataset("UCL-DARK/openai-tldr-filtered", split="test")
        index = {}
        for row in ds:
            prompt_key = f"SUBREDDIT: {row.get('subreddit', '')}\nTITLE: {row.get('title', '')}"[:200].strip()
            index[prompt_key] = row.get("summary", "")
        _SUMM_GOLD_INDEXES[task_short] = index
        return index
    elif task_short == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        index = {}
        for row in ds:
            prompt_key = f"ARTICLE: {row['article']}"[:200].strip()
            index[prompt_key] = row.get("highlights", "")
        _SUMM_GOLD_INDEXES[task_short] = index
        return index
    elif task_short == "xsum":
        ds = load_dataset("EdinburghNLP/xsum", split="test")
        index = {}
        for row in ds:
            prompt_key = f"ARTICLE: {row['document']}"[:200].strip()
            index[prompt_key] = row.get("summary", "")
        _SUMM_GOLD_INDEXES[task_short] = index
        return index
    return {}


def load_data_records(
    path: Path, task_filter: str = "", task_short: str = ""
) -> List[Dict]:
    """Load records from parquet or JSONL, filling in references if needed."""
    if path.suffix == ".parquet":
        return load_parquet_records(path, task_filter)

    records = load_jsonl_records(path, task_filter)
    # Fill in references from HF dataset for JSONL records
    if records and not records[0]["reference"]:
        gold_index = _get_summ_gold_index(task_short)
        if gold_index:
            matched = 0
            for rec in records:
                query = rec["query"]
                # For TL;DR JSONL records, strip POST body to match HF index key format.
                # HF index keys are "SUBREDDIT: ...\nTITLE: ..."[:200], but JSONL
                # prompts include the POST body which shifts the first 200 chars.
                if "\nPOST:" in query:
                    query = query[:query.index("\nPOST:")]
                key = query[:200].strip()
                ref = gold_index.get(key, "")
                if ref:
                    rec["reference"] = ref
                    matched += 1
            print(f"    Matched {matched}/{len(records)} references from HF dataset")
    return records


# ---------------------------------------------------------------------------
# Prepare mode: build batch JSONL for OpenAI Batch API
# ---------------------------------------------------------------------------


def _build_batch_line(
    custom_id: str,
    system_content: str,
    user_content: str,
    model: str = JUDGE_MODEL,
    temperature: float = 0,
    max_tokens: int = 256,
) -> dict:
    """Build a single batch API request line."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }


def prepare(args: argparse.Namespace) -> None:
    """Build batch JSONL files for the OpenAI Batch API.

    For each (model, task) pair, reads the parquet, extracts gen-0 and
    reference, builds the Kirk pairwise prompt (with position debiasing),
    and writes one JSONL file per task to the output directory.

    Also writes a manifest JSON that maps custom_id -> metadata for scoring.
    """
    runs_dir: Path = args.runs_dir
    output_dir: Path = args.output_dir
    dry_run: bool = args.dry_run

    output_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(runs_dir)
    print(f"Discovered {len(models)} models")

    manifest: List[dict] = []
    total_requests = 0
    batch_lines: List[dict] = []

    for task_short, task_parquet in sorted(TASK_PARQUET_NAMES.items()):
        doc_type = TASK_DOC_TYPE[task_short]
        task_lines: List[dict] = []

        for model_name in sorted(models):
            data_path = _find_data_file(runs_dir, model_name, task_parquet)
            if data_path is None:
                print(f"  SKIP {model_name} / {task_short}: no data file found")
                continue

            records = load_data_records(data_path, task_filter=task_parquet, task_short=task_short)
            # Skip records with no reference (couldn't match from HF)
            records = [r for r in records if r["reference"]]
            print(
                f"  {model_name} / {task_short}: {len(records)} samples"
                f" ({data_path.suffix}: {data_path.name})"
            )

            if dry_run:
                total_requests += len(records)
                continue

            for rec in records:
                query = rec["query"]
                reference = rec["reference"]
                gen_0 = _strip_cot(rec["gen_0"])
                idx = rec["idx"]

                if not gen_0 or not reference:
                    continue

                source = _extract_source(query, task_short)
                if not source:
                    continue

                # Determine swap for position debiasing
                swap = _should_swap(source, gen_0, reference)

                if swap:
                    summary_a = reference
                    summary_b = gen_0
                else:
                    summary_a = gen_0
                    summary_b = reference

                user_prompt = _build_kirk_prompt(
                    source, summary_a, summary_b, doc_type
                )

                swap_flag = "1" if swap else "0"
                custom_id = f"{model_name}||{task_short}||{idx}||{swap_flag}"

                line = _build_batch_line(
                    custom_id=custom_id,
                    system_content=SYSTEM_PROMPT,
                    user_content=user_prompt,
                )
                task_lines.append(line)
                batch_lines.append(line)

                manifest.append({
                    "custom_id": custom_id,
                    "model": model_name,
                    "task": task_short,
                    "idx": idx,
                    "swap": swap,
                })

        if not dry_run and task_lines:
            task_jsonl = output_dir / f"batch_{task_short}.jsonl"
            with task_jsonl.open("w") as f:
                for line in task_lines:
                    f.write(json.dumps(line) + "\n")
            print(f"  Wrote {len(task_lines)} requests to {task_jsonl}")

        total_requests += len(task_lines)

    # Write combined batch file (all tasks)
    if not dry_run and batch_lines:
        combined_path = output_dir / "batch_all_summarization.jsonl"
        with combined_path.open("w") as f:
            for line in batch_lines:
                f.write(json.dumps(line) + "\n")
        print(f"\nWrote combined batch: {combined_path} ({len(batch_lines)} requests)")

    # Write manifest
    if not dry_run and manifest:
        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote manifest: {manifest_path} ({len(manifest)} entries)")

    print(f"\nTotal requests: {total_requests}")
    if dry_run:
        print("(dry-run: no files written)")

    # Cost estimate
    # Rough token budget: ~500 input tokens per request, 50 output tokens
    est_input_tokens = total_requests * 500
    est_output_tokens = total_requests * 50
    print(f"Estimated tokens: ~{est_input_tokens:,} input, ~{est_output_tokens:,} output")


# ---------------------------------------------------------------------------
# Score mode: parse batch results and aggregate
# ---------------------------------------------------------------------------


def score(args: argparse.Namespace) -> None:
    """Read completed batch results, parse Kirk verdicts, aggregate to CSV.

    Expected results directory structure (from OpenAI Batch API output):
        results_dir/*.jsonl   -- one or more result JSONL files

    Each line has:
        {"custom_id": "MODEL||TASK||IDX||SWAP",
         "response": {"body": {"choices": [{"message": {"content": "..."}}]}}}

    Aggregates to: model, task, win_rate, n_wins, n_ties, n_losses, n_samples
    """
    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir

    if results_dir is None:
        raise SystemExit("--results-dir is required for --score mode")

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all result JSONL files
    result_files = sorted(results_dir.glob("*.jsonl"))
    if not result_files:
        raise SystemExit(f"No JSONL files found in {results_dir}")

    print(f"Found {len(result_files)} result files in {results_dir}")

    # Parse results
    # Key: (model, task) -> list of verdicts
    verdicts: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    seen_ids: set = set()
    n_parsed = 0
    n_failed = 0
    n_dedup = 0

    for rf in result_files:
        with rf.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    n_failed += 1
                    continue

                custom_id = record.get("custom_id", "")

                # Deduplicate: skip if we already processed this custom_id
                if custom_id in seen_ids:
                    n_dedup += 1
                    continue
                seen_ids.add(custom_id)

                parts = custom_id.split("||")
                if len(parts) != 4:
                    n_failed += 1
                    continue

                model_name, task_short, idx_str, swap_flag = parts
                swap = swap_flag == "1"

                # Extract judge response text
                response_body = (
                    record.get("response", {}).get("body", {})
                )
                choices = response_body.get("choices", [])
                if not choices:
                    n_failed += 1
                    continue

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    n_failed += 1
                    continue

                # Parse the Kirk verdict (from model_1's perspective)
                raw_verdict = _parse_kirk_verdict(content)
                if raw_verdict is None:
                    n_failed += 1
                    continue

                # Account for swap:
                # If swap=False: model_1 = candidate, model_2 = reference
                #   -> raw "win" means candidate wins
                # If swap=True:  model_1 = reference, model_2 = candidate
                #   -> raw "win" means reference wins, so candidate loses
                if swap:
                    if raw_verdict == "win":
                        final_verdict = "loss"
                    elif raw_verdict == "loss":
                        final_verdict = "win"
                    else:
                        final_verdict = "tie"
                else:
                    final_verdict = raw_verdict

                verdicts[(model_name, task_short)].append(final_verdict)
                n_parsed += 1

    print(f"Parsed {n_parsed} verdicts, {n_failed} failures, {n_dedup} duplicates skipped")

    if not verdicts:
        raise SystemExit("No valid verdicts found")

    # Aggregate
    rows = []
    for (model_name, task_short), verdict_list in sorted(verdicts.items()):
        n_wins = sum(1 for v in verdict_list if v == "win")
        n_ties = sum(1 for v in verdict_list if v == "tie")
        n_losses = sum(1 for v in verdict_list if v == "loss")
        n_samples = len(verdict_list)
        win_rate = (n_wins + 0.5 * n_ties) / max(n_samples, 1)

        rows.append({
            "model": model_name,
            "task": task_short,
            "win_rate": win_rate,
            "n_wins": n_wins,
            "n_ties": n_ties,
            "n_losses": n_losses,
            "n_samples": n_samples,
        })

    df = pd.DataFrame(rows)
    col_order = ["model", "task", "win_rate", "n_wins", "n_ties", "n_losses", "n_samples"]
    df = df[[c for c in col_order if c in df.columns]]

    out_path = output_dir / "all_summarization_quality.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved summarization quality results to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARIZATION QUALITY (LLM judge win-rate vs reference)")
    print("=" * 70)
    for task_short in sorted(TASK_DOC_TYPE.keys()):
        task_df = df[df["task"] == task_short].sort_values("win_rate", ascending=False)
        if task_df.empty:
            continue
        print(f"\n--- {task_short} ---")
        for _, row in task_df.iterrows():
            print(
                f"  {row['model']:50s}  "
                f"win={row['win_rate']:.3f}  "
                f"({row['n_wins']}W/{row['n_ties']}T/{row['n_losses']}L, "
                f"n={row['n_samples']})"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarization quality via LLM-judge (Kirk et al., ICLR 2024).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Prepare batch files (dry-run first to check counts):\n"
            "  python scripts/compute_summarization_quality.py --prepare --dry-run\n"
            "  python scripts/compute_summarization_quality.py --prepare\n"
            "\n"
            "  # Score completed results:\n"
            "  python scripts/compute_summarization_quality.py --score "
            "--results-dir analysis/judge_results/summarization/\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Build batch JSONL files for the OpenAI Batch API",
    )
    mode.add_argument(
        "--score",
        action="store_true",
        help="Parse completed batch results and aggregate to CSV",
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root runs directory containing parquets (default: runs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/judge_batches/summarization"),
        help="Output directory for batch files or scored CSV (default: analysis/judge_batches/summarization/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing batch result JSONL files (required for --score)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be prepared without writing files",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prepare:
        prepare(args)
    elif args.score:
        score(args)


if __name__ == "__main__":
    main()
