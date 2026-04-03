#!/usr/bin/env python3
"""
Quality-filtered code diversity: UniXcoder SBERT + AST subtree diversity
on correct-only outputs.

For HumanEval and MBPP, correctness = passes test cases (sandboxed execution).
For CRUXEval, correctness = predicted output matches executed result.

Metrics computed on the correct subset only:
  - code_sbert: mean pairwise cosine distance (UniXcoder embeddings)
  - code_vendi: eigenvalue-entropy Vendi score (UniXcoder kernel)
  - ast_distinct: unique/total AST subtree ratio (height 4)
  - ast_pairwise: mean pairwise Jaccard distance on subtree multisets

Usage:
    # Single model + task (for HPC submission)
    python scripts/compute_code_diversity.py --task humaneval --model OLMo-3-7B-Think

    # All models for one task
    python scripts/compute_code_diversity.py --task humaneval

    # All tasks, all models
    python scripts/compute_code_diversity.py

    # Merge per-job results into combined CSVs
    python scripts/compute_code_diversity.py --merge
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.code_execution import (
    execute_code_with_tests,
    extract_code_from_markdown,
    prepare_humaneval_execution,
    prepare_mbpp_execution,
)
from src.evaluation.lighteval_diversity_metrics import _strip_cot

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default model name; on HPC use the safetensors-converted local path
# via --sbert-model or env UNIXCODER_PATH
UNIXCODER_MODEL = "microsoft/unixcoder-base"
BATCH_SIZE = 512
SEED = 42

CODE_TASKS = ["humaneval", "humaneval_instruct", "mbpp", "cruxeval"]

# experiment dirs where code tasks live
TASK_EXPERIMENTS = {
    "humaneval":          "extended",
    "humaneval_instruct": "extended",
    "mbpp":               "extended",
    "cruxeval":           "open_ended",
}

ALL_MODELS = [
    "OLMo-3-1025-7B",
    "OLMo-3-7B-RL-Zero-General",
    "OLMo-3-7B-RL-Zero-IF",
    "OLMo-3-7B-RL-Zero-Code",
    "OLMo-3.1-7B-RL-Zero-Code",
    "OLMo-3-7B-RL-Zero-Math",
    "OLMo-3.1-7B-RL-Zero-Math",
    "OLMo-3-7B-Instruct-SFT",
    "OLMo-3-7B-Instruct-DPO",
    "OLMo-3-7B-Instruct",
    "OLMo-3-7B-Think-SFT",
    "OLMo-3-7B-Think-DPO",
    "OLMo-3-7B-Think",
]


# ---------------------------------------------------------------------------
# Data loading  (mirrors quality_filtered_diversity.py)
# ---------------------------------------------------------------------------

def find_parquet(runs_dir: Path, experiment: str, model: str, task: str) -> Path | None:
    """Find parquet file for a model-task pair (walks nested details/ dirs)."""
    alt_name = model.replace("OLMo", "Olmo")
    candidates = [
        runs_dir / experiment / f"allenai__{model}" / "details",
        runs_dir / experiment / model / "details",
        runs_dir / experiment / alt_name / "details",
    ]
    for model_dir in candidates:
        if not model_dir.exists():
            continue
        for root, _dirs, files in os.walk(model_dir):
            for fn in files:
                if fn.endswith(".parquet") and f"{task}_diversity" in fn:
                    return Path(root) / fn
    return None


def find_data_file(runs_dir: Path, experiment: str, model: str, task: str) -> Path | None:
    """Find parquet or JSONL for a model-task pair. Parquets preferred."""
    pq = find_parquet(runs_dir, experiment, model, task)
    if pq is not None:
        return pq
    alt_name = model.replace("OLMo", "Olmo")
    for prefix in [f"allenai__{model}", model, alt_name]:
        jf = runs_dir / experiment / prefix / f"diversity_generations_{task}_diversity.jsonl"
        if jf.exists():
            return jf
        jf = runs_dir / experiment / prefix / "diversity_generations.jsonl"
        if jf.exists():
            return jf
    return None


def load_data(data_path: Path, task: str) -> list[dict]:
    """Load generations + doc metadata from parquet or JSONL.

    Returns list of dicts: {"gens": [...], "doc": {...}}
    """
    if data_path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(data_path)
        records = []
        for _, row in df.iterrows():
            gens = list(row["model_response"]["text"])
            doc = row["doc"]
            records.append({"gens": gens, "doc": doc})
        return records

    # JSONL
    records = []
    seen: set[str] = set()
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r_task = r.get("task", "").split("|")[0]
            if f"{task}_diversity" not in r_task:
                continue
            input_id = str(r.get("input_id", ""))
            if input_id in seen:
                continue
            seen.add(input_id)
            doc = {"query": r.get("prompt", ""), "choices": []}
            records.append({"gens": r.get("outputs", []), "doc": doc})
    return records


# ---------------------------------------------------------------------------
# Correctness checking (from quality_filtered_diversity.py)
# ---------------------------------------------------------------------------

_HUMANEVAL_INDEX = None
_MBPP_INDEX = None
_CRUXEVAL_INDEX = None


def _extract_code(text: str) -> str:
    """Extract executable code from a model generation (strip CoT + markdown)."""
    text = _strip_cot(text)
    blocks = re.findall(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text


def _get_humaneval_index():
    global _HUMANEVAL_INDEX
    if _HUMANEVAL_INDEX is None:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", "openai_humaneval", split="test")
        id_idx = {}
        prefix_idx = {}
        for row in ds:
            tid = int(row["task_id"].split("/")[1])
            info = {"prompt": row["prompt"], "test": row["test"],
                    "entry_point": row["entry_point"]}
            id_idx[tid] = info
            prefix_idx[info["prompt"][:200].strip()] = (tid, info)
        _HUMANEVAL_INDEX = (id_idx, prefix_idx)
    return _HUMANEVAL_INDEX


def _get_mbpp_index():
    global _MBPP_INDEX
    if _MBPP_INDEX is None:
        from datasets import load_dataset
        ds = load_dataset("mbpp", "full", split="test")
        idx = {}
        for row in ds:
            text = str(row["text"])
            prompt_key = f'"""\n{text}\n"""'[:200].strip()
            idx[prompt_key] = {
                "text": text, "test_list": list(row["test_list"]),
                "code": row["code"], "task_id": row["task_id"],
            }
        _MBPP_INDEX = idx
    return _MBPP_INDEX


def _get_cruxeval_index():
    global _CRUXEVAL_INDEX
    if _CRUXEVAL_INDEX is None:
        from datasets import load_dataset
        ds = load_dataset("cruxeval-org/cruxeval", split="test")
        idx = {}
        for row in ds:
            code = row.get("code", "")
            inp = row.get("input", "")
            output = row.get("output", "")
            # Build key from prompt prefix (matches how prompts are generated)
            prompt_snippet = code[:150].strip()
            idx[prompt_snippet] = {"code": code, "input": inp, "output": output}
        _CRUXEVAL_INDEX = idx
    return _CRUXEVAL_INDEX


def check_code_correctness(
    generations: list[str], doc: dict, task: str,
) -> list[bool]:
    """Check code correctness via sandboxed test execution or output matching."""
    prompt = doc.get("original_query") or doc.get("query", "")

    if task in ("humaneval", "humaneval_instruct"):
        _, prefix_idx = _get_humaneval_index()
        key = prompt[:200].strip()
        match = prefix_idx.get(key)
        if match is None:
            return [False] * len(generations)
        _, test_case = match
        entry_point = test_case["entry_point"]
        results = []
        for raw_gen in generations:
            code_text = _extract_code(raw_gen)
            if f"def {entry_point}" in code_text:
                full = f"{code_text}\n\n{test_case['test']}\n\ncheck({entry_point})\n"
            else:
                full = prepare_humaneval_execution(
                    prompt=test_case["prompt"], completion=code_text,
                    test_code=test_case["test"], entry_point=entry_point,
                )
            passed, _ = execute_code_with_tests(full, timeout=10.0)
            results.append(passed)
        return results

    elif task == "mbpp":
        mbpp_idx = _get_mbpp_index()
        key = prompt[:200].strip()
        match = mbpp_idx.get(key)
        if match is None:
            return [False] * len(generations)
        test_list = match["test_list"]
        expected_fn = None
        for t in test_list:
            fn_match = re.search(r"assert\s+(\w+)\s*\(", t)
            if fn_match:
                expected_fn = fn_match.group(1)
                break
        results = []
        for raw_gen in generations:
            code_text = _extract_code(raw_gen)
            if expected_fn:
                fn_match = re.search(r"^(def\s+)(\w+)(\s*\()", code_text, re.MULTILINE)
                if fn_match and fn_match.group(2) != expected_fn:
                    old_name = fn_match.group(2)
                    code_text = re.sub(
                        rf"\bdef\s+{re.escape(old_name)}\b",
                        f"def {expected_fn}", code_text,
                    )
                    code_text = re.sub(
                        rf"\b{re.escape(old_name)}\s*\(",
                        f"{expected_fn}(", code_text,
                    )
            full = prepare_mbpp_execution(
                generated_code=code_text, test_assertions=test_list,
            )
            passed, _ = execute_code_with_tests(full, timeout=10.0)
            results.append(passed)
        return results

    elif task == "cruxeval":
        # CRUXEval: model predicts output of f(input). Check via execution.
        gold_choices = doc.get("choices", [])
        gold = gold_choices[0].strip() if gold_choices else ""
        if not gold:
            return [False] * len(generations)
        results = []
        for raw_gen in generations:
            stripped = _strip_cot(raw_gen).strip()
            # Check if the gold output appears in the model response
            results.append(gold in stripped)
        return results

    return [False] * len(generations)


# ---------------------------------------------------------------------------
# UniXcoder SBERT + Vendi
# ---------------------------------------------------------------------------

def compute_sbert_diversity(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance (1 - cosine_sim) over upper triangle."""
    if len(embeddings) < 2:
        return float("nan")
    sim = embeddings @ embeddings.T
    n = len(sim)
    idx = np.triu_indices(n, k=1)
    return float(1.0 - np.mean(sim[idx]))


def compute_vendi_score(embeddings: np.ndarray) -> float:
    """Eigenvalue-entropy Vendi score on cosine similarity kernel."""
    if len(embeddings) < 2:
        return float("nan")
    K = embeddings @ embeddings.T
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total < 1e-12:
        return 1.0
    probs = eigvals / total
    probs = probs[probs > 1e-12]
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def compute_code_sbert(
    correct_outputs_per_prompt: list[list[str]],
    sbert_model,
) -> dict:
    """Compute UniXcoder SBERT diversity and Vendi score on correct outputs per prompt."""
    sbert_scores = []
    vendi_scores = []
    for outputs in correct_outputs_per_prompt:
        code_blocks = [extract_code_from_markdown(_strip_cot(o)) for o in outputs]
        code_blocks = [c for c in code_blocks if c.strip()]
        if len(code_blocks) < 2:
            continue
        emb = sbert_model.encode(code_blocks, batch_size=BATCH_SIZE, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        sbert_scores.append(compute_sbert_diversity(emb))
        vendi_scores.append(compute_vendi_score(emb))

    return {
        "code_sbert": float(np.mean(sbert_scores)) if sbert_scores else float("nan"),
        "code_vendi": float(np.mean(vendi_scores)) if vendi_scores else float("nan"),
        "n_prompts_sbert": len(sbert_scores),
    }


# ---------------------------------------------------------------------------
# AST Subtree Diversity
# ---------------------------------------------------------------------------

def _subtree_height(node: ast.AST, _depth: int = 0) -> int:
    """Compute height of an AST node (leaf = 1). Returns -1 if too deep."""
    if _depth > 500:
        return -1
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    child_heights = [_subtree_height(c, _depth + 1) for c in children]
    if -1 in child_heights:
        return -1
    return 1 + max(child_heights)


def _extract_subtrees(node: ast.AST, target_height: int, _depth: int = 0) -> list[str]:
    """Extract all subtrees of exactly target_height as canonical strings."""
    if _depth > 500:
        return []
    subtrees = []
    h = _subtree_height(node)
    if h < 0:
        return []
    if h == target_height:
        subtrees.append(ast.dump(node))
    if h > target_height:
        for child in ast.iter_child_nodes(node):
            subtrees.extend(_extract_subtrees(child, target_height, _depth + 1))
    return subtrees


def compute_ast_metrics(
    correct_outputs_per_prompt: list[list[str]],
    subtree_height: int = 4,
) -> dict:
    """Compute AST subtree diversity on correct outputs per prompt."""
    distinct_scores = []
    pairwise_scores = []
    total_outputs = 0
    parseable_outputs = 0

    for outputs in correct_outputs_per_prompt:
        code_blocks = [extract_code_from_markdown(_strip_cot(o)) for o in outputs]

        per_output_subtrees: list[Counter] = []
        for code in code_blocks:
            total_outputs += 1
            try:
                tree = ast.parse(code)
            except (SyntaxError, MemoryError, RecursionError):
                continue
            parseable_outputs += 1
            try:
                subs = _extract_subtrees(tree, subtree_height)
            except (RecursionError, MemoryError):
                continue
            if subs:
                per_output_subtrees.append(Counter(subs))

        if not per_output_subtrees:
            continue

        # ast_distinct: pool all subtrees, unique / total
        pooled = Counter()
        for c in per_output_subtrees:
            pooled.update(c)
        total_count = sum(pooled.values())
        unique_count = len(pooled)
        if total_count > 0:
            distinct_scores.append(unique_count / total_count)

        # ast_pairwise: mean Jaccard distance over pairs
        if len(per_output_subtrees) >= 2:
            jaccard_dists = []
            n = len(per_output_subtrees)
            for i in range(n):
                for j in range(i + 1, n):
                    keys = set(per_output_subtrees[i].keys()) | set(per_output_subtrees[j].keys())
                    if not keys:
                        continue
                    intersection = sum(
                        min(per_output_subtrees[i][k], per_output_subtrees[j][k])
                        for k in keys
                    )
                    union = sum(
                        max(per_output_subtrees[i][k], per_output_subtrees[j][k])
                        for k in keys
                    )
                    jaccard_dists.append(1.0 - intersection / union if union > 0 else 0.0)
            if jaccard_dists:
                pairwise_scores.append(float(np.mean(jaccard_dists)))

    parseability = parseable_outputs / total_outputs if total_outputs > 0 else 0.0

    return {
        "ast_distinct": float(np.mean(distinct_scores)) if distinct_scores else float("nan"),
        "ast_pairwise": float(np.mean(pairwise_scores)) if pairwise_scores else float("nan"),
        "parseability_rate": round(parseability, 6),
        "n_prompts_ast": len(distinct_scores),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_model_task(
    model: str,
    task: str,
    runs_dir: Path,
    sbert_model,
    skip_ast: bool = False,
) -> list[dict] | None:
    """Process one (model, task) pair with correctness filtering."""
    experiment = TASK_EXPERIMENTS[task]

    data_file = find_data_file(runs_dir, experiment, model, task)
    no_cot_file = find_data_file(runs_dir, f"no_cot_{experiment}", model, task)

    results = []
    for lbl, df_path in [(model, data_file), (f"no_cot__{model}", no_cot_file)]:
        if df_path is None:
            continue
        print(f"  Loading {lbl} × {task} from {df_path.name}...", flush=True)
        data = load_data(df_path, task)
        if not data:
            print(f"    No data found, skipping.", flush=True)
            continue

        n_prompts = len(data)
        k_per_prompt = len(data[0]["gens"])
        print(f"    {n_prompts} prompts, {k_per_prompt} outputs each", flush=True)

        # Correctness check
        print(f"    Checking correctness...", flush=True)
        correct_outputs_per_prompt: list[list[str]] = []
        total_correct = 0
        n_computable = 0
        for entry in data:
            gens = entry["gens"]
            doc = entry["doc"]
            correct = check_code_correctness(gens, doc, task)
            correct_texts = [gens[j] for j in range(len(gens)) if correct[j]]
            total_correct += len(correct_texts)
            if len(correct_texts) >= 2:
                correct_outputs_per_prompt.append(correct_texts)
                n_computable += 1

        mean_k = total_correct / n_prompts if n_prompts > 0 else 0
        print(f"    mean_k_correct={mean_k:.2f}, prompts_with_>=2_correct={n_computable}/{n_prompts}", flush=True)

        if not correct_outputs_per_prompt:
            print(f"    No prompts with >=2 correct outputs, skipping diversity.", flush=True)
            row = {"model": lbl, "task": task, "mean_k_correct": round(mean_k, 4),
                   "n_computable": n_computable, "n_total": n_prompts}
            results.append(row)
            continue

        # UniXcoder SBERT on correct-only
        print(f"    Computing UniXcoder SBERT on correct outputs...", flush=True)
        sbert_result = compute_code_sbert(correct_outputs_per_prompt, sbert_model)

        # AST metrics on correct-only
        ast_result = {}
        if not skip_ast and task in ("humaneval", "humaneval_instruct", "mbpp"):
            # AST only makes sense for code-generation tasks, not CRUXEval (output prediction)
            print(f"    Computing AST subtree diversity on correct outputs...", flush=True)
            ast_result = compute_ast_metrics(correct_outputs_per_prompt)

        row = {"model": lbl, "task": task, "mean_k_correct": round(mean_k, 4),
               "n_computable": n_computable, "n_total": n_prompts}
        row.update(sbert_result)
        row.update(ast_result)
        results.append(row)

    return results if results else None


def _safe_model_slug(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def merge_results(output_dir: Path) -> None:
    """Merge per-job JSON results into combined CSVs."""
    import pandas as pd

    parts_dir = output_dir / "parts"
    if not parts_dir.exists():
        print(f"No parts directory found at {parts_dir}", flush=True)
        sys.exit(1)

    all_rows = []
    for jf in sorted(parts_dir.glob("*.json")):
        with jf.open() as f:
            all_rows.extend(json.load(f))

    if not all_rows:
        print("No results found in parts/", flush=True)
        sys.exit(1)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(output_dir / "code_diversity_all.csv", index=False)
    print(f"Merged {len(all_rows)} rows from {len(list(parts_dir.glob('*.json')))} files", flush=True)
    print(f"  -> {output_dir / 'code_diversity_all.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Quality-filtered code diversity metrics")
    parser.add_argument("--task", choices=CODE_TASKS, help="Single task to process")
    parser.add_argument("--model", help="Single model to process")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs",
                        help="Root directory containing run outputs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis" / "code_diversity",
                        help="Output directory for CSVs")
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--no-ast", action="store_true", help="Skip AST metrics")
    parser.add_argument("--sbert-model", default=None,
                        help="Path or name for UniXcoder model (default: env UNIXCODER_PATH or microsoft/unixcoder-base)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-job JSON results into combined CSVs (no computation)")
    args = parser.parse_args()

    if args.merge:
        merge_results(args.output_dir)
        return

    tasks = [args.task] if args.task else CODE_TASKS
    models = [args.model] if args.model else ALL_MODELS

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load UniXcoder
    from sentence_transformers import SentenceTransformer
    model_path = args.sbert_model or os.environ.get("UNIXCODER_PATH", UNIXCODER_MODEL)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading UniXcoder model: {model_path}", flush=True)
    sbert_model = SentenceTransformer(model_path, device=device)
    sbert_model.max_seq_length = 512
    print(f"  Device: {device}, max_seq_length: {sbert_model.max_seq_length}", flush=True)

    all_rows: list[dict] = []

    for task in tasks:
        print(f"\n=== Task: {task} ===", flush=True)
        for model in models:
            print(f"\nModel: {model}", flush=True)
            result = process_model_task(
                model, task, args.runs_dir, sbert_model, skip_ast=args.no_ast,
            )
            if result:
                all_rows.extend(result)

    if not all_rows:
        print("No results computed.", flush=True)
        sys.exit(1)

    # Write per-job JSON
    parts_dir = args.output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{args.task or 'all'}_{_safe_model_slug(args.model or 'all')}"
    part_path = parts_dir / f"{slug}.json"
    with part_path.open("w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nWrote {part_path} ({len(all_rows)} rows)", flush=True)

    if not args.task and not args.model:
        merge_results(args.output_dir)


if __name__ == "__main__":
    main()
