#!/usr/bin/env python3
"""
Quality-filtered diversity: compute diversity metrics on correct-only outputs.

For verifiable tasks (math, code, IFEval), labels each of K=16 generations as
correct/incorrect, then computes SBERT diversity and Vendi score on the
correct-only subset.

Usage:
    # Single model + task (for HPC submission)
    python scripts/quality_filtered_diversity.py --task gsm8k --model OLMo-3-7B-Think

    # All models for one task
    python scripts/quality_filtered_diversity.py --task gsm8k

    # All tasks, all models
    python scripts/quality_filtered_diversity.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.math_answer_extraction import (
    answers_match,
    extract_generated_answer,
    extract_gsm8k_gold_answer,
    normalize_number,
    strip_think_tags,
)
from src.evaluation.code_execution import (
    execute_code_with_tests,
    extract_code_from_markdown,
    prepare_humaneval_execution,
    prepare_mbpp_execution,
)
from src.evaluation.ifeval_constraints import verify_constraints

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
SBERT_MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 512
SEED = 42

# Task definitions: task_name -> (experiment_dir, correctness_function_name)
VERIFIABLE_TASKS = {
    "gsm8k":         ("open_ended", "math"),
    "math_algebra":  ("open_ended", "math"),
    "math_geometry": ("open_ended", "math"),
    "truthfulqa":    ("open_ended", "truthfulqa"),
    "humaneval":     ("extended",   "code"),
    "humaneval_instruct": ("extended", "code"),
    "mbpp":          ("extended",   "code"),
    "ifeval":        ("extended",   "ifeval"),
    "cruxeval":      ("open_ended", "cruxeval"),
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
# Correctness checkers
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} in MATH-style gold solutions."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return normalize_number(match.group(1).strip())
    return None


_MATH_GOLD_INDEXES: dict[str, dict] = {}


def _get_math_gold_index(task: str) -> dict[str, str]:
    """Lazy-load HF gold answer index for math tasks (for JSONL fallback)."""
    if task in _MATH_GOLD_INDEXES:
        return _MATH_GOLD_INDEXES[task]
    from datasets import load_dataset
    if task == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        q_field, a_field, extractor = "question", "answer", extract_gsm8k_gold_answer
    elif task == "math_algebra":
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
        q_field, a_field, extractor = "problem", "solution", extract_boxed_answer
    elif task == "math_geometry":
        ds = load_dataset("EleutherAI/hendrycks_math", "geometry", split="test")
        q_field, a_field, extractor = "problem", "solution", extract_boxed_answer
    else:
        _MATH_GOLD_INDEXES[task] = {}
        return {}
    index = {}
    for row in ds:
        q = str(row[q_field])
        gold = extractor(str(row[a_field]))
        if gold is not None:
            index[q[:200].strip()] = gold
    _MATH_GOLD_INDEXES[task] = index
    return index


def _extract_question_from_prompt(prompt: str) -> str:
    """Extract question from formatted prompt (strip Problem:/Question: markers)."""
    for marker in ["Problem:", "Question:", "Q:"]:
        pos = prompt.rfind(marker)
        if pos >= 0:
            text = prompt[pos + len(marker):]
            for end in ["Solution:", "Answer:", "A:", "\n\n"]:
                epos = text.rfind(end)
                if epos >= 0:
                    text = text[:epos]
            return text.strip()
    return prompt.strip()


def check_math_correctness(
    generations: list[str], gold_text: str, task: str = "gsm8k",
    prompt: str = "",
) -> list[bool]:
    """Check correctness for GSM8K / MATH tasks."""
    if task == "gsm8k":
        gold_answer = extract_gsm8k_gold_answer(gold_text)
    else:
        gold_answer = extract_boxed_answer(gold_text)

    # Fallback: fetch gold from HF dataset via prompt matching (JSONL input)
    if gold_answer is None and prompt:
        index = _get_math_gold_index(task)
        question = _extract_question_from_prompt(prompt)
        gold_answer = index.get(question[:200].strip())

    if gold_answer is None:
        return [False] * len(generations)
    results = []
    for gen in generations:
        stripped = strip_think_tags(gen)
        pred = extract_generated_answer(stripped)
        results.append(answers_match(pred, gold_answer))
    return results


def check_truthfulqa_correctness(
    generations: list[str], doc: dict,
) -> list[bool]:
    """Check correctness for TruthfulQA (mc2-style: check against reference answers)."""
    # TruthfulQA has multiple correct answers in choices
    correct_answers = doc.get("correct_answers", [])
    if not correct_answers:
        # Fall back to choices[0] as the single correct answer
        choices = doc.get("choices", [])
        if choices:
            correct_answers = [choices[0]]
    if not correct_answers:
        return [False] * len(generations)

    results = []
    for gen in generations:
        stripped = strip_think_tags(gen).strip().lower()
        is_correct = any(
            ca.strip().lower() in stripped or stripped in ca.strip().lower()
            for ca in correct_answers
        )
        results.append(is_correct)
    return results


def _strip_cot(text: str) -> str:
    """Strip chain-of-thought <think> tags from model output."""
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


def _extract_code_from_generation(text: str) -> str:
    """Extract executable code from a model generation."""
    text = _strip_cot(text)
    return extract_code_from_markdown(text)


# Lazily-loaded indexes for code tasks
_HUMANEVAL_INDEX = None
_MBPP_INDEX = None


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


def check_code_correctness(
    generations: list[str], doc: dict, task: str = "humaneval",
) -> list[bool]:
    """Check code correctness via sandboxed test execution."""
    prompt = doc.get("original_query") or doc.get("query", "")
    results = []

    if task in ("humaneval", "humaneval_instruct"):
        _, prefix_idx = _get_humaneval_index()
        key = prompt[:200].strip()
        match = prefix_idx.get(key)
        if match is None:
            return [False] * len(generations)
        _, test_case = match
        entry_point = test_case["entry_point"]

        for raw_gen in generations:
            code_text = _extract_code_from_generation(raw_gen)
            if f"def {entry_point}" in code_text:
                full = f"{code_text}\n\n{test_case['test']}\n\ncheck({entry_point})\n"
            else:
                full = prepare_humaneval_execution(
                    prompt=test_case["prompt"], completion=code_text,
                    test_code=test_case["test"], entry_point=entry_point,
                )
            passed, _ = execute_code_with_tests(full, timeout=10.0)
            results.append(passed)

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

        for raw_gen in generations:
            code_text = _extract_code_from_generation(raw_gen)
            # Rename function if needed
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
    else:
        return [False] * len(generations)

    return results


_IFEVAL_INDEX = None


def _get_ifeval_index():
    """Build prompt prefix -> {instruction_id_list, kwargs_list} from HuggingFace."""
    global _IFEVAL_INDEX
    if _IFEVAL_INDEX is None:
        import json as _json
        from datasets import load_dataset
        ds = load_dataset("google/IFEval", "default", split="train")
        index = {}
        for row in ds:
            prompt = str(row["prompt"])
            key = prompt[:200].strip()
            instruction_id_list = list(row["instruction_id_list"])
            kwargs_list = []
            for kw in row["kwargs"]:
                if isinstance(kw, str):
                    try:
                        kwargs_list.append(_json.loads(kw))
                    except (ValueError, _json.JSONDecodeError):
                        kwargs_list.append({})
                elif isinstance(kw, dict):
                    kwargs_list.append(kw)
                else:
                    kwargs_list.append({})
            index[key] = {
                "instruction_id_list": instruction_id_list,
                "kwargs_list": kwargs_list,
            }
        _IFEVAL_INDEX = index
    return _IFEVAL_INDEX


def check_ifeval_correctness(generations: list[str], doc: dict) -> list[bool]:
    """Check IFEval strict constraint satisfaction per generation."""
    ifeval_index = _get_ifeval_index()
    query = doc.get("query", "")
    key = query[:200].strip()
    constraints = ifeval_index.get(key)
    if constraints is None:
        return [False] * len(generations)
    instruction_id_list = constraints["instruction_id_list"]
    kwargs_list = constraints["kwargs_list"]
    results = []
    for gen in generations:
        stripped = strip_think_tags(gen)
        try:
            strict_pass, _ = verify_constraints(stripped, instruction_id_list, kwargs_list)
            results.append(strict_pass)
        except Exception:
            results.append(False)
    return results


# --- CruxEval correctness ---

_CRUXEVAL_GOLD_INDEX: dict[str, str] | None = None


def _get_cruxeval_gold_index() -> dict[str, str]:
    """Lazy-load CruxEval gold answers from HuggingFace."""
    global _CRUXEVAL_GOLD_INDEX
    if _CRUXEVAL_GOLD_INDEX is not None:
        return _CRUXEVAL_GOLD_INDEX
    from datasets import load_dataset
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    index = {}
    for row in ds:
        code, inp, output = str(row["code"]), str(row["input"]), str(row["output"])
        key = f"Given the following Python function:\n\n```python\n{code}\n```\n\nWhat is the output of `f({inp})`?"[:200].strip()
        index[key] = output
    _CRUXEVAL_GOLD_INDEX = index
    return index


def _extract_cruxeval_pred(text: str) -> str | None:
    """Extract predicted output from a CruxEval generation."""
    text = strip_think_tags(text)
    if not text.strip():
        return None
    # **Answer:** block
    m = re.search(r"\*\*Answer[:\s]*\*\*\s*\n\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    if m:
        return m.group(1).strip().strip("`").strip()
    # Answer:/Output: inline
    m = re.search(r"(?:Answer|Output)\s*:\s*\n?\s*`?(.+?)`?\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Last code block
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Last backtick with Python-like value
    ticks = re.findall(r"`([^`]+)`", text)
    for t in reversed(ticks):
        t = t.strip()
        if re.match(r"^[\[\({'\"-]|^\d|^True|^False|^None", t):
            return t
    # First non-empty line (base model)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[0] if lines else None


def _normalize_crux(text: str) -> str:
    """Normalize CruxEval answer for comparison."""
    if not text:
        return ""
    text = text.strip().rstrip(".:;,")
    for q in ['"""', "'''", '"', "'"]:
        if text.startswith(q) and text.endswith(q):
            text = text[len(q):-len(q)]
            break
    return " ".join(text.strip("`").split())


def _crux_match(pred: str, gold: str) -> bool:
    """Check CruxEval answer match (string + ast.literal_eval fallback)."""
    import ast as _ast
    pn, gn = _normalize_crux(pred), _normalize_crux(gold)
    if not pn:
        return False
    if pn == gn:
        return True
    try:
        return _ast.literal_eval(pn) == _ast.literal_eval(gn)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False


def check_cruxeval_correctness(
    generations: list[str], doc: dict, prompt: str = "",
) -> list[bool]:
    """Check CruxEval output-prediction correctness per generation."""
    # Get gold from doc
    gold = None
    choices = doc.get("choices", [])
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    if choices:
        gold = str(choices[0]).strip()
    # Fallback: HF dataset
    if not gold:
        index = _get_cruxeval_gold_index()
        gold = index.get(prompt[:200].strip())
    if not gold:
        return [False] * len(generations)
    results = []
    for gen in generations:
        pred = _extract_cruxeval_pred(gen)
        results.append(_crux_match(pred, gold) if pred else False)
    return results


# ---------------------------------------------------------------------------
# Diversity computation
# ---------------------------------------------------------------------------

def compute_sbert_diversity(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return float("nan")
    sim = embeddings @ embeddings.T
    n = len(sim)
    idx = np.triu_indices(n, k=1)
    return float(1.0 - np.mean(sim[idx]))


def compute_vendi_score(embeddings: np.ndarray) -> float:
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


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def find_parquet(runs_dir: Path, experiment: str, model: str, task: str) -> Path | None:
    """Find the parquet file for a given model and task.

    Searches both allenai__ prefixed dirs (via lighteval)
    and plain model name dirs (alternate naming).
    """
    # Handle naming variants: allenai__OLMo, OLMo, Olmo
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
            for f in files:
                if f"details_{task}_diversity" in f and f.endswith(".parquet"):
                    return Path(root) / f
    return None


def _load_data(data_path: Path, task: str) -> list[dict]:
    """Load generation data from parquet or JSONL into a uniform format.

    Returns list of dicts with: gens (list[str]), doc (dict with query, choices),
    gold_text (str).
    """
    import json as _json

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
        records = []
        for _, row in df.iterrows():
            doc = row["doc"]
            gens = list(row["model_response"]["text"])
            gold_text = doc.get("choices", [""])[0] if doc.get("choices") else ""
            records.append({"gens": gens, "doc": doc, "gold_text": gold_text})
        return records

    elif data_path.suffix == ".jsonl":
        records = []
        seen = set()
        with data_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = _json.loads(line)
                r_task = r.get("task", "").split("|")[0]
                # Filter to target task (JSONL may contain multiple tasks)
                if f"{task}_diversity" not in r_task:
                    continue
                input_id = str(r.get("input_id", ""))
                if input_id in seen:
                    continue
                seen.add(input_id)
                doc = {"query": r.get("prompt", ""), "choices": []}
                records.append({
                    "gens": r.get("outputs", []),
                    "doc": doc,
                    "gold_text": "",  # No gold in JSONL — fetched by checkers via HF
                })
        return records

    raise ValueError(f"Unsupported file format: {data_path.suffix}")


def find_data_file(runs_dir: Path, experiment: str, model: str, task: str) -> Path | None:
    """Find parquet or JSONL for a model-task pair. Parquets preferred."""
    pq = find_parquet(runs_dir, experiment, model, task)
    if pq is not None:
        return pq

    # Try JSONL (task-specific naming)
    alt_name = model.replace("OLMo", "Olmo")
    for prefix in [f"allenai__{model}", model, alt_name]:
        jf = runs_dir / experiment / prefix / f"diversity_generations_{task}_diversity.jsonl"
        if jf.exists():
            return jf
        # Generic JSONL (multi-task)
        jf = runs_dir / experiment / prefix / "diversity_generations.jsonl"
        if jf.exists():
            return jf
    return None


def process_single(
    model: str,
    task: str,
    data_path: Path,
    sbert_model: SentenceTransformer,
    task_type: str,
) -> tuple[dict, pd.DataFrame]:
    """Process one model-task pair from parquet or JSONL."""
    data = _load_data(data_path, task)
    n_prompts = len(data)

    prompt_data = []
    for i in range(n_prompts):
        entry = data[i]
        gens = entry["gens"]
        doc = entry["doc"]
        gold_text = entry["gold_text"]
        stripped = [_strip_cot(g) for g in gens]

        if task_type == "math":
            correct = check_math_correctness(gens, gold_text, task=task, prompt=doc.get("query", ""))
        elif task_type == "truthfulqa":
            correct = check_truthfulqa_correctness(gens, doc)
        elif task_type == "code":
            correct = check_code_correctness(gens, doc, task=task)
        elif task_type == "ifeval":
            correct = check_ifeval_correctness(gens, doc)
        elif task_type == "cruxeval":
            correct = check_cruxeval_correctness(gens, doc, prompt=doc.get("query", ""))
        else:
            correct = [False] * len(gens)

        correct_texts = [stripped[j] for j in range(len(stripped)) if correct[j]]
        prompt_data.append({
            "idx": i,
            "k_correct": sum(correct),
            "acc1": 1.0 if correct[0] else 0.0,
            "stripped": stripped,
            "correct_texts": correct_texts,
        })

    # Bulk encode
    all_texts = []
    ranges = []
    for pd_entry in prompt_data:
        a_start = len(all_texts)
        all_texts.extend(pd_entry["stripped"])
        a_end = len(all_texts)
        c_start = len(all_texts)
        all_texts.extend(pd_entry["correct_texts"])
        c_end = len(all_texts)
        ranges.append((a_start, a_end, c_start, c_end))

    print(f"  Encoding {len(all_texts)} texts...", flush=True)
    if all_texts:
        embs = sbert_model.encode(
            all_texts, batch_size=BATCH_SIZE,
            show_progress_bar=False, normalize_embeddings=True,
        )
    else:
        embs = np.array([])

    # Per-prompt metrics
    records = []
    d_all_vals, d_corr_vals, v_corr_vals = [], [], []

    for i, pd_entry in enumerate(prompt_data):
        a_s, a_e, c_s, c_e = ranges[i]
        d_all = compute_sbert_diversity(embs[a_s:a_e])
        d_all_vals.append(d_all)

        k_c = pd_entry["k_correct"]
        if k_c >= 2:
            d_corr = compute_sbert_diversity(embs[c_s:c_e])
            v_corr = compute_vendi_score(embs[c_s:c_e])
            d_corr_vals.append(d_corr)
            v_corr_vals.append(v_corr)
        else:
            d_corr = float("nan")
            v_corr = float("nan")

        records.append({
            "model": model, "task": task, "prompt_idx": pd_entry["idx"],
            "k_correct": k_c, "acc1": pd_entry["acc1"],
            "d_all_sbert": d_all, "d_correct_sbert": d_corr,
            "d_correct_vendi": v_corr,
        })

    n_comp = len(d_corr_vals)
    summary = {
        "model": model, "task": task,
        "accuracy_at_1": np.mean([p["acc1"] for p in prompt_data]),
        "mean_k_correct": np.mean([p["k_correct"] for p in prompt_data]),
        "d_all_sbert": np.nanmean(d_all_vals),
        "d_correct_sbert": np.mean(d_corr_vals) if d_corr_vals else float("nan"),
        "d_correct_vendi": np.mean(v_corr_vals) if v_corr_vals else float("nan"),
        "n_computable": n_comp, "n_total": len(prompt_data),
        "frac_computable": n_comp / len(prompt_data) if prompt_data else 0.0,
    }
    return summary, pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Quality-filtered diversity analysis")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (gsm8k, math_algebra, etc.). Default: all.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. OLMo-3-7B-Think). Default: all.")
    parser.add_argument("--runs-dir", type=str, default=str(PROJECT_ROOT / "runs"),
                        help="Runs directory containing parquets.")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "analysis" / "quality_filtered_diversity"),
                        help="Output directory for CSVs.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [args.task] if args.task else list(VERIFIABLE_TASKS.keys())
    models = [args.model] if args.model else ALL_MODELS

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading SBERT model: {SBERT_MODEL_NAME}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    print(f"  Device: {device}", flush=True)

    all_summaries = []
    all_details = []

    for task in tasks:
        if task not in VERIFIABLE_TASKS:
            print(f"SKIP unknown task: {task}", flush=True)
            continue
        experiment, task_type = VERIFIABLE_TASKS[task]

        for model in models:
            # Search standard experiment dir
            data_file = find_data_file(runs_dir, experiment, model, task)
            # Also search no_cot_ experiment dir
            no_cot_exp = f"no_cot_{experiment}"
            no_cot_file = find_data_file(runs_dir, no_cot_exp, model, task)

            for label, df_path in [(model, data_file), (f"no_cot__{model}", no_cot_file)]:
                if df_path is None:
                    continue
                print(f"Processing {label} × {task} ({df_path.suffix})...", flush=True)
                summary, details = process_single(label, task, df_path, sbert_model, task_type)
                all_summaries.append(summary)
                all_details.append(details)

                s = summary
                d_c = f"{s['d_correct_sbert']:.4f}" if not np.isnan(s["d_correct_sbert"]) else "N/A"
                v_c = f"{s['d_correct_vendi']:.2f}" if not np.isnan(s["d_correct_vendi"]) else "N/A"
                print(
                    f"  acc@1={s['accuracy_at_1']:.3f} "
                    f"K_corr={s['mean_k_correct']:.1f} "
                    f"D_all={s['d_all_sbert']:.4f} "
                    f"D_corr={d_c} "
                    f"V_corr={v_c} "
                    f"n={s['n_computable']}/{s['n_total']}",
                    flush=True,
                )

    if all_summaries:
        sdf = pd.DataFrame(all_summaries)
        out_file = out_dir / "qfd_summary.csv"
        sdf.to_csv(out_file, index=False)
        print(f"\nSummary saved: {out_file}", flush=True)

    if all_details:
        ddf = pd.concat(all_details, ignore_index=True)
        out_file = out_dir / "qfd_per_prompt.csv"
        ddf.to_csv(out_file, index=False)
        print(f"Per-prompt saved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
