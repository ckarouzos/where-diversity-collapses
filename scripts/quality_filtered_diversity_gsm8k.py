#!/usr/bin/env python3
"""
Quality-filtered diversity on GSM8K.

For each model's GSM8K generations (16 per prompt × 500 prompts):
  1. Extract gold answer from doc['choices'][0] (#### N format)
  2. For each generation, strip CoT think tags, extract final answer
  3. Label correct/incorrect by comparing to gold
  4. Compute SBERT diversity on the correct-only subset (K_correct >= 2)

Outputs a table with:
  model, accuracy@1, mean_k_correct, d_all_sbert, d_correct_sbert,
  d_correct_vendi, n_computable_prompts

Also saves per-prompt detail CSV for downstream analysis.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.math_answer_extraction import (
    answers_match,
    extract_generated_answer,
    extract_gsm8k_gold_answer,
    strip_think_tags,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUNS_DIR = PROJECT_ROOT / "runs" / "open_ended"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "quality_filtered_diversity"
SBERT_MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 256  # for SBERT encoding
SEED = 42

# Canonical model ordering: base → RL-Zero → SFT → Instruct → DPO → Think
MODEL_ORDER = [
    "OLMo-3-1025-7B",
    "OLMo-3-7B-RL-Zero-General",
    "OLMo-3-7B-RL-Zero-IF",
    "OLMo-3-7B-RL-Zero-Code",
    "OLMo-3.1-7B-RL-Zero-Code",
    "OLMo-3-7B-RL-Zero-Math",
    "OLMo-3.1-7B-RL-Zero-Math",
    "OLMo-3-7B-Instruct-SFT",
    "OLMo-3-7B-Instruct",
    "OLMo-3-7B-Instruct-DPO",
    "OLMo-3-7B-Think-SFT",
    "OLMo-3-7B-Think",
    "OLMo-3-7B-Think-DPO",
]


def find_gsm8k_parquets() -> dict[str, Path]:
    """Discover GSM8K parquet files for each model.

    Uses os.walk instead of glob because filenames contain pipe characters
    which glob doesn't handle correctly.
    """
    import os

    model_files: dict[str, Path] = {}
    for model_dir_name in os.listdir(RUNS_DIR):
        if not model_dir_name.startswith("allenai__"):
            continue
        details_dir = RUNS_DIR / model_dir_name / "details"
        if not details_dir.exists():
            continue
        for root, _dirs, files in os.walk(details_dir):
            for f in files:
                if "gsm8k" in f and f.endswith(".parquet"):
                    model_name = model_dir_name.replace("allenai__", "")
                    model_files[model_name] = Path(root) / f
                    break
    return model_files


def compute_sbert_diversity(embeddings: np.ndarray) -> float:
    """Compute SBERT diversity = 1 - mean(pairwise cosine similarity).

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) L2-normalized embeddings.

    Returns
    -------
    float
        Diversity score in [0, 1]. Higher = more diverse.
    """
    if len(embeddings) < 2:
        return float("nan")
    # Cosine similarity matrix (embeddings are already L2-normalized by sentence-transformers)
    sim_matrix = embeddings @ embeddings.T
    # Extract upper triangle (excluding diagonal)
    n = len(sim_matrix)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    return float(1.0 - np.mean(pairwise_sims))


def compute_vendi_score(embeddings: np.ndarray) -> float:
    """Compute Vendi score = exp(entropy of normalized eigenvalues of the kernel matrix).

    Uses SBERT cosine similarity as the kernel.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) L2-normalized embeddings.

    Returns
    -------
    float
        Vendi score (effective number of modes). >= 1.
    """
    if len(embeddings) < 2:
        return float("nan")
    # Kernel matrix (cosine similarity, which is PSD for normalized vectors)
    K = embeddings @ embeddings.T
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(K)
    # Normalize to sum to 1 (these are like probabilities)
    eigenvalues = np.clip(eigenvalues, 0, None)  # numerical stability
    total = eigenvalues.sum()
    if total < 1e-12:
        return 1.0
    probs = eigenvalues / total
    # Shannon entropy, then exp
    probs = probs[probs > 1e-12]
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def process_model(
    model_name: str,
    parquet_path: Path,
    sbert_model: SentenceTransformer,
) -> tuple[dict, pd.DataFrame]:
    """Process one model's GSM8K parquet file.

    Returns
    -------
    summary : dict
        Aggregate metrics for this model.
    per_prompt : pd.DataFrame
        Per-prompt detail rows.
    """
    df = pd.read_parquet(parquet_path)
    n_prompts = len(df)

    per_prompt_records = []

    prompt_data: list[dict] = []

    for prompt_idx in range(n_prompts):
        row = df.iloc[prompt_idx]
        generations = list(row["model_response"]["text"])
        gold_text = row["doc"]["choices"][0]
        gold_answer = extract_gsm8k_gold_answer(gold_text)

        if gold_answer is None:
            # Skip malformed gold answers
            continue

        # Strip think tags for answer extraction and diversity
        stripped_gens = [strip_think_tags(g) for g in generations]

        # Extract answers and check correctness
        correctness = []
        for gen in stripped_gens:
            pred = extract_generated_answer(gen)
            correctness.append(answers_match(pred, gold_answer))

        k_correct = sum(correctness)
        accuracy_at_1 = 1.0 if correctness[0] else 0.0

        # Collect texts for embedding
        correct_texts = [stripped_gens[i] for i in range(len(stripped_gens)) if correctness[i]]

        prompt_data.append({
            "prompt_idx": prompt_idx,
            "k_correct": k_correct,
            "accuracy_at_1": accuracy_at_1,
            "n_gens": len(stripped_gens),
            "stripped_gens": stripped_gens,
            "correct_texts": correct_texts,
            "correct_indices": [i for i, c in enumerate(correctness) if c],
        })

    # Compute SBERT diversity per-prompt (avoids cross-prompt padding waste).
    # Truncate texts to 1500 chars (~384 tokens, SBERT's max_seq_length).
    MAX_CHAR_LEN = 1500

    d_all_sbert_vals = []
    d_correct_sbert_vals = []
    d_correct_vendi_vals = []
    accuracy_at_1_vals = []
    k_correct_vals = []

    n_total = len(prompt_data)
    print(f"  Encoding {n_total} prompts × 16 gens for {model_name}...")

    for i, pd_entry in enumerate(prompt_data):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_total} prompts done...")

        # Encode all 16 generations for this prompt
        texts = [t[:MAX_CHAR_LEN] for t in pd_entry["stripped_gens"]]
        all_emb = sbert_model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # D_all SBERT (all 16 generations)
        d_all = compute_sbert_diversity(all_emb)
        d_all_sbert_vals.append(d_all)

        k_correct = pd_entry["k_correct"]
        k_correct_vals.append(k_correct)
        accuracy_at_1_vals.append(pd_entry["accuracy_at_1"])

        # D_correct SBERT (correct subset)
        if k_correct >= 2:
            correct_indices = pd_entry["correct_indices"]
            correct_emb = all_emb[correct_indices]
            d_corr_sbert = compute_sbert_diversity(correct_emb)
            d_corr_vendi = compute_vendi_score(correct_emb)
            d_correct_sbert_vals.append(d_corr_sbert)
            d_correct_vendi_vals.append(d_corr_vendi)
        else:
            d_corr_sbert = float("nan")
            d_corr_vendi = float("nan")

        per_prompt_records.append({
            "model": model_name,
            "prompt_idx": pd_entry["prompt_idx"],
            "k_correct": k_correct,
            "accuracy_at_1": pd_entry["accuracy_at_1"],
            "d_all_sbert": d_all,
            "d_correct_sbert": d_corr_sbert,
            "d_correct_vendi": d_corr_vendi,
            "n_gens": pd_entry["n_gens"],
        })

    n_computable = len(d_correct_sbert_vals)

    summary = {
        "model": model_name,
        "accuracy_at_1": np.mean(accuracy_at_1_vals) if accuracy_at_1_vals else float("nan"),
        "mean_k_correct": np.mean(k_correct_vals) if k_correct_vals else float("nan"),
        "d_all_sbert": np.mean(d_all_sbert_vals) if d_all_sbert_vals else float("nan"),
        "d_correct_sbert": np.mean(d_correct_sbert_vals) if d_correct_sbert_vals else float("nan"),
        "d_correct_vendi": np.mean(d_correct_vendi_vals) if d_correct_vendi_vals else float("nan"),
        "n_computable_prompts": n_computable,
        "n_prompts_total": len(prompt_data),
        "frac_computable": n_computable / len(prompt_data) if prompt_data else 0.0,
    }

    per_prompt_df = pd.DataFrame(per_prompt_records)
    return summary, per_prompt_df


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Discover files
    model_files = find_gsm8k_parquets()
    print(f"Found GSM8K parquet files for {len(model_files)} models:")
    for name in sorted(model_files.keys()):
        print(f"  {name}")
    print()

    # Load SBERT model
    print(f"Loading SBERT model: {SBERT_MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    print(f"  Device: {device}")
    print()

    # Process each model
    summaries = []
    all_per_prompt = []

    for model_name in MODEL_ORDER:
        if model_name not in model_files:
            print(f"  SKIP {model_name} (no parquet found)")
            continue
        print(f"Processing {model_name}...")
        summary, per_prompt_df = process_model(model_name, model_files[model_name], sbert_model)
        summaries.append(summary)
        all_per_prompt.append(per_prompt_df)
        # Print incremental result
        s = summary
        print(
            f"  acc@1={s['accuracy_at_1']:.3f}  "
            f"mean_k={s['mean_k_correct']:.1f}  "
            f"D_all={s['d_all_sbert']:.4f}  "
            f"D_corr={s['d_correct_sbert']:.4f}  "
            f"Vendi_corr={s['d_correct_vendi']:.2f}  "
            f"n_comp={s['n_computable_prompts']}/{s['n_prompts_total']}"
        )
        print()

    # Build summary table
    summary_df = pd.DataFrame(summaries)

    # Print final table
    print("=" * 120)
    print("QUALITY-FILTERED DIVERSITY ON GSM8K (500 prompts × 16 generations)")
    print("=" * 120)
    print()

    header = (
        f"{'Model':<30s}  {'Acc@1':>6s}  {'MeanK':>6s}  "
        f"{'D_all':>8s}  {'D_corr':>8s}  {'V_corr':>8s}  "
        f"{'N_comp':>7s}  {'Frac':>6s}"
    )
    print(header)
    print("-" * len(header))

    for _, row in summary_df.iterrows():
        d_corr = f"{row['d_correct_sbert']:.4f}" if not np.isnan(row["d_correct_sbert"]) else "  N/A "
        v_corr = f"{row['d_correct_vendi']:.2f}" if not np.isnan(row["d_correct_vendi"]) else "  N/A "
        line = (
            f"{row['model']:<30s}  "
            f"{row['accuracy_at_1']:>6.3f}  "
            f"{row['mean_k_correct']:>6.1f}  "
            f"{row['d_all_sbert']:>8.4f}  "
            f"{d_corr:>8s}  "
            f"{v_corr:>8s}  "
            f"{row['n_computable_prompts']:>7d}  "
            f"{row['frac_computable']:>6.1%}"
        )
        print(line)

    print()

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "gsm8k_quality_filtered_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    per_prompt_path = OUTPUT_DIR / "gsm8k_quality_filtered_per_prompt.csv"
    if all_per_prompt:
        per_prompt_all = pd.concat(all_per_prompt, ignore_index=True)
        per_prompt_all.to_csv(per_prompt_path, index=False)
    else:
        print("WARNING: No per-prompt data to save.")
        return
    print(f"Per-prompt details saved to: {per_prompt_path}")

    # Also print a LaTeX-ready table
    print()
    print("LaTeX table:")
    print("\\begin{tabular}{l r r r r r r}")
    print("\\toprule")
    print("Model & Acc@1 & $\\bar{K}$ & $D_{\\text{all}}$ & $D_{\\text{corr}}$ & $V_{\\text{corr}}$ & $N$ \\\\")
    print("\\midrule")
    for _, row in summary_df.iterrows():
        mname = row["model"].replace("_", "\\_").replace("-", "-")
        d_corr = f"{row['d_correct_sbert']:.3f}" if not np.isnan(row["d_correct_sbert"]) else "--"
        v_corr = f"{row['d_correct_vendi']:.1f}" if not np.isnan(row["d_correct_vendi"]) else "--"
        print(
            f"{mname} & {row['accuracy_at_1']:.3f} & {row['mean_k_correct']:.1f} "
            f"& {row['d_all_sbert']:.3f} & {d_corr} & {v_corr} "
            f"& {row['n_computable_prompts']:.0f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
