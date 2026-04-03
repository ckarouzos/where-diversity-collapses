#!/usr/bin/env python3
"""Compute diversity metrics (including Vendi) from existing generation JSONL files.

Usage:
    # Single model directory
    python scripts/compute_vendi_from_generations.py \
        --generations-dir runs/prism/Olmo-3-7B-Instruct

    # All models under a parent directory
    python scripts/compute_vendi_from_generations.py \
        --generations-dir runs/prism --all-models

    # Only Vendi (skip other metrics for speed)
    python scripts/compute_vendi_from_generations.py \
        --generations-dir runs/prism --all-models --metrics VENDI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.lighteval_diversity_metrics import DiversityConfig, compute_diversity


def find_model_dirs(parent: Path) -> list[Path]:
    """Find all subdirectories containing diversity_generations.jsonl."""
    dirs = []
    for jsonl in sorted(parent.rglob("diversity_generations.jsonl")):
        dirs.append(jsonl.parent)
    return dirs


def load_records_by_task(jsonl_path: Path) -> dict[str, list[dict]]:
    """Load JSONL and group records by task name."""
    by_task: dict[str, list[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task = record.get("task", "unknown")
            # Normalise task key: strip lighteval suffix like "|0" or "|5"
            task_clean = task.split("|")[0] if "|" in task else task
            by_task.setdefault(task_clean, []).append(record)
    return by_task


def compute_for_model(
    model_dir: Path,
    metrics: list[str],
    output_dir: Path,
) -> list[dict] | None:
    """Compute diversity metrics per task for one model's generations."""
    import tempfile

    jsonl_path = model_dir / "diversity_generations.jsonl"
    if not jsonl_path.exists():
        logger.warning("No diversity_generations.jsonl in %s, skipping", model_dir)
        return None

    model_name = model_dir.name
    by_task = load_records_by_task(jsonl_path)

    if not by_task:
        logger.warning("Empty generations file for %s, skipping", model_name)
        return None

    logger.info(
        "Processing %s: %d tasks (%s)",
        model_name,
        len(by_task),
        ", ".join(f"{t}={len(r)}" for t, r in by_task.items()),
    )

    task_results = []
    for task_name, records in sorted(by_task.items()):
        n_prompts = len(records)
        n_outputs = sum(len(r.get("outputs", [])) for r in records)
        k = n_outputs // n_prompts if n_prompts > 0 else 0

        if n_prompts == 0 or k == 0:
            logger.warning("Skipping %s/%s: no valid records", model_name, task_name)
            continue

        # Write per-task JSONL to a temp directory for compute_diversity
        with tempfile.TemporaryDirectory() as tmp:
            tmp_jsonl = Path(tmp) / "diversity_generations.jsonl"
            with open(tmp_jsonl, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            config = DiversityConfig(
                model_name=model_name,
                max_inputs=n_prompts,
                generations_per_input=k,
                require_exact_counts=False,
            )

            result = compute_diversity(
                Path(tmp),
                metrics=metrics,
                per_input=True,
                across_input=True,
                config=config,
            )

        result["task"] = task_name
        result["model_name"] = model_name
        task_results.append(result)

        logger.info(
            "  %s/%s: N=%d K=%d %s",
            model_name, task_name, n_prompts, k,
            " ".join(
                f"{m}={result['metrics'].get(m, {}).get('per_input_mean', 0):.3f}"
                for m in metrics
            ),
        )

    # Save per-model results JSON (all tasks)
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_output_dir / "diversity_results.json"
    with open(result_path, "w") as f:
        json.dump(task_results, f, indent=2, default=str)
    logger.info("Saved %d task results to %s", len(task_results), result_path)

    return task_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute diversity metrics (including Vendi) from saved generation JSONL files."
    )
    parser.add_argument(
        "--generations-dir",
        type=Path,
        required=True,
        help="Path to a model output directory (with diversity_generations.jsonl) "
             "or a parent directory containing multiple model subdirs (with --all-models).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Scan all subdirectories for diversity_generations.jsonl files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["EAD", "SBERT", "NLI", "VENDI"],
        help="Metrics to compute (default: EAD SBERT NLI VENDI).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: <generations-dir>/vendi_results).",
    )
    args = parser.parse_args()

    gen_dir = args.generations_dir.resolve()
    if not gen_dir.exists():
        logger.error("Directory does not exist: %s", gen_dir)
        sys.exit(1)

    output_dir = args.output_dir or gen_dir / "vendi_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.upper() for m in args.metrics]
    logger.info("Metrics to compute: %s", metrics)

    if args.all_models:
        model_dirs = find_model_dirs(gen_dir)
        if not model_dirs:
            logger.error("No diversity_generations.jsonl files found under %s", gen_dir)
            sys.exit(1)
        logger.info("Found %d model directories", len(model_dirs))
    else:
        if not (gen_dir / "diversity_generations.jsonl").exists():
            logger.error("No diversity_generations.jsonl in %s", gen_dir)
            sys.exit(1)
        model_dirs = [gen_dir]

    all_task_results = []
    models_ok = 0
    for model_dir in model_dirs:
        task_results = compute_for_model(model_dir, metrics, output_dir)
        if task_results is not None:
            all_task_results.extend(task_results)
            models_ok += 1

    # Write combined summary CSV (one row per model×task)
    if all_task_results:
        summary_path = output_dir / "vendi_summary.csv"
        _write_summary_csv(all_task_results, summary_path, metrics)
        logger.info("Summary CSV: %s", summary_path)

    logger.info(
        "Done. %d models, %d task results.", models_ok, len(all_task_results)
    )


def _write_summary_csv(
    results: list[dict], output_path: Path, metrics: list[str]
) -> None:
    """Write a summary CSV with one row per model x task."""
    import csv

    fieldnames = ["model", "task", "N", "K"]
    for m in metrics:
        fieldnames.extend([
            f"{m}_per_input_mean",
            f"{m}_per_input_std",
            f"{m}_across_input",
        ])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "model": result.get("model_name", ""),
                "task": result.get("task", ""),
                "N": result.get("N", ""),
                "K": result.get("K", ""),
            }
            for m in metrics:
                metric_data = result.get("metrics", {}).get(m, {})
                row[f"{m}_per_input_mean"] = metric_data.get(
                    "per_input_mean", metric_data.get(f"mean_per_input_{m}", "")
                )
                row[f"{m}_per_input_std"] = metric_data.get(
                    "per_input_std", metric_data.get(f"std_per_input_{m}", "")
                )
                row[f"{m}_across_input"] = metric_data.get("across_input", "")
            writer.writerow(row)


if __name__ == "__main__":
    main()
