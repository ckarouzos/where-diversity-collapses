#!/usr/bin/env python3
"""
Lighteval integration for evaluating OLMo-3 models.

This script provides a convenient wrapper around Lighteval for evaluating
OLMo-3 models on standard benchmarks and custom tasks.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from olmo3_models import filter_models, format_models_table, list_models


def run_lighteval(
    model: str,
    tasks: List[str],
    output_dir: Path,
    backend: str = "transformers",
    trust_remote_code: bool = True,
    tensor_parallel_size: Optional[int] = None,
    repo_id: Optional[str] = None,
    additional_args: Optional[List[str]] = None,
) -> int:
    """
    Run Lighteval evaluation on specified model and tasks.

    Parameters
    ----------
    model : str
        Model name or path (e.g., 'allenai/OLMo-3-1025-7B')
    tasks : List[str]
        List of tasks to evaluate
    output_dir : Path
        Directory to save evaluation results
    backend : str
        Backend to use (transformers, vllm, etc.)
    trust_remote_code : bool
        Whether to trust remote code (required for OLMo-3)
    tensor_parallel_size : Optional[int]
        Tensor parallel size for distributed evaluation (vllm backend)
    repo_id : Optional[str]
        Hugging Face Hub repo ID to push results
    additional_args : Optional[List[str]]
        Additional arguments to pass to lighteval

    Returns
    -------
    int
        Exit code from lighteval command
    """
    # Construct model specifier with backend
    model_spec = f"{backend}/{model}"

    # Construct tasks string
    tasks_str = ",".join(tasks)

    # Build command
    cmd = [
        "lighteval",
        "eval",
        model_spec,
        tasks_str,
        "--output-dir",
        str(output_dir),
    ]

    # Add optional arguments
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if tensor_parallel_size is not None and backend == "vllm":
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

    if repo_id:
        cmd.extend(["--repo-id", repo_id])

    if additional_args:
        cmd.extend(additional_args)

    # Print command for user
    print(f"Running: {' '.join(cmd)}")

    # Execute command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running lighteval: {e}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print(
            "Error: lighteval command not found. Please install with: pip install lighteval",
            file=sys.stderr,
        )
        return 1


def _split_values(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    parts: List[str] = []
    for value in values:
        parts.extend([chunk.strip() for chunk in value.split(",") if chunk.strip()])
    return parts


def _model_output_dir(base_dir: Path, model: str, multi: bool) -> Path:
    if not multi:
        return base_dir
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return base_dir / safe_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Lighteval evaluations on OLMo-3 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run diversity evaluation on a single model
  python src/evaluation/lighteval_runner.py \\
      --model allenai/OLMo-3-1025-7B \\
      --tasks tldr_diversity cnn_dailymail_diversity xsum_diversity \\
      --output-dir ./runs/diversity \\
      --lighteval-args --custom-tasks src/evaluation/lighteval_summarization_tasks.py

  # Evaluate all 7B Think models
  python src/evaluation/lighteval_runner.py \\
      --tasks tldr_diversity \\
      --family think \\
      --size 7B \\
      --output-dir ./runs/diversity

  # List available OLMo-3 models
  python src/evaluation/lighteval_runner.py --list-models
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path (e.g., allenai/OLMo-3-1025-7B)",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Explicit model list (comma-separated or space-separated)",
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate all known OLMo-3 checkpoints from the registry",
    )

    parser.add_argument(
        "--family",
        type=str,
        nargs="+",
        choices=["base", "think", "instruct", "rl-zero"],
        help="Filter by model family",
    )

    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        choices=["base", "sft", "dpo", "final", "rl-zero"],
        help="Filter by training stage",
    )

    parser.add_argument(
        "--size",
        type=str,
        nargs="+",
        choices=["7B"],
        help="Filter by model size",
    )

    parser.add_argument(
        "--series",
        type=str,
        nargs="+",
        choices=["3", "3.1"],
        help="Filter by series (3 or 3.1)",
    )

    parser.add_argument(
        "--domain",
        type=str,
        nargs="+",
        choices=["math", "code", "if", "general"],
        help="Filter RL-Zero models by domain focus",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all known OLMo-3 checkpoints and exit",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Tasks to evaluate on (e.g., tldr_diversity gsm8k_diversity)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./eval-results"),
        help="Directory to save evaluation results (default: ./eval-results)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm", "nanotron", "tgi", "sglang", "inference-endpoint"],
        help="Backend to use for evaluation (default: transformers)",
    )

    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trusting remote code (enabled by default for OLMo-3 compatibility)",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallel size for distributed evaluation (vllm backend only)",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="Hugging Face Hub repo ID to push results (optional)",
    )

    parser.add_argument(
        "--lighteval-args",
        type=str,
        nargs="*",
        help="Additional arguments to pass to lighteval",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print(format_models_table(list_models()))
        sys.exit(0)

    explicit_models = _split_values(args.models)
    if args.model:
        explicit_models.append(args.model)

    filter_used = any(
        [args.all_models, args.family, args.stage, args.size, args.series, args.domain]
    )

    if explicit_models:
        selected_models = explicit_models
    elif filter_used:
        registry = list_models()
        filtered = filter_models(
            registry,
            sizes=args.size,
            families=args.family,
            stages=args.stage,
            series=args.series,
            domains=args.domain,
        )
        selected_models = [model.name for model in filtered]
    else:
        raise SystemExit(
            "Error: provide --model/--models, or select from the registry with "
            "--all-models or any of --family/--stage/--size/--series/--domain."
        )

    if not selected_models:
        raise SystemExit("Error: no models matched the requested filters.")

    # Handle trust_remote_code logic - default to True for OLMo-3
    trust_remote_code = not args.no_trust_remote_code

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    multi = len(selected_models) > 1
    exit_codes: List[int] = []
    for model in selected_models:
        model_output_dir = _model_output_dir(args.output_dir, model, multi)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        exit_code = run_lighteval(
            model=model,
            tasks=args.tasks,
            output_dir=model_output_dir,
            backend=args.backend,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            repo_id=args.repo_id,
            additional_args=args.lighteval_args,
        )
        exit_codes.append(exit_code)

    if all(code == 0 for code in exit_codes):
        print(f"\n✓ Evaluation complete! Results saved to {args.output_dir}")
        sys.exit(0)

    print(f"\n✗ Evaluation failed with exit codes: {exit_codes}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
