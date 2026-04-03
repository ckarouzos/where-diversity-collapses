#!/usr/bin/env python3
"""
Generate text using OLMo-3 models via vLLM for high-throughput inference.

vLLM provides faster inference with PagedAttention and continuous batching.
Suitable for generating large numbers of outputs efficiently.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using OLMo-3 models via vLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMo-3-1025-7B",
        help="Model name or path (default: allenai/OLMo-3-1025-7B)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Text prompts for generation (can specify multiple)",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Path to file with prompts (one per line or JSONL with 'prompt' field)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (JSONL format). If not specified, prints to stdout",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of generations per prompt (default: 1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling (default: -1, disabled)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "bfloat16", "float"],
        help="Model dtype (default: auto)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    return parser.parse_args()


def load_prompts(args) -> List[str]:
    """Load prompts from command line or file."""
    prompts = []

    if args.prompts:
        prompts.extend(args.prompts)

    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    if isinstance(data, dict) and "prompt" in data:
                        prompts.append(data["prompt"])
                    else:
                        raise ValueError(
                            f"Invalid JSONL prompt entry in {args.prompts_file}: {line}"
                        )
                else:
                    prompts.append(line)

    if not prompts:
        print("Error: No prompts provided. Use --prompts or --prompts-file", file=sys.stderr)
        sys.exit(1)

    return prompts


def main():
    args = parse_args()

    print(f"Loading model: {args.model}", file=sys.stderr)
    print(f"Tensor parallel size: {args.tensor_parallel_size}", file=sys.stderr)

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Load prompts
    prompts = load_prompts(args)
    print(f"Loaded {len(prompts)} prompt(s)", file=sys.stderr)

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.num_generations,
    )

    print("Generating...", file=sys.stderr)

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Format results
    results = []
    for i, output in enumerate(outputs):
        generations = [out.text for out in output.outputs]
        result = {
            "id": f"prompt_{i}",
            "prompt": output.prompt,
            "outputs": generations,
        }
        results.append(result)

    # Write output
    if args.output:
        print(f"Writing results to {args.output}", file=sys.stderr)
        with open(args.output, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    else:
        for result in results:
            print(json.dumps(result))

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
