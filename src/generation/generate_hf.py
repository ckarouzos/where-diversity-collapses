#!/usr/bin/env python3
"""
Generate text using OLMo-3 models via Hugging Face transformers.

Supports both base and instruct variants of OLMo-3 models.
Use --chat to enable chat template formatting for instruct models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using OLMo-3 models via Hugging Face transformers"
    )
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
        "--max-new-tokens",
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
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use chat template formatting (for instruct models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
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

    # Set dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}", file=sys.stderr)
    print(f"Device: {args.device}", file=sys.stderr)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=True,
    )

    if args.device == "cpu":
        model = model.to(args.device)

    model.eval()

    # Load prompts
    prompts = load_prompts(args)
    print(f"Loaded {len(prompts)} prompt(s)", file=sys.stderr)

    # Generate
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Generating for prompt {i + 1}/{len(prompts)}...", file=sys.stderr)

        # Format prompt
        if args.chat:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.temperature > 0,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode
        generations = []
        for output in outputs:
            generated_text = tokenizer.decode(
                output[inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            generations.append(generated_text)

        result = {
            "id": f"prompt_{i}",
            "prompt": prompt,
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
