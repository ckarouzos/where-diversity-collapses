#!/usr/bin/env python3
"""
Extended Lighteval tasks for diversity evaluation beyond summarization.

Adds instruction-following and code generation diversity tasks with
appropriate length normalization and task-specific adaptations.

Tasks:
    - alpaca_diversity: Instruction-following (tatsu-lab/alpaca_eval)
    - ifeval_diversity: Instruction-following evaluation (google/IFEval)
    - humaneval_diversity: Code generation (openai_humaneval)
    - mbpp_diversity: Code generation (mbpp)

Usage:
    lighteval accelerate \\
        "model_name=allenai/OLMo-3-1025-7B,trust_remote_code=True,generation_parameters={temperature:1.0,top_p:1.0}" \\
        alpaca_diversity,humaneval_diversity \\
        --custom-tasks src/evaluation/lighteval_extended_tasks.py
"""

import os
import sys
from pathlib import Path

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evaluation import (
    lighteval_diversity_metrics as _lighteval_diversity_metrics,  # noqa: F401, E402
)

_DIVERSITY_AVAILABLE = hasattr(Metrics, "DIVERSITY_ALL")


def _in_slice(line: dict) -> bool:
    """Check if sample is within the env-var-defined slice.

    Set LIGHTEVAL_SAMPLE_OFFSET and LIGHTEVAL_SAMPLE_LIMIT to enable
    data-parallel slicing across multiple jobs. When LIGHTEVAL_SAMPLE_LIMIT
    is 0 or unset, all samples pass through (no slicing).
    """
    limit = int(os.environ.get("LIGHTEVAL_SAMPLE_LIMIT", "0"))
    if limit <= 0:
        return True
    offset = int(os.environ.get("LIGHTEVAL_SAMPLE_OFFSET", "0"))
    idx = line.get("__index", 0)
    return offset <= idx < offset + limit


# ============================================================================
# Instruction-Following Tasks
# ============================================================================


def alpaca_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """
    Format prompts for Alpaca-style instruction-following.

    Uses a structured instruction format with optional input context.
    """
    if not _in_slice(line):
        return None
    instruction = line.get("instruction", "")
    input_text = line.get("input", "")
    output = line.get("output", "")

    if input_text:
        prompt = (
            f"Below is an instruction that describes a task, paired with an input "
            f"that provides further context. Write a response that appropriately "
            f"completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task. Write a response "
            f"that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[output],
        gold_index=0,
        instruction="",
    )


def ifeval_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """
    Format prompts for IFEval instruction-following evaluation.

    Direct instruction prompting without additional framing.
    """
    if not _in_slice(line):
        return None
    prompt = line.get("prompt", "")
    # IFEval doesn't have a single gold answer; use empty placeholder
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[""],
        gold_index=0,
        instruction="",
    )


# ============================================================================
# Code Generation Tasks
# ============================================================================


def humaneval_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """
    Format prompts for HumanEval code generation.

    Uses the function signature + docstring as the prompt.
    """
    if not _in_slice(line):
        return None
    prompt = line.get("prompt", "")
    canonical_solution = line.get("canonical_solution", "")

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[canonical_solution],
        gold_index=0,
        instruction="",
    )


_HUMANEVAL_INSTRUCT_PREFIX = (
    "Solve the following code problem step by step. "
    "The last part of your response should be the solution to the problem "
    "in form ```\npython\nCODE\n``` where CODE is the solution for the problem.\n\n"
)
_HUMANEVAL_INSTRUCT_SUFFIX = (
    "\nRemember to put your solution inside the ```\npython\nCODE\n``` tags"
)


def humaneval_instruct_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """
    Format prompts for HumanEval with an instruction wrapper.

    Wraps the function signature + docstring in an instruction that asks the
    model to solve step-by-step and output code in markdown blocks.  This
    matches the prompt format used by OLMo-3.1 RL-Zero-Code's chat template,
    applied uniformly to all models for fair comparison.
    """
    if not _in_slice(line):
        return None
    prompt = line.get("prompt", "")
    canonical_solution = line.get("canonical_solution", "")

    return Doc(
        task_name=task_name,
        query=_HUMANEVAL_INSTRUCT_PREFIX + prompt + _HUMANEVAL_INSTRUCT_SUFFIX,
        choices=[canonical_solution],
        gold_index=0,
        instruction="",
        original_query=prompt,
    )


def mbpp_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """
    Format prompts for MBPP code generation.

    Uses the task description as a docstring-style prompt.
    """
    if not _in_slice(line):
        return None
    text = line.get("text", "")
    code = line.get("code", "")

    prompt = f'"""\n{text}\n"""\n'

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[code],
        gold_index=0,
        instruction="",
    )


# ============================================================================
# Task Configurations
# ============================================================================

TASKS_TABLE = []

# Instruction-following diversity tasks
if _DIVERSITY_AVAILABLE:
    alpaca_diversity_task = LightevalTaskConfig(
        name="alpaca_diversity",
        prompt_function=alpaca_prompt_fn,
        hf_repo="tatsu-lab/alpaca",
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],  # Alpaca only has train split
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,  # OLMo-3 paper recommended for Think model CoT
        generation_grammar=None,
        stop_sequence=["\n\n###", "\n\n##"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(alpaca_diversity_task)

    ifeval_diversity_task = LightevalTaskConfig(
        name="ifeval_diversity",
        prompt_function=ifeval_prompt_fn,
        hf_repo="google/IFEval",
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,  # OLMo-3 paper recommended for Think model CoT
        generation_grammar=None,
        stop_sequence=[],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(ifeval_diversity_task)

    # Code generation diversity tasks
    # HumanEval: diversity + pass@1 approximation via canonical solution match
    humaneval_diversity_task = LightevalTaskConfig(
        name="humaneval_diversity",
        prompt_function=humaneval_prompt_fn,
        hf_repo="openai_humaneval",
        hf_subset="openai_humaneval",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,  # OLMo-3 paper recommended for Think model CoT
        generation_grammar=None,
        stop_sequence=["\nclass ", "\ndef ", "\n#", "\nif __name__"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(humaneval_diversity_task)

    # HumanEval with instruction wrapper — for fair comparison of base/RL-Zero
    # models using the same prompt format as OLMo-3.1-RL-Zero-Code's template.
    humaneval_instruct_diversity_task = LightevalTaskConfig(
        name="humaneval_instruct_diversity",
        prompt_function=humaneval_instruct_prompt_fn,
        hf_repo="openai_humaneval",
        hf_subset="openai_humaneval",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=4096,
        generation_grammar=None,
        stop_sequence=[],  # No stop seqs — instruction output uses markdown blocks
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(humaneval_instruct_diversity_task)

    mbpp_diversity_task = LightevalTaskConfig(
        name="mbpp_diversity",
        prompt_function=mbpp_prompt_fn,
        hf_repo="mbpp",
        hf_subset="full",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=32768,  # OLMo-3 paper recommended for Think model CoT
        generation_grammar=None,
        stop_sequence=["\nclass ", "\n#", "\nif __name__"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(mbpp_diversity_task)
