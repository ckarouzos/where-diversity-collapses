#!/usr/bin/env python3
"""
Open-ended diversity tasks for lighteval: WritingPrompts, TruthfulQA,
MATH (Algebra/Geometry), GSM8K, CRUXEval, WildBench, PRISM, Alpaca.

Usage:
    lighteval vllm \\
        "model_name=allenai/OLMo-3-1025-7B" \\
        "writingprompts_diversity|0,truthfulqa_diversity|0,gsm8k_diversity|0" \\
        --custom-tasks src/evaluation/lighteval_open_ended_tasks.py \\
        --max-samples 500
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


# Creative Writing — WritingPrompts (euclaise/writingprompts)
# Schema: prompt (str), story (str). Splits: train/validation/test.


def writingprompts_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompt for creative story generation from WritingPrompts."""
    if not _in_slice(line):
        return None
    prompt_text = line.get("prompt", "")
    story = line.get("story", "")

    # WritingPrompts uses [WP], [EU], [CW] etc. tags — preserve them.
    # Add a clear separator for the model to begin its story.
    query = f"{prompt_text.strip()}\n\n"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[story],
        gold_index=0,
        instruction="",
    )


# TruthfulQA (truthfulqa/truthful_qa, config: generation)
# Schema: question, best_answer, correct_answers, incorrect_answers.
# Splits: validation only (817 items).


def truthfulqa_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompt for open-ended truthful question answering."""
    if not _in_slice(line):
        return None
    question = line.get("question", "")
    best_answer = line.get("best_answer", "")

    query = f"Q: {question}\nA:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {best_answer}"],
        gold_index=0,
        instruction="",
    )


# MATH (EleutherAI/hendrycks_math, configs: algebra, geometry, etc.)
# Schema: problem, solution, level, type. Splits: train, test per config.


def math_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompt for competition math problem solving."""
    if not _in_slice(line):
        return None
    problem = line.get("problem", "")
    solution = line.get("solution", "")

    query = f"Problem: {problem}\n\nSolution:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {solution}"],
        gold_index=0,
        instruction="",
    )


# GSM8K (openai/gsm8k, config: main)
# Schema: question, answer (step-by-step with #### final_answer).


def gsm8k_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompt for grade-school math problem solving."""
    if not _in_slice(line):
        return None
    question = line.get("question", "")
    answer = line.get("answer", "")

    query = f"Question: {question}\n\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {answer}"],
        gold_index=0,
        instruction="",
    )


# CRUXEval (cruxeval-org/cruxeval)
# Schema: code, input, output, id. Splits: test (800).



def cruxeval_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompt for code reasoning (predict output given function + input)."""
    if not _in_slice(line):
        return None
    code = line.get("code", "")
    inp = line.get("input", "")
    output = line.get("output", "")

    query = f"Given the following Python function:\n\n```python\n{code}\n```\n\nWhat is the output of `f({inp})`?\n\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {output}"],
        gold_index=0,
        instruction="",
    )


# WildBench (allenai/WildBench, config: v2)
# Schema: conversation_input, checklist, primary_tag, intent.
# Splits: test only (1,020 items).


def wildbench_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompt from WildBench conversation input."""
    if not _in_slice(line):
        return None
    conversation = line.get("conversation_input", [])

    # Extract the user's first message from the conversation structure
    if conversation and isinstance(conversation, list):
        first_turn = conversation[0]
        if isinstance(first_turn, dict):
            query = first_turn.get("content", "")
        else:
            query = str(first_turn)
    else:
        query = str(conversation)

    # Use checklist as rough reference for expected content
    checklist = line.get("checklist", [])
    reference = "; ".join(checklist) if checklist else ""

    return Doc(
        task_name=task_name,
        query=query.strip(),
        choices=[reference] if reference else [""],
        gold_index=0,
        instruction="",
    )


# PRISM (HannahRoseKirk/prism-alignment, config: conversations)
# Schema: opening_prompt, conversation_type, user_id. Split: train (8,011).
# Balanced subset: 2,232 per type (controversy/values/unguided).


def _is_prism_valid(line: dict) -> bool:
    """Filter PRISM prompts: balanced subset, no greetings, English only."""
    if not line.get("included_in_balanced_subset", False):
        return False
    prompt = line.get("opening_prompt", "").strip()
    if not prompt:
        return False
    # Remove trivial greetings
    greetings = {"hello", "hi", "hey", "hola", "hello!", "hi!", "hey!", "hello.", "hi."}
    if prompt.lower().rstrip(".!?,") in greetings:
        return False
    # Remove non-English (>20% non-ASCII characters)
    if len(prompt) > 0:
        non_ascii = sum(1 for c in prompt if ord(c) > 127)
        if non_ascii / len(prompt) > 0.2:
            return False
    return True


def prism_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompt from PRISM conversation opening prompt."""
    if not _in_slice(line):
        return None
    prompt = line.get("opening_prompt", "")

    return Doc(
        task_name=task_name,
        query=prompt.strip(),
        choices=[""],
        gold_index=0,
        instruction="",
    )


# ============================================================================
# Task Configurations
# ============================================================================

TASKS_TABLE = []

if _DIVERSITY_AVAILABLE:
    # --- Creative Writing: WritingPrompts ---
    # Stories are long (avg ~734 tokens), so generation_size=32768 is appropriate.
    # No stop sequence — creative writing has no natural terminator.
    writingprompts_diversity_task = LightevalTaskConfig(
        name="writingprompts_diversity",
        prompt_function=writingprompts_prompt_fn,
        hf_repo="euclaise/writingprompts",
        hf_subset="default",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=[],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(writingprompts_diversity_task)

    # --- Open-Ended QA: TruthfulQA (generation) ---
    # Answers are short-to-medium. Only validation split exists (817 items).
    # Stop at next question marker or paragraph break.
    truthfulqa_diversity_task = LightevalTaskConfig(
        name="truthfulqa_diversity",
        prompt_function=truthfulqa_prompt_fn,
        hf_repo="truthfulqa/truthful_qa",
        hf_subset="generation",
        hf_avail_splits=["validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        generation_grammar=None,
        stop_sequence=["\nQ:", "\n\n"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(truthfulqa_diversity_task)

    # --- Competition Math: MATH (algebra) ---
    # Solutions require step-by-step reasoning. 32768 tokens for CoT chains.
    # Using algebra config as representative subject; other configs can be
    # added as separate tasks for difficulty/subject stratification.
    math_algebra_diversity_task = LightevalTaskConfig(
        name="math_algebra_diversity",
        prompt_function=math_prompt_fn,
        hf_repo="EleutherAI/hendrycks_math",
        hf_subset="algebra",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=["\nProblem:", "\n\nProblem"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(math_algebra_diversity_task)

    # Additional MATH subjects for difficulty/subject stratification
    _MATH_EXTRA_SUBJECTS = [
        "number_theory",
        "geometry",
        "counting_and_probability",
    ]
    for subject in _MATH_EXTRA_SUBJECTS:
        task = LightevalTaskConfig(
            name=f"math_{subject}_diversity",
            prompt_function=math_prompt_fn,
            hf_repo="EleutherAI/hendrycks_math",
            hf_subset=subject,
            hf_avail_splits=["train", "test"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling",
            generation_size=32768,
            generation_grammar=None,
            stop_sequence=["\nProblem:", "\n\nProblem"],
            num_samples=16,
            metrics=[Metrics.DIVERSITY_ALL],
        )
        TASKS_TABLE.append(task)

    # --- Grade-School Math: GSM8K ---
    gsm8k_diversity_task = LightevalTaskConfig(
        name="gsm8k_diversity",
        prompt_function=gsm8k_prompt_fn,
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=["\nQuestion:", "\n\nQuestion"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(gsm8k_diversity_task)

    # --- Code Reasoning: CRUXEval ---
    cruxeval_diversity_task = LightevalTaskConfig(
        name="cruxeval_diversity",
        prompt_function=cruxeval_prompt_fn,
        hf_repo="cruxeval-org/cruxeval",
        hf_subset=None,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=2048,
        generation_grammar=None,
        stop_sequence=["\nGiven", "\n\nGiven"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(cruxeval_diversity_task)

    # --- Value Pluralism: PRISM (NeurIPS 2024 D&B Award) ---
    # Opening prompts from 1,500 participants across 75 countries, covering
    # controversy-guided, values-guided, and unguided conversations.
    # Tests whether post-training homogenizes responses on topics where
    # humans genuinely disagree (Israel-Palestine, euthanasia, wealth
    # inequality, gender, religion, etc.).
    # Full dataset: 8,011 conversations. With --max-samples 500, a random
    # subsample is taken across all conversation types. Post-hoc filtering
    # by conversation_type is possible from the saved generations JSONL.
    prism_diversity_task = LightevalTaskConfig(
        name="prism_diversity",
        prompt_function=prism_prompt_fn,
        hf_repo="HannahRoseKirk/prism-alignment",
        hf_subset="conversations",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=[],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
        hf_filter=_is_prism_valid,
    )
    TASKS_TABLE.append(prism_diversity_task)

    # --- Real-World Queries: WildBench v2 ---
    # Open-ended real user queries. No natural stop sequence.
    # 1,020 test items spanning multiple task types.
    wildbench_diversity_task = LightevalTaskConfig(
        name="wildbench_diversity",
        prompt_function=wildbench_prompt_fn,
        hf_repo="allenai/WildBench",
        hf_subset="v2",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=[],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )
    TASKS_TABLE.append(wildbench_diversity_task)


# ============================================================================
# Validation & Testing
# ============================================================================

