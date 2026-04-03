#!/usr/bin/env python3
"""
Lighteval custom tasks for summarization evaluation.

Task definitions for evaluating language models on summarization:
- TL;DR (Reddit)
- CNN/DailyMail
- XSum

Usage:
    lighteval accelerate \
        "model_name=allenai/OLMo-3-1025-7B,trust_remote_code=True" \
        "tldr_diversity,cnn_dailymail_diversity,xsum_diversity" \
        --custom-tasks src/evaluation/lighteval_summarization_tasks.py
"""

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

# ============================================================================
# Prompt Functions
# ============================================================================


def tldr_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompts for TL;DR (Reddit) summarization task."""
    subreddit = line.get("subreddit", "")
    title = line.get("title", "")
    post = line.get("post", "")
    prompt = f"SUBREDDIT: {subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"
    summary = line.get("summary", "")
    target = f" {summary}"
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[target],
        gold_index=0,
        instruction="",
    )


def cnn_dailymail_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompts for CNN/DailyMail summarization task."""
    article = line.get("article", "")
    prompt = f"ARTICLE: {article}\nTL;DR:"
    highlights = line.get("highlights", "")
    target = f" {highlights}"
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[target],
        gold_index=0,
        instruction="",
    )


def xsum_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Format prompts for XSum summarization task."""
    document = line.get("document", "")
    prompt = f"ARTICLE: {document}\nTL;DR:"
    summary = line.get("summary", "")
    target = f" {summary}"
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[target],
        gold_index=0,
        instruction="",
    )


# ============================================================================
# Task Configurations
# ============================================================================

tldr_task = LightevalTaskConfig(
    name="tldr",
    prompt_function=tldr_prompt_fn,
    hf_repo="UCL-DARK/openai-tldr-filtered",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=256,
    generation_grammar=None,
    stop_sequence=["\n\n"],
    metrics=[Metrics.rouge1, Metrics.rouge2, Metrics.rougeL],
)

cnn_dailymail_task = LightevalTaskConfig(
    name="cnn_dailymail",
    prompt_function=cnn_dailymail_prompt_fn,
    hf_repo="cnn_dailymail",
    hf_subset="3.0.0",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=256,
    generation_grammar=None,
    stop_sequence=["\n\n"],
    metrics=[Metrics.rouge1, Metrics.rouge2, Metrics.rougeL],
)

xsum_task = LightevalTaskConfig(
    name="xsum",
    prompt_function=xsum_prompt_fn,
    hf_repo="xsum",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=128,
    generation_grammar=None,
    stop_sequence=["\n"],
    metrics=[Metrics.rouge1, Metrics.rouge2, Metrics.rougeL],
)

# ============================================================================
# Task Table Export
# ============================================================================

TASKS_TABLE = [
    tldr_task,
    cnn_dailymail_task,
    xsum_task,
]

if _DIVERSITY_AVAILABLE:
    tldr_diversity_task = LightevalTaskConfig(
        name="tldr_diversity",
        prompt_function=tldr_prompt_fn,
        hf_repo="UCL-DARK/openai-tldr-filtered",
        hf_subset="default",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=["\n\n"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )

    cnn_dailymail_diversity_task = LightevalTaskConfig(
        name="cnn_dailymail_diversity",
        prompt_function=cnn_dailymail_prompt_fn,
        hf_repo="cnn_dailymail",
        hf_subset="3.0.0",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=["\n\n"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )

    xsum_diversity_task = LightevalTaskConfig(
        name="xsum_diversity",
        prompt_function=xsum_prompt_fn,
        hf_repo="xsum",
        hf_subset="default",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=32768,
        generation_grammar=None,
        stop_sequence=["\n"],
        num_samples=16,
        metrics=[Metrics.DIVERSITY_ALL],
    )

    TASKS_TABLE.extend([
        tldr_diversity_task,
        cnn_dailymail_diversity_task,
        xsum_diversity_task,
    ])
