#!/usr/bin/env python3
"""
Answer extraction and matching for GSM8K math evaluation.

Extracts numeric answers from chain-of-thought reasoning, normalizes them,
and checks for equivalence. Handles common model output formats including
#### N, "the answer is N", \\boxed{N}, and fallback to last number.

Functions:
    extract_gsm8k_gold_answer: Parse #### <number> from GSM8K reference
    extract_generated_answer: Extract answer from model CoT output
    strip_think_tags: Remove <think>...</think> blocks
    normalize_number: Normalize numeric string
    answers_match: Check numeric equivalence
"""

from __future__ import annotations

import math
import re
from typing import Optional


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Handles two cases:
    1. Full tags: ``<think>reasoning</think>answer`` → ``answer``
    2. Closing tag only (opening was in prompt): ``reasoning</think>answer`` → ``answer``
    """
    # Case 1: full tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Case 2: closing tag only (opening tag was in the prompt)
    if "</think>" in text and "<think>" not in text:
        idx = text.find("</think>")
        text = text[idx + len("</think>"):]
    return text.strip()


def normalize_number(s: str) -> str:
    """Normalize a numeric string for comparison.

    Strips commas, dollar signs, percent signs, trailing zeros after decimal,
    and leading/trailing whitespace.

    Examples:
        "1,234" -> "1234"
        "42.00" -> "42"
        "$100" -> "100"
        "-3.50" -> "-3.5"
    """
    s = s.strip()
    # Remove currency/percent symbols
    s = s.replace("$", "").replace("%", "").replace(",", "")
    # Try to normalize as a float
    try:
        val = float(s)
        if not math.isfinite(val):
            return s
        # Convert to int if no fractional part
        if val == int(val):
            return str(int(val))
        # Remove trailing zeros
        return f"{val:g}"
    except (ValueError, OverflowError):
        return s


def extract_gsm8k_gold_answer(text: str) -> Optional[str]:
    """Extract the gold answer from a GSM8K answer field.

    GSM8K gold answers end with "#### <number>".

    Parameters
    ----------
    text : str
        The answer field from the GSM8K dataset.

    Returns
    -------
    str or None
        Normalized answer string, or None if no pattern found.
    """
    match = re.search(r"####\s*(.+?)$", text.strip(), re.MULTILINE)
    if match:
        return normalize_number(match.group(1).strip())
    return None


def extract_generated_answer(text: str) -> Optional[str]:
    """Extract the numeric answer from a model's chain-of-thought output.

    Tries multiple patterns in order of specificity:
    1. #### N (GSM8K-style explicit answer marker)
    2. \\boxed{N} (LaTeX-style answer box)
    3. "the answer is N" / "answer: N" (natural language)
    4. Last number in the text (fallback)

    Parameters
    ----------
    text : str
        Model-generated text (may include thinking tags).

    Returns
    -------
    str or None
        Normalized answer string, or None if no number found.
    """
    # Remove think tags first
    text = strip_think_tags(text)

    # Pattern 1: #### N
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return normalize_number(match.group(1).strip())

    # Pattern 2: \boxed{N}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return normalize_number(match.group(1).strip())

    # Pattern 3: "the answer is N" / "answer: N" / "= N" at end
    match = re.search(
        r"(?:the\s+)?answer\s*(?:is|:)\s*([\-\$]?[\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if match:
        return normalize_number(match.group(1).strip())

    # Pattern 4: Last number in the text
    numbers = re.findall(r"(?<!\w)([\-]?\d[\d,]*\.?\d*)(?!\w)", text)
    if numbers:
        return normalize_number(numbers[-1])

    return None


def answers_match(predicted: Optional[str], gold: Optional[str]) -> bool:
    """Check if predicted and gold answers are numerically equivalent.

    Parameters
    ----------
    predicted : str or None
        Extracted predicted answer.
    gold : str or None
        Extracted gold answer.

    Returns
    -------
    bool
        True if both are non-None and numerically equal.
    """
    if predicted is None or gold is None:
        return False

    pred_norm = normalize_number(predicted)
    gold_norm = normalize_number(gold)

    # Direct string match
    if pred_norm == gold_norm:
        return True

    # Numeric comparison
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-6
    except ValueError:
        return False
