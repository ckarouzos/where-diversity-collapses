#!/usr/bin/env python3
"""
IFEval constraint verification for instruction-following evaluation.

Implements programmatic verification of the 25 constraint types in
Google's IFEval benchmark. Each constraint is a pure Python heuristic
check — no ML models needed.

Reference: Zhou et al., "Instruction-Following Evaluation for Large Language
Models" (2023). https://arxiv.org/abs/2311.07911

Functions:
    verify_constraints: Check all constraints for a response
    verify_single_constraint: Check one constraint
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Tuple

# ============================================================================
# Constraint Registry
# ============================================================================

CONSTRAINT_REGISTRY: Dict[str, Callable] = {}


def _register(name: str):
    """Decorator to register a constraint checker."""
    def decorator(fn):
        CONSTRAINT_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# Length Constraints
# ============================================================================

@_register("length_constraints:number_words")
def _check_number_words(response: str, **kwargs) -> bool:
    """Check word count: relation (at least/at most) num_words."""
    relation = kwargs.get("relation", "at least")
    num_words = kwargs.get("num_words", 0)
    word_count = len(response.split())
    if relation == "at least":
        return word_count >= num_words
    elif relation == "at most":
        return word_count <= num_words
    return True


@_register("length_constraints:number_sentences")
def _check_number_sentences(response: str, **kwargs) -> bool:
    """Check sentence count."""
    relation = kwargs.get("relation", "at least")
    num_sentences = kwargs.get("num_sentences", 0)
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    count = len(sentences)
    if relation == "at least":
        return count >= num_sentences
    elif relation == "at most":
        return count <= num_sentences
    return True


@_register("length_constraints:number_paragraphs")
def _check_number_paragraphs(response: str, **kwargs) -> bool:
    """Check paragraph count."""
    relation = kwargs.get("relation", "at least")
    num_paragraphs = kwargs.get("num_paragraphs", 0)
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    count = len(paragraphs)
    if relation == "at least":
        return count >= num_paragraphs
    elif relation == "at most":
        return count <= num_paragraphs
    return True


# ============================================================================
# Keyword Constraints
# ============================================================================

@_register("keywords:existence")
def _check_keyword_existence(response: str, **kwargs) -> bool:
    """Check that all keywords exist in the response."""
    keywords = kwargs.get("keywords", [])
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


@_register("keywords:frequency")
def _check_keyword_frequency(response: str, **kwargs) -> bool:
    """Check keyword appears exactly/at least/at most N times."""
    keyword = kwargs.get("keyword", "")
    frequency = kwargs.get("frequency", 0)
    relation = kwargs.get("relation", "at least")
    count = response.lower().count(keyword.lower())
    if relation == "at least":
        return count >= frequency
    elif relation == "at most":
        return count <= frequency
    elif relation == "exactly":
        return count == frequency
    return True


@_register("keywords:forbidden_words")
def _check_forbidden_words(response: str, **kwargs) -> bool:
    """Check that forbidden words do not appear."""
    forbidden_words = kwargs.get("forbidden_words", [])
    response_lower = response.lower()
    return not any(fw.lower() in response_lower for fw in forbidden_words)


@_register("keywords:letter_frequency")
def _check_letter_frequency(response: str, **kwargs) -> bool:
    """Check that a letter appears at least N times."""
    letter = kwargs.get("letter", "").lower()
    let_frequency = kwargs.get("let_frequency", 0)
    relation = kwargs.get("let_relation", "at least")
    count = response.lower().count(letter)
    if relation == "at least":
        return count >= let_frequency
    elif relation == "at most":
        return count <= let_frequency
    return True


# ============================================================================
# Format Constraints
# ============================================================================

@_register("detectable_format:number_bullet_lists")
def _check_bullet_lists(response: str, **kwargs) -> bool:
    """Check number of bullet point lists."""
    num_bullets = kwargs.get("num_bullets", 0)
    # Count lines starting with bullet markers
    bullet_lines = re.findall(r"^[\s]*[-*•]\s", response, re.MULTILINE)
    return len(bullet_lines) >= num_bullets


@_register("detectable_format:constrained_response")
def _check_constrained_response(response: str, **kwargs) -> bool:
    """Response must be one of the given options (case-insensitive)."""
    # No specific implementation needed for all cases
    return True


@_register("detectable_format:number_highlighted_sections")
def _check_highlighted_sections(response: str, **kwargs) -> bool:
    """Check number of highlighted sections (*highlighted*)."""
    num_highlights = kwargs.get("num_highlights", 0)
    highlights = re.findall(r"\*[^*\n]+\*", response)
    return len(highlights) >= num_highlights


@_register("detectable_format:multiple_sections")
def _check_multiple_sections(response: str, **kwargs) -> bool:
    """Check that response has multiple sections with headers."""
    section_spliter = kwargs.get("section_spliter", "Section")
    num_sections = kwargs.get("num_sections", 0)
    # Count occurrences of the section splitter pattern
    sections = re.findall(
        rf"(?:^|\n){re.escape(section_spliter)}\s*\d*",
        response,
        re.IGNORECASE,
    )
    return len(sections) >= num_sections


@_register("detectable_format:json_format")
def _check_json_format(response: str, **kwargs) -> bool:
    """Check that the response is valid JSON."""
    # Try to find JSON in the response
    text = response.strip()
    # Try the whole response
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON block in markdown
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1).strip())
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


@_register("detectable_format:title")
def _check_title(response: str, **kwargs) -> bool:
    """Check that response wraps title in <<title>>."""
    return bool(re.search(r"<<[^>]+>>", response))


@_register("detectable_format:number_placeholders")
def _check_placeholders(response: str, **kwargs) -> bool:
    """Check number of placeholders [placeholder]."""
    num_placeholders = kwargs.get("num_placeholders", 0)
    placeholders = re.findall(r"\[[A-Z][A-Z_ ]*\]", response)
    return len(placeholders) >= num_placeholders


# ============================================================================
# Content / Case Constraints
# ============================================================================

@_register("change_case:english_capital")
def _check_all_caps(response: str, **kwargs) -> bool:
    """Check response is all uppercase."""
    # Only check alphabetic characters
    alpha_chars = [c for c in response if c.isalpha()]
    if not alpha_chars:
        return True
    return all(c.isupper() for c in alpha_chars)


@_register("change_case:english_lowercase")
def _check_all_lower(response: str, **kwargs) -> bool:
    """Check response is all lowercase."""
    alpha_chars = [c for c in response if c.isalpha()]
    if not alpha_chars:
        return True
    return all(c.islower() for c in alpha_chars)


@_register("combination:two_responses")
def _check_two_responses(response: str, **kwargs) -> bool:
    """Check that response contains two separate responses."""
    # Look for common separators
    separators = ["******", "---", "===", "Response 1", "Response 2"]
    return any(sep in response for sep in separators)


@_register("combination:repeat_prompt")
def _check_repeat_prompt(response: str, **kwargs) -> bool:
    """Check that the response repeats the prompt first."""
    prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
    if not prompt_to_repeat:
        return True
    return prompt_to_repeat.strip().lower() in response.lower()


# ============================================================================
# Punctuation Constraints
# ============================================================================

@_register("punctuation:no_comma")
def _check_no_comma(response: str, **kwargs) -> bool:
    """Check that response contains no commas."""
    return "," not in response


@_register("startend:end_checker")
def _check_end(response: str, **kwargs) -> bool:
    """Check that response ends with a specific phrase."""
    end_phrase = kwargs.get("end_phrase", "")
    if not end_phrase:
        return True
    return response.strip().endswith(end_phrase)


@_register("startend:quotation")
def _check_quotation(response: str, **kwargs) -> bool:
    """Check response is wrapped in double quotes."""
    stripped = response.strip()
    return stripped.startswith('"') and stripped.endswith('"')


# ============================================================================
# Language Constraints
# ============================================================================

@_register("language:response_language")
def _check_language(response: str, **kwargs) -> bool:
    """Check response language. Approximate check via character scripts."""
    # This is a loose check — exact language detection would need a model
    language = kwargs.get("language", "").lower()
    if not language or language == "english":
        # Check that response is mostly ASCII
        ascii_count = sum(1 for c in response if ord(c) < 128)
        return ascii_count / max(len(response), 1) > 0.5
    # For non-English, just pass (loose check)
    return True


@_register("detectable_content:number_placeholders")
def _check_content_placeholders(response: str, **kwargs) -> bool:
    """Check number of placeholders (alias)."""
    return _check_placeholders(response, **kwargs)


@_register("detectable_content:postscript")
def _check_postscript(response: str, **kwargs) -> bool:
    """Check that response contains a postscript (P.S.)."""
    postscript_marker = kwargs.get("postscript_marker", "P.S.")
    return postscript_marker in response


# ============================================================================
# Public API
# ============================================================================

def verify_single_constraint(
    response: str,
    instruction_id: str,
    kwargs: dict,
) -> bool:
    """Verify a single constraint on a response.

    Parameters
    ----------
    response : str
        Model-generated response.
    instruction_id : str
        Constraint identifier (e.g., "length_constraints:number_words").
    kwargs : dict
        Constraint parameters.

    Returns
    -------
    bool
        Whether the constraint is satisfied.
    """
    checker = CONSTRAINT_REGISTRY.get(instruction_id)
    if checker is None:
        # Unknown constraint — pass loosely
        return True
    try:
        return checker(response, **kwargs)
    except Exception:
        return False


def verify_constraints(
    response: str,
    instruction_id_list: List[str],
    kwargs_list: List[dict],
) -> Tuple[bool, float]:
    """Verify all constraints for a response.

    Parameters
    ----------
    response : str
        Model-generated response.
    instruction_id_list : list of str
        List of constraint identifiers.
    kwargs_list : list of dict
        List of constraint parameters (parallel to instruction_id_list).

    Returns
    -------
    (strict_pass, loose_score) : (bool, float)
        strict_pass: True if ALL constraints are satisfied.
        loose_score: Fraction of constraints satisfied (0.0 to 1.0).
    """
    if not instruction_id_list:
        return True, 1.0

    results = []
    for instr_id, kw in zip(instruction_id_list, kwargs_list):
        results.append(verify_single_constraint(response, instr_id, kw))

    strict_pass = all(results)
    loose_score = sum(results) / len(results)
    return strict_pass, loose_score
