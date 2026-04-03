#!/usr/bin/env python3
"""
Sandboxed code execution for HumanEval and MBPP evaluation.

Single source of truth for code extraction and execution. Both
compute_code_quality.py and compute_all_quality.py should import from here.

Functions:
    execute_code_with_tests: Run code+tests in a subprocess
    extract_code_from_markdown: Extract code from markdown blocks
    extract_expected_fn_name: Get expected function name from test assertions
    rename_function: Rename a function definition + recursive calls
    prepare_humaneval_execution: Build executable string for HumanEval
    prepare_mbpp_execution: Build executable string for MBPP
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple


def execute_code_with_tests(
    code: str,
    timeout: float = 5.0,
    max_memory_mb: int = 256,
) -> Tuple[bool, str]:
    """Execute code in an isolated subprocess with timeout and memory limits.

    Parameters
    ----------
    code : str
        Complete Python code to execute (includes function + tests).
    timeout : float
        Maximum execution time in seconds.
    max_memory_mb : int
        Maximum memory in MB (via resource.setrlimit in child).

    Returns
    -------
    (passed, output) : (bool, str)
        Whether execution succeeded without errors, and stdout/stderr.
    """
    wrapper = f"""
import resource
import sys

# Set memory limit
mem_bytes = {max_memory_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except (ValueError, resource.error):
    pass  # Some systems don't support RLIMIT_AS

{code}
"""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(wrapper)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": "", "PYTHONHASHSEED": "0"},
        )

        output = result.stdout + result.stderr
        passed = result.returncode == 0
        return passed, output.strip()

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Execution error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


# ---------------------------------------------------------------------------
# Code extraction utilities
# ---------------------------------------------------------------------------


def extract_code_from_markdown(text: str, entry_point: str = "") -> str:
    """Extract Python code from markdown code blocks in model output.

    Handles instruction-tuned and no-CoT models that wrap code in
    ```python ... ``` blocks within prose explanations.

    If entry_point is given and found in a block, that block is preferred.
    If no markdown blocks are found, returns the original text unchanged
    (assumed to be a direct code continuation).
    """
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text

    # Prefer the block containing the target function
    if entry_point:
        for block in blocks:
            if f"def {entry_point}" in block:
                return block.strip()

    # Fall back to the longest block (most likely the main function)
    return max(blocks, key=len).strip()


def extract_expected_fn_name(test_assertions: List[str]) -> Optional[str]:
    """Extract the expected function name from MBPP test assertions.

    Looks for patterns like ``assert function_name(...)`` in the test list.
    """
    for test in test_assertions:
        match = re.search(r"assert\s+(\w+)\s*\(", test)
        if match:
            return match.group(1)
    return None


def rename_function(code: str, old_name: str, new_name: str) -> str:
    """Rename a function definition and all its call sites in code."""
    # Rename definition
    code = re.sub(
        rf"\bdef\s+{re.escape(old_name)}\b",
        f"def {new_name}",
        code,
    )
    # Rename calls (including recursive)
    code = re.sub(
        rf"\b{re.escape(old_name)}\s*\(",
        f"{new_name}(",
        code,
    )
    return code


def _extract_and_align_code(
    raw_output: str,
    entry_point: str = "",
) -> str:
    """Extract code from model output and align function name if needed.

    This is the core extraction logic shared by both HumanEval and MBPP.
    Handles: CoT stripping residuals, markdown extraction, function renaming.
    """
    code = extract_code_from_markdown(raw_output, entry_point)

    # If we have an expected function name, ensure the generated function matches
    if entry_point:
        fn_match = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
        if fn_match and fn_match.group(1) != entry_point:
            code = rename_function(code, fn_match.group(1), entry_point)

    return code


# ---------------------------------------------------------------------------
# Execution preparation
# ---------------------------------------------------------------------------


def prepare_humaneval_execution(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
) -> str:
    """Build executable code for a HumanEval problem.

    Handles two output formats:
    1. Direct continuation (base models): concatenate prompt + completion
    2. Standalone function in markdown (instruction/no-CoT models): extract
       and use directly, renaming if needed

    Parameters
    ----------
    prompt : str
        Function signature and docstring.
    completion : str
        Model-generated function body or full response with code blocks.
    test_code : str
        Test function from the dataset (defines `check`).
    entry_point : str
        Function name to test (e.g., "add").

    Returns
    -------
    str
        Complete executable Python code.
    """
    extracted = _extract_and_align_code(completion, entry_point)
    if f"def {entry_point}" in extracted:
        # Standalone function — use directly
        return f"{extracted}\n\n{test_code}\n\ncheck({entry_point})\n"

    # Standard continuation mode — prepend prompt
    return f"{prompt}{completion}\n\n{test_code}\n\ncheck({entry_point})\n"


def prepare_mbpp_execution(
    generated_code: str,
    test_assertions: List[str],
) -> str:
    """Build executable code for an MBPP problem.

    Extracts code from markdown blocks if present, then renames the
    generated function to match the name expected by test assertions.

    Parameters
    ----------
    generated_code : str
        Complete function generated by the model (may contain markdown).
    test_assertions : list[str]
        List of assert statements from the dataset.

    Returns
    -------
    str
        Complete executable Python code.
    """
    expected_fn = extract_expected_fn_name(test_assertions)
    code = _extract_and_align_code(generated_code, entry_point=expected_fn or "")
    tests = "\n".join(test_assertions)
    return f"{code}\n\n{tests}\n"
