"""Prompt templates for the multi-agent (Thinker + Executor + Verifier) trainer.

Three-stage pipeline:
  Stage 1 (Thinker):   problem -> thought (reasoning approach, no final answer)
  Stage 2 (Executor):  (problem, thought) -> final answer (boxed)
  Stage 3 (Verifier):  (problem, thought, answer) -> verdict (correct/incorrect + reason)

Design notes:
  - Thinker must NOT produce a \\boxed{...} answer, so the Executor is forced
    to do its own computation rather than copy-pasting.
  - Verifier outputs a structured verdict so it can be parsed programmatically.
"""

from __future__ import annotations


# ============================================================================
# Role constants
# ============================================================================

THINKER = "thinker"
EXECUTOR = "executor"
VERIFIER = "verifier"

ALL_ROLES = (THINKER, EXECUTOR, VERIFIER)


# ============================================================================
# System instructions
# ============================================================================

THINKER_SYSTEM_INSTRUCTION = (
    "You are a math thinker. Read the problem carefully and write ONLY the "
    "solving thought, key observations, and the high-level approach. "
    "Do NOT compute or reveal the final numerical answer. Do NOT use a "
    "\\boxed{...} expression. Stop right before you would state the final "
    "answer."
)

EXECUTOR_SYSTEM_INSTRUCTION = (
    "You are a math executor. You are given a problem and a thinker's thought. "
    "Use the thought as guidance to write the final solution and state the "
    "final answer in a \\boxed{...} expression on the last line."
)

VERIFIER_SYSTEM_INSTRUCTION = (
    "You are a math verifier. You are given a problem, a thinker's thought, "
    "and an executor's answer. Your job is to verify whether the final answer "
    "is correct.\n\n"
    "Respond in exactly this format:\n"
    "Verdict: CORRECT\n"
    "or\n"
    "Verdict: INCORRECT\n\n"
    "Then briefly explain your reasoning (1-3 sentences)."
)


# ============================================================================
# Chat-style prompts (for instruct/chat models with apply_chat_template)
# ============================================================================


def build_thinker_chat(problem: str) -> list[dict]:
    """Chat prompt for the Thinker agent.

    Args:
        problem: The original math problem text.

    Returns:
        A list of chat messages [{"role": "system", ...}, {"role": "user", ...}].
    """
    return [
        {"role": "system", "content": THINKER_SYSTEM_INSTRUCTION},
        {"role": "user", "content": problem},
    ]


def build_executor_chat(problem: str, thought: str) -> list[dict]:
    """Chat prompt for the Executor agent.

    Args:
        problem: The original math problem text.
        thought: The Thinker's reasoning output (cleaned of any \\boxed{...}).

    Returns:
        A list of chat messages.
    """
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": f"Problem:\n{problem}\n\nThinker's thought:\n{thought}",
        },
    ]


def build_verifier_chat(problem: str, thought: str, answer: str) -> list[dict]:
    """Chat prompt for the Verifier agent.

    Args:
        problem: The original math problem text.
        thought: The Thinker's reasoning output.
        answer: The Executor's final answer (including \\boxed{...}).

    Returns:
        A list of chat messages.
    """
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                f"Thinker's thought:\n{thought}\n\n"
                f"Executor's answer:\n{answer}"
            ),
        },
    ]


# ============================================================================
# Plain-text prompts (for base models without chat template)
# ============================================================================


def build_thinker_prompt(problem: str) -> str:
    """Plain-text prompt for a base model (no chat template)."""
    return (
        f"{THINKER_SYSTEM_INSTRUCTION}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Thought (no final answer):\n"
    )


def build_executor_prompt(problem: str, thought: str) -> str:
    """Plain-text prompt for a base model (no chat template)."""
    return (
        f"{EXECUTOR_SYSTEM_INSTRUCTION}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Thinker's thought:\n{thought}\n\n"
        f"Final solution (end with \\boxed{{...}}):\n"
    )


def build_verifier_prompt(problem: str, thought: str, answer: str) -> str:
    """Plain-text prompt for a base model (no chat template)."""
    return (
        f"{VERIFIER_SYSTEM_INSTRUCTION}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Thinker's thought:\n{thought}\n\n"
        f"Executor's answer:\n{answer}\n\n"
        f"Verdict:\n"
    )
