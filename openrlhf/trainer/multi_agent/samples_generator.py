"""Multi-agent samples generator: 3-stage Thinker -> Executor -> Verifier rollout.

Extends :class:`SamplesGenerator` to run a single model through three roles
in sequence, then return a combined experience list ordered so that
``compute_advantages_and_returns`` (with ``group_norm``) produces per-role
per-problem advantage groups when reshaping by ``n_samples_per_prompt``.

Experience ordering (M = #problems, N = n_samples_per_prompt):

    [t_p0_s0..sN-1, e_p0_s0..sN-1, v_p0_s0..sN-1,
     t_p1_s0..sN-1, e_p1_s0..sN-1, v_p1_s0..sN-1,
     ...]

``reshape(-1, N)`` yields ``3*M`` groups, each containing the N samples of a
single (role, problem) pair — exactly what GRPO needs for group-normalized
advantages.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import ray
import torch

from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.trainer.ppo_utils.samples_generator import SamplesGenerator, _collect_prompt_batch
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.math_utils import extract_boxed_answer, grade_answer

from .prompt_templates import (
    EXECUTOR,
    THINKER,
    VERIFIER,
    build_executor_chat,
    build_executor_prompt,
    build_thinker_chat,
    build_thinker_prompt,
    build_verifier_chat,
    build_verifier_prompt,
)

logger = init_logger(__name__)


# ============================================================================
# Helper functions
# ============================================================================


def _strip_final_boxed(text: str) -> str:
    """Remove a trailing ``\\boxed{...}`` from text, handling nested braces.

    Safety net: if the Thinker accidentally produces a boxed answer, we strip
    it so the Executor cannot just copy-paste. Handles nested braces such as
    ``\\boxed{\\frac{1}{2}}`` which the naive regex ``[^{}]*`` cannot match.
    """
    text = text.rstrip()
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text
    # Find the matching closing brace by tracking brace depth.
    brace_open = idx + len("\\boxed{") - 1  # position of the '{'
    depth = 0
    for i in range(brace_open, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                # Matching close brace at i. Only strip if only whitespace
                # follows (i.e., the boxed is truly trailing).
                if text[i + 1:].strip() == "":
                    return text[:idx].rstrip()
                return text  # \boxed{...} is not trailing; leave as-is.
    return text  # unbalanced braces; return unchanged


def _parse_verifier_verdict(text: str) -> Optional[bool]:
    """Parse a verifier response into a boolean verdict.

    Looks for ``CORRECT`` or ``INCORRECT`` (case-insensitive). Returns
    ``True`` for correct, ``False`` for incorrect, ``None`` if unparseable.
    """
    text_lower = text.lower().strip()
    if "verdict:" in text_lower:
        after_verdict = text_lower.split("verdict:")[-1].strip()
        if after_verdict.startswith("correct"):
            return True
        if after_verdict.startswith("incorrect"):
            return False
    if "correct" in text_lower and "incorrect" not in text_lower:
        return True
    if "incorrect" in text_lower:
        return False
    return None


# ============================================================================
# Multi-agent samples generator
# ============================================================================


class MultiAgentSamplesGenerator(SamplesGenerator):
    """Three-stage rollout generator for single-model multi-agent GRPO.

    One model plays three roles (Thinker + Executor + Verifier) via different
    system prompts. All three roles are trained simultaneously every step
    using a combined batch.

    Args:
        strategy: DeepSpeed strategy.
        prompts_dataloader: Training prompt dataloader.
        eval_dataloader: Evaluation prompt dataloader.
        tokenizer: HF tokenizer.
        vllm_engines: List of vLLM Ray actors.
        role_max_response: Per-role max response tokens, e.g.
            ``{"thinker": 4096, "executor": 6144, "verifier": 1024}``.
        verifier_reward_weight: Weight for the verifier agreement term in
            Thinker/Executor rewards.
    """

    def __init__(
        self,
        strategy,
        prompts_dataloader,
        eval_dataloader,
        tokenizer,
        vllm_engines: List,
        role_max_response: Optional[dict] = None,
        verifier_reward_weight: float = 0.3,
    ):
        super().__init__(strategy, prompts_dataloader, eval_dataloader, tokenizer, vllm_engines)

        self.role_max_response = role_max_response or {
            THINKER: 4096,
            EXECUTOR: 6144,
            VERIFIER: 1024,
        }
        self.verifier_reward_weight = float(verifier_reward_weight)

    # ── Public entry point ──────────────────────────────────────────────

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Draw one batch of prompts and run the 3-stage rollout.

        Returns:
            (rollout_samples, filter_pass_rate, prompts_consumed, exhausted)
        """
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)

        num_prompts = self.args.rollout.batch_size
        prompts, labels, images, exhausted = _collect_prompt_batch(self._dataloader_iter, num_prompts)
        if not prompts:
            return [], None, 0, True

        rollout_samples = self.three_stage_rollout(prompts, labels, **generate_kwargs)
        return rollout_samples, None, len(prompts), exhausted

    # ── Three-stage pipeline ────────────────────────────────────────────

    @torch.no_grad()
    def three_stage_rollout(self, problems: List[str], labels: List[str], **generate_kwargs) -> List[Experience]:
        """Run the full Thinker -> Executor -> Verifier pipeline.

        The same model is used for all three roles; only the prompt and
        ``max_new_tokens`` differ per stage.

        Returns a combined experience list ordered for per-role per-problem
        GRPO group normalization.
        """
        n_samples = self.args.rollout.n_samples_per_prompt
        M = len(problems)

        if self.args.vllm.enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        try:
            # ── Stage 1: Thinker (M problems × N samples) ──
            thinker_prompts = [self._build_role_prompt(THINKER, p) for p in problems]
            thinker_exps = self._generate_stage(
                thinker_prompts, labels, THINKER, n_samples, **generate_kwargs
            )
            # thinker_exps: M*N experiences, ordered [p0_s0..sN-1, p1_s0..sN-1, ...]

            thinker_texts = [self._decode_response(exp) for exp in thinker_exps]
            thinker_texts = [_strip_final_boxed(t) for t in thinker_texts]

            # ── Stage 2: Executor (M*N prompts × 1 sample) ──
            executor_prompts: List[str] = []
            executor_labels: List[str] = []
            for i in range(M):
                for j in range(n_samples):
                    idx = i * n_samples + j
                    executor_prompts.append(
                        self._build_role_prompt(EXECUTOR, problems[i], thought=thinker_texts[idx])
                    )
                    executor_labels.append(labels[i])

            executor_exps = self._generate_stage(
                executor_prompts, executor_labels, EXECUTOR, 1, **generate_kwargs
            )
            # executor_exps: M*N experiences, ordered [p0_s0, p0_s1, ..., p0_sN-1, p1_s0, ...]

            executor_texts = [self._decode_response(exp) for exp in executor_exps]

            # ── Stage 3: Verifier (M*N prompts × 1 sample) ──
            verifier_prompts: List[str] = []
            verifier_labels: List[str] = []
            for i in range(M):
                for j in range(n_samples):
                    idx = i * n_samples + j
                    verifier_prompts.append(
                        self._build_role_prompt(
                            VERIFIER,
                            problems[i],
                            thought=thinker_texts[idx],
                            answer=executor_texts[idx],
                        )
                    )
                    verifier_labels.append(labels[i])

            verifier_exps = self._generate_stage(
                verifier_prompts, verifier_labels, VERIFIER, 1, **generate_kwargs
            )

            # ── Compute per-role rewards ──
            rewards = self._compute_multi_agent_rewards(
                labels, executor_texts, verifier_exps, M, n_samples
            )

            # Attach rewards to experiences. Use .clone() for info["reward"]
            # to avoid aliasing the same tensor object (safe against later
            # in-place modifications on either field).
            for exp, r in zip(thinker_exps, rewards[THINKER]):
                exp.rewards = torch.tensor([r], dtype=torch.float32)
                exp.info["reward"] = exp.rewards.clone()
            for exp, r in zip(executor_exps, rewards[EXECUTOR]):
                exp.rewards = torch.tensor([r], dtype=torch.float32)
                exp.info["reward"] = exp.rewards.clone()
            for exp, r in zip(verifier_exps, rewards[VERIFIER]):
                exp.rewards = torch.tensor([r], dtype=torch.float32)
                exp.info["reward"] = exp.rewards.clone()
        finally:
            if self.args.vllm.enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")

        # ── Interleave: [t_p0_s0..sN-1, e_p0_s0..sN-1, v_p0_s0..sN-1, ...] ──
        combined: List[Experience] = []
        for i in range(M):
            base = i * n_samples
            combined.extend(thinker_exps[base : base + n_samples])
            combined.extend(executor_exps[base : base + n_samples])
            combined.extend(verifier_exps[base : base + n_samples])

        logger.info(
            f"[MultiAgent] 3-stage rollout done: M={M}, N={n_samples}, "
            f"total_samples={len(combined)} "
            f"(thinker={len(thinker_exps)}, executor={len(executor_exps)}, verifier={len(verifier_exps)})"
        )
        return combined

    # ── Single-stage generation ─────────────────────────────────────────

    def _generate_stage(
        self,
        prompts: List[str],
        labels: List[str],
        role: str,
        n_samples: int,
        **generate_kwargs,
    ) -> List[Experience]:
        """Generate experiences for one role.

        Uses ``ray.get`` on all refs to preserve prompt order (critical for
        correct GRPO group reshaping downstream).
        """
        stage_kwargs = dict(generate_kwargs)
        stage_kwargs["max_new_tokens"] = self.role_max_response[role]
        stage_kwargs["n_samples_per_prompt"] = n_samples

        refs = self._dispatch_prompts_to_vllm(prompts, labels, images=None, **stage_kwargs)
        results = ray.get(refs)  # ordered by prompt index

        experiences: List[Experience] = []
        for ref_results in results:
            for resp in ref_results:
                experiences.append(self._process_response_into_experience(resp, **stage_kwargs))

        return experiences

    # ── Prompt construction ─────────────────────────────────────────────

    def _build_role_prompt(self, role: str, problem: str, thought: str = None, answer: str = None) -> str:
        """Build a role-specific prompt string.

        Uses the tokenizer's chat template if available; otherwise falls back
        to plain-text prompts for base models.
        """
        if role == THINKER:
            messages = build_thinker_chat(problem)
        elif role == EXECUTOR:
            messages = build_executor_chat(problem, thought)
        elif role == VERIFIER:
            messages = build_verifier_chat(problem, thought, answer)
        else:
            raise ValueError(f"Unknown role: {role}")

        # Prefer chat template for instruct models.
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(
                    f"apply_chat_template failed for role={role}: {e}; falling back to plain prompt."
                )

        # Plain-text fallback for base models.
        if role == THINKER:
            return build_thinker_prompt(problem)
        if role == EXECUTOR:
            return build_executor_prompt(problem, thought)
        return build_verifier_prompt(problem, thought, answer)

    # ── Response decoding ───────────────────────────────────────────────

    def _decode_response(self, exp: Experience) -> str:
        """Extract the response text from a single-sample Experience.

        ``action_mask`` covers positions 1..T-1 of ``sequences`` (shifted by 1
        due to next-token prediction alignment). Response tokens are
        ``sequences[0, 1:][action_mask[0].bool()]``.
        """
        sequences = exp.sequences[0]  # (T,)
        action_mask = exp.action_mask[0]  # (T-1,)
        response_tokens = sequences[1:][action_mask.bool()]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)

    # ── Reward computation ──────────────────────────────────────────────

    def _compute_multi_agent_rewards(
        self,
        labels: List[str],
        executor_texts: List[str],
        verifier_exps: List[Experience],
        M: int,
        n_samples: int,
    ) -> dict:
        """Compute per-role scalar rewards.

        Reward design:
            - Thinker  : gt_correct + weight * verifier_agreement
            - Executor : gt_correct + weight * verifier_agreement
            - Verifier : verifier_agreement (did it agree with ground truth?)

        ``gt_correct`` ∈ {+1, -1}; ``verifier_agreement`` ∈ {+1, -1, -0.5}.
        """
        total = M * n_samples

        # Ground-truth correctness from Executor's final answer.
        gt_correct: List[float] = []
        for i in range(total):
            label = labels[i // n_samples]
            pred = extract_boxed_answer(executor_texts[i])
            is_correct = grade_answer(pred, label)
            gt_correct.append(1.0 if is_correct else -1.0)

        # Verifier verdicts.
        verifier_texts = [self._decode_response(exp) for exp in verifier_exps]
        verifier_verdicts = [_parse_verifier_verdict(t) for t in verifier_texts]

        verifier_agreement: List[float] = []
        for verdict, gt in zip(verifier_verdicts, gt_correct):
            if verdict is None:
                verifier_agreement.append(-0.5)
            elif verdict == (gt > 0):
                verifier_agreement.append(1.0)
            else:
                verifier_agreement.append(-1.0)

        # Per-role rewards.
        thinker_rewards: List[float] = []
        executor_rewards: List[float] = []
        verifier_rewards: List[float] = []
        for i in range(total):
            gt = gt_correct[i]
            va = verifier_agreement[i]
            thinker_rewards.append(gt + self.verifier_reward_weight * va)
            executor_rewards.append(gt + self.verifier_reward_weight * va)
            verifier_rewards.append(va)

        # Logging summary.
        gt_rate = sum(1 for v in gt_correct if v > 0) / max(total, 1)
        va_rate = sum(1 for v in verifier_agreement if v > 0) / max(total, 1)
        logger.info(
            f"[MultiAgent] rewards: gt_correct_rate={gt_rate:.3f}, "
            f"verifier_agreement_rate={va_rate:.3f}, "
            f"thinker_mean={sum(thinker_rewards)/max(total,1):.3f}, "
            f"executor_mean={sum(executor_rewards)/max(total,1):.3f}, "
            f"verifier_mean={sum(verifier_rewards)/max(total,1):.3f}"
        )

        return {
            THINKER: thinker_rewards,
            EXECUTOR: executor_rewards,
            VERIFIER: verifier_rewards,
        }
