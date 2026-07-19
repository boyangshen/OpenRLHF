"""Multi-agent GRPO trainer for OpenRLHF.

Extends :class:`PPOTrainer` to use :class:`MultiAgentSamplesGenerator` for
the 3-stage Thinker -> Executor -> Verifier rollout. The training loop
(``fit``), experience making (``train_step``), and PPO optimization
(``ppo_train``) are inherited unchanged — only the sample generator is
swapped, so the combined 3-role batch flows through the standard GRPO
advantage computation and actor update.

Per-role GRPO normalization is achieved by ordering the combined experiences
as ``[t_p0_s0..sN-1, e_p0_s0..sN-1, v_p0_s0..sN-1, ...]`` so that
``compute_advantages_and_returns`` with ``group_norm`` and
``n_samples_per_prompt=N`` produces ``3*M`` independent groups.
"""

from __future__ import annotations

import ray

from openrlhf.trainer.ppo_trainer import PPOTrainer
from openrlhf.utils.logging_utils import init_logger

from .prompt_templates import EXECUTOR, THINKER, VERIFIER
from .samples_generator import MultiAgentSamplesGenerator

logger = init_logger(__name__)


@ray.remote
class MultiAgentTrainer(PPOTrainer):
    """Single-model, multi-role GRPO trainer.

    One LLM plays three roles (Thinker + Executor + Verifier) via different
    system prompts. All three roles are trained simultaneously every step
    using a combined batch.

    Extra config (read from ``args`` via CLI):
        ``--multi_agent.thinker_max_response``    (int,   default 4096)
        ``--multi_agent.executor_max_response``   (int,   default 6144)
        ``--multi_agent.verifier_max_response``   (int,   default 1024)
        ``--multi_agent.verifier_reward_weight``  (float, default 0.3)
    """

    def __init__(
        self,
        pretrain: str,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_group,
        reference_model_group,
        vllm_engines,
        **generate_kwargs,
    ) -> None:
        # Read multi-agent config from the hierarchized args namespace.
        ma_cfg = getattr(strategy.args, "multi_agent", None)
        role_max_response = {
            THINKER: getattr(ma_cfg, "thinker_max_response", 4096) if ma_cfg else 4096,
            EXECUTOR: getattr(ma_cfg, "executor_max_response", 6144) if ma_cfg else 6144,
            VERIFIER: getattr(ma_cfg, "verifier_max_response", 1024) if ma_cfg else 1024,
        }
        verifier_reward_weight = (
            getattr(ma_cfg, "verifier_reward_weight", 0.3) if ma_cfg else 0.3
        )

        # Force-disable the reward model: multi-agent rewards are computed
        # inside the samples generator after all 3 stages complete.
        # We keep reward_model_group=None so make_experience skips RM forward.
        reward_model_group = None

        # Base PPOTrainer.__init__ loads tokenizer, datasets, and BasePPOTrainer.
        # We call super().__init__ first, then replace the samples_generator.
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            **generate_kwargs,
        )

        # Replace the default SamplesGenerator with the multi-agent variant.
        self.samples_generator = MultiAgentSamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=self.tokenizer,
            vllm_engines=vllm_engines,
            role_max_response=role_max_response,
            verifier_reward_weight=verifier_reward_weight,
        )

        # M1: Multi-agent generates 3x the experiences per step (Thinker +
        # Executor + Verifier). The base prepare_datasets() computes
        # max_steps = len(prompts) * n_samples_per_prompt // batch_size * ...,
        # which undercounts by 3x. Correct it so the LR scheduler reaches
        # min_lr at the right time.
        self.max_steps = self.max_steps * 3

        logger.info(
            f"[MultiAgentTrainer] initialized with role_max_response={role_max_response}, "
            f"verifier_reward_weight={verifier_reward_weight}, "
            f"max_steps={self.max_steps} (3x base for multi-agent)"
        )
