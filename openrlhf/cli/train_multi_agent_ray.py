"""CLI entry point for multi-agent (Thinker + Executor + Verifier) GRPO training.

Mirrors :mod:`openrlhf.cli.train_ppo_ray` but uses
:class:`MultiAgentTrainer` and adds ``--multi_agent.*`` arguments.

Required configuration:
    - ``--algo.advantage.estimator group_norm`` (GRPO-style per-role normalization)
    - ``--rollout.n_samples_per_prompt N`` with N > 1 (needed for group_norm)
    - ``--data.apply_chat_template`` MUST NOT be set; the generator applies the
      role-specific chat template internally. Setting it would cause double
      template application (the dataset pre-formats the prompt, then the
      generator wraps it again), corrupting all role prompts.

Example:
    python -m openrlhf.cli.train_multi_agent_ray \\
        --actor.model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \\
        --data.prompt_dataset zhuzilin/dapo-math-17k \\
        --data.input_key prompt --data.label_key label \\
        --algo.advantage.estimator group_norm \\
        --rollout.n_samples_per_prompt 8 \\
        --multi_agent.thinker_max_response 4096 \\
        --multi_agent.executor_max_response 6144 \\
        --multi_agent.verifier_max_response 1024 \\
        --multi_agent.verifier_reward_weight 0.3 \\
        ...
"""

import argparse
import os
from datetime import datetime

import ray
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.utils import get_strategy


def train(args):
    """Set up Ray, models, and launch the MultiAgentTrainer."""
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
                    "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
                    "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": os.environ.get(
                        "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "1"
                    ),
                }
            }
        )

    strategy = get_strategy(args)
    strategy.print(args)

    # ── Placement group for colocated actor + ref ──
    pg = None
    if args.train.colocate_actor_ref or args.train.colocate_all:
        if args.algo.kl.init_coef > 0:
            assert (
                args.actor.num_nodes == args.ref.num_nodes
                and args.actor.num_gpus_per_node == args.ref.num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must match when colocating actor and ref."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor.num_nodes * args.actor.num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # ── vLLM engines ──
    vllm_engines = None
    if args.vllm.num_engines is not None and args.vllm.num_engines > 0:
        max_len = args.data.max_len
        if args.train.colocate_all and not args.train.async_enable:
            assert (
                args.actor.num_nodes * args.actor.num_gpus_per_node
                == args.vllm.num_engines * args.vllm.tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must equal "
                f"vllm_num_engines * vllm_tensor_parallel_size, got "
                f"{args.actor.num_nodes * args.actor.num_gpus_per_node} "
                f"and {args.vllm.num_engines * args.vllm.tensor_parallel_size}"
            )

        vllm_engines = create_vllm_engines(
            args.vllm.num_engines,
            args.vllm.tensor_parallel_size,
            args.actor.model_name_or_path,
            args.train.seed,
            args.train.full_determinism_enable,
            args.vllm.enable_prefix_caching,
            args.vllm.enforce_eager,
            max_len,
            pg if args.train.colocate_all and not args.train.async_enable else None,
            args.vllm.gpu_memory_utilization,
            args.vllm.enable_sleep,
            "processed_logprobs" if args.algo.advantage.is_correction_enable else None,
            agent_func_path=None,  # Multi-agent generator handles prompting internally.
            remote_rm_url=None,    # Rewards are computed in the generator.
            max_images_per_prompt=getattr(args.data, "max_images_per_prompt", 0),
        )

    # ── Actor model ──
    actor_model = RayActorGroup(
        args.actor.num_nodes,
        args.actor.num_gpus_per_node,
        PolicyModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ds.ring_attn_size * args.ds.tensor_parallel_size,
    )

    # ── Reference model (optional, for KL) ──
    if args.algo.kl.init_coef > 0:
        ref_model = RayActorGroup(
            args.ref.num_nodes,
            args.ref.num_gpus_per_node,
            ReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ds.ring_attn_size * args.ds.tensor_parallel_size,
        )
    else:
        ref_model = None

    if not args.train.colocate_all:
        pg = None

    # ── Critic / Reward model: NOT used in multi-agent GRPO ──
    # group_norm (GRPO) does not use a critic. Rewards are computed inside
    # the samples generator, so no reward model is needed.
    critic_model = None
    reward_model = None

    # ── MultiAgentTrainer ──
    from openrlhf.trainer.multi_agent.trainer import MultiAgentTrainer

    ppo_trainer = MultiAgentTrainer.remote(
        args.actor.model_name_or_path,
        strategy,
        actor_model,
        critic_model,
        reward_model,
        ref_model,
        vllm_engines,
        # generate kwargs
        do_sample=True,
        max_len=max_len,
        max_new_tokens=args.rollout.max_new_tokens,
        temperature=args.rollout.temperature,
        top_p=args.rollout.top_p,
    )

    # ── Init model weights ──
    max_steps = ray.get(ppo_trainer.get_max_steps.remote())

    refs = []
    refs.extend(
        actor_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path, max_steps, vllm_engines)
    )
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path))
    ray.get(refs)

    # ── Train ──
    ray.get(ppo_trainer.fit.remote())

    # ── Save final model ──
    ray.get(actor_model.async_save_model())


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all PPO + multi-agent arguments."""
    parser = argparse.ArgumentParser()

    # ── Ray and vLLM ──
    parser.add_argument("--ref.num_nodes", type=int, default=1)
    parser.add_argument("--ref.num_gpus_per_node", type=int, default=8)
    parser.add_argument("--reward.num_nodes", type=int, default=1)
    parser.add_argument("--reward.num_gpus_per_node", type=int, default=8)
    parser.add_argument("--train.colocate_actor_ref", action="store_true", default=False)
    parser.add_argument("--actor.num_nodes", type=int, default=1)
    parser.add_argument("--actor.num_gpus_per_node", type=int, default=8)
    parser.add_argument("--critic.num_nodes", type=int, default=1)
    parser.add_argument("--critic.num_gpus_per_node", type=int, default=8)
    parser.add_argument("--train.colocate_critic_reward", action="store_true", default=False)
    parser.add_argument("--train.colocate_all", action="store_true", default=False)

    # ── vLLM ──
    parser.add_argument("--vllm.num_engines", type=int, default=None)
    parser.add_argument("--vllm.tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm.sync_backend", type=str, default="nccl")
    parser.add_argument("--vllm.sync_with_ray", action="store_true", default=False)
    parser.add_argument("--vllm.enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--vllm.enforce_eager", action="store_true", default=False)
    parser.add_argument("--vllm.enable_sleep", action="store_true", default=False)
    parser.add_argument("--vllm.gpu_memory_utilization", type=float, default=0.95)

    # ── IS correction ──
    parser.add_argument("--algo.advantage.is_correction_enable", action="store_true", default=False)
    parser.add_argument("--algo.advantage.is_correction_threshold", type=float, nargs=2, default=[0.5, 5.0])
    parser.add_argument(
        "--algo.advantage.is_correction_type",
        type=str,
        default="tis",
        choices=["tis", "icepop", "seq-mask-tis"],
    )

    # ── Async training (not typically used with multi-agent) ──
    parser.add_argument("--train.async_enable", action="store_true", default=False)
    parser.add_argument("--train.async_queue_size", type=int, default=1)
    parser.add_argument("--train.partial_rollout_enable", action="store_true", default=False)

    # ── Checkpoints ──
    parser.add_argument("--eval.steps", type=int, default=-1)
    parser.add_argument("--ckpt.save_steps", type=int, default=-1)
    parser.add_argument("--logger.logging_steps", type=int, default=1)
    parser.add_argument("--ckpt.path", type=str, default="./ckpt/checkpoints_multi_agent_ray")
    parser.add_argument("--ckpt.save_hf", action="store_true", default=False)
    parser.add_argument("--ckpt.disable_ds", action="store_true", default=False)
    parser.add_argument("--ckpt.max_num", type=int, default=3)
    parser.add_argument("--ckpt.max_mem", type=float, default=float("inf"))
    parser.add_argument("--ckpt.load_enable", action="store_true", default=False)
    parser.add_argument("--ckpt.best_metric_key", type=str, default="")
    parser.add_argument("--ds.use_universal_ckpt", action="store_true", default=False)

    # ── DeepSpeed ──
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ds.zero_stage", type=int, default=2)
    parser.add_argument("--actor.gradient_checkpointing_enable", action="store_true", default=False)
    parser.add_argument("--ds.deepcompile", action="store_true", default=False)
    parser.add_argument("--ds.param_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--train.enable_ema", action="store_true", default=False)
    parser.add_argument("--train.ema_beta", type=float, default=0.992)
    parser.add_argument("--ds.zpg", type=int, default=1)
    parser.add_argument("--ds.adam_offload", action="store_true", default=False)
    parser.add_argument(
        "--ds.attn_implementation",
        type=str,
        default="flash_attention_2",
    )
    parser.add_argument(
        "--ds.experts_implementation",
        type=str,
        default=None,
        choices=["eager", "batched_mm", "grouped_mm", "deepgemm"],
    )
    parser.add_argument("--ds.use_liger_kernel", action="store_true", default=False)
    parser.add_argument("--ds.grad_accum_dtype", type=str, default=None)
    parser.add_argument("--ds.overlap_comm", action="store_true", default=False)
    parser.add_argument("--actor.gradient_checkpointing_reentrant", action="store_true", default=False)
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--data.dataloader_num_workers", type=int, default=0)
    parser.add_argument("--ds.enable_sleep", action="store_true", default=False)
    parser.add_argument("--ds.tensor_parallel_size", type=int, default=1)
    parser.add_argument("--ds.packing_samples", action="store_true", default=False)
    parser.add_argument("--train.dynamic_batch_enable", action="store_true", default=False)
    parser.add_argument("--rollout.max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train.max_tokens_per_gpu", type=int, default=16192)

    # ── LoRA ──
    parser.add_argument("--ds.load_in_4bit", action="store_true", default=False)
    parser.add_argument("--ds.lora.rank", type=int, default=0)
    parser.add_argument("--ds.lora.alpha", type=int, default=16)
    parser.add_argument("--ds.lora.target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--ds.lora.dropout", type=float, default=0)

    # ── PPO / GRPO ──
    parser.add_argument("--ckpt.output_dir", type=str, default="./ckpt")
    parser.add_argument("--train.num_episodes", type=int, default=1)
    parser.add_argument("--rollout.batch_size", type=int, default=1024)
    parser.add_argument("--rollout.vllm_generate_batch_size", type=int, default=None)
    parser.add_argument("--rollout.micro_batch_size", type=int, default=1)
    parser.add_argument("--train.max_epochs", type=int, default=1)
    parser.add_argument("--data.max_len", type=int, default=2048)
    parser.add_argument("--rollout.max_new_tokens", type=int, default=None)
    parser.add_argument("--data.max_samples", type=int, default=int(1e8))
    parser.add_argument("--actor.eps_clip", type=float, default=0.2)
    parser.add_argument("--actor.eps_clip_low_high", type=float, nargs=2, default=None)
    parser.add_argument("--actor.dual_clip", type=float, default=None)
    parser.add_argument("--critic.value_clip", type=float, default=0.5)
    parser.add_argument("--algo.advantage.lambd", type=float, default=1)
    parser.add_argument("--algo.advantage.gamma", type=float, default=1)
    parser.add_argument("--train.micro_batch_size", type=int, default=1)
    parser.add_argument("--train.batch_size", type=int, default=128)
    parser.add_argument("--reward.normalize_enable", action="store_true", default=False)
    parser.add_argument("--rollout.top_p", type=float, default=1.0)
    parser.add_argument("--rollout.temperature", type=float, default=1.0)
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument("--train.full_determinism_enable", action="store_true", default=False)
    parser.add_argument("--critic.freezing_steps", type=int, default=-1)
    parser.add_argument("--rollout.n_samples_per_prompt", type=int, default=1)
    parser.add_argument("--critic.save_value_network", action="store_true", default=False)
    parser.add_argument("--algo.kl.target", type=float, default=None)
    parser.add_argument("--algo.kl.horizon", type=int, default=10000)
    parser.add_argument("--algo.kl.init_coef", type=float, default=0.01)
    parser.add_argument("--actor.policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo"])
    parser.add_argument(
        "--algo.kl.estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
    )
    parser.add_argument("--actor.aux_loss_coef", type=float, default=0)
    parser.add_argument("--actor.entropy_coef", type=float, default=None)
    parser.add_argument("--reward.clip_range", type=float, nargs=2, default=(-10, 10))

    # ── Optimizer + scheduler + grad clip ──
    for prefix in ("actor", "critic"):
        parser.add_argument(f"--{prefix}.optim", type=str, default="adam", choices=["adam", "muon"])
        parser.add_argument(f"--{prefix}.muon.lr", type=float, default=0.02)
        parser.add_argument(f"--{prefix}.muon.momentum", type=float, default=0.95)
        parser.add_argument(f"--{prefix}.muon.ns_steps", type=int, default=5)
        parser.add_argument(f"--{prefix}.muon.nesterov", action="store_true", default=True)
        parser.add_argument(f"--{prefix}.muon.no_nesterov", dest=f"{prefix}.muon.nesterov", action="store_false")
        parser.add_argument(f"--{prefix}.adam.lr", type=float, default=1e-6 if prefix == "actor" else 9e-6)
        parser.add_argument(f"--{prefix}.adam.betas", type=float, nargs=2, default=(0.9, 0.95))
        parser.add_argument(f"--{prefix}.adam.eps", type=float, default=1e-8)
        parser.add_argument(f"--{prefix}.adam.weight_decay", type=float, default=0.0)
        parser.add_argument(f"--{prefix}.lr_scheduler", type=str, default="cosine_with_min_lr")
        parser.add_argument(f"--{prefix}.lr_warmup_ratio", type=float, default=0.03)
        parser.add_argument(f"--{prefix}.min_lr_ratio", type=float, default=0.1)
        parser.add_argument(f"--{prefix}.max_norm", type=float, default=1.0)

    # ── Reinforce/GRPO ──
    parser.add_argument(
        "--algo.advantage.estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
    )
    parser.add_argument("--algo.kl.use_loss", action="store_true", default=False)
    parser.add_argument("--algo.advantage.no_std_norm", action="store_true", default=False)
    parser.add_argument("--reward.overlong_buffer_len", type=float, default=None)
    parser.add_argument("--reward.overlong_penalty_factor", type=float, default=1)
    parser.add_argument("--reward.stop_properly_penalty_coef", type=float, default=None)

    # ── Context Parallel ──
    parser.add_argument("--ds.ring_attn_size", type=int, default=1)
    parser.add_argument("--ds.ring_attn_head_stride", type=int, default=1)

    # ── Models ──
    parser.add_argument("--actor.model_name_or_path", type=str, default=None)
    parser.add_argument("--reward.model_name_or_path", type=str, default=None)
    parser.add_argument("--reward.remote_url", type=str, default=None)
    parser.add_argument("--critic.model_name_or_path", type=str, default=None)
    parser.add_argument("--ds.value_head_prefix", type=str, default="score")
    parser.add_argument("--ref.offload", action="store_true", default=False)
    parser.add_argument("--reward.offload", action="store_true", default=False)
    parser.add_argument("--train.agent_func_path", type=str, default=None)

    # ── Dataset ──
    parser.add_argument("--data.prompt_dataset", type=str, default=None)
    parser.add_argument("--data.prompt_probs", type=str, default=None)
    parser.add_argument("--data.prompt_split", type=str, default="train")
    parser.add_argument("--eval.dataset", type=str, default=None)
    parser.add_argument("--eval.split", type=str, default="train")
    parser.add_argument("--eval.temperature", type=float, default=0.6)
    parser.add_argument("--eval.n_samples_per_prompt", type=int, default=4)
    parser.add_argument("--data.input_key", type=str, default="input")
    parser.add_argument("--data.label_key", type=str, default=None)
    parser.add_argument("--data.input_template", type=str, default=None)
    parser.add_argument("--data.apply_chat_template", action="store_true", default=False)

    # ── wandb ──
    parser.add_argument("--logger.wandb.key", type=str, default=None)
    parser.add_argument("--logger.wandb.org", type=str, default=None)
    parser.add_argument("--logger.wandb.group", type=str, default=None)
    parser.add_argument("--logger.wandb.project", type=str, default="openrlhf_train_multi_agent")
    parser.add_argument(
        "--logger.wandb.run_name",
        type=str,
        default="multi_agent_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # ── Dynamic filtering (not supported in multi-agent mode) ──
    parser.add_argument("--algo.dynamic_filtering_enable", action="store_true", default=False)
    parser.add_argument("--algo.dynamic_filtering_range", nargs=2, default=(0, 1), type=float)

    # ── VLM ──
    parser.add_argument("--data.image_key", type=str, default="images")
    parser.add_argument("--data.max_images_per_prompt", type=int, default=0)
    parser.add_argument("--actor.freeze_visual_encoder", action="store_true", default=False)

    # ── TensorBoard ──
    parser.add_argument("--logger.tensorboard_dir", type=str, default=None)

    # ── ModelScope ──
    parser.add_argument("--use_ms", action="store_true", default=False)

    # ── Multi-agent specific arguments ──
    parser.add_argument(
        "--multi_agent.thinker_max_response",
        type=int,
        default=4096,
        help="Max new tokens for the Thinker role.",
    )
    parser.add_argument(
        "--multi_agent.executor_max_response",
        type=int,
        default=6144,
        help="Max new tokens for the Executor role.",
    )
    parser.add_argument(
        "--multi_agent.verifier_max_response",
        type=int,
        default=1024,
        help="Max new tokens for the Verifier role.",
    )
    parser.add_argument(
        "--multi_agent.verifier_reward_weight",
        type=float,
        default=0.3,
        help="Weight for the verifier agreement term in Thinker/Executor rewards.",
    )

    return parser


def _validate_args(args):
    """Run multi-agent specific argument validation."""
    from openrlhf.utils.config import hierarchize

    args = hierarchize(args)

    if args.actor.eps_clip_low_high is None:
        args.actor.eps_clip_low_high = (args.actor.eps_clip, args.actor.eps_clip)

    # Multi-agent requires group_norm (GRPO) for per-role normalization.
    assert args.algo.advantage.estimator == "group_norm", (
        "Multi-agent training requires --algo.advantage.estimator group_norm "
        f"(got {args.algo.advantage.estimator})."
    )

    # group_norm requires n_samples_per_prompt > 1.
    assert args.rollout.n_samples_per_prompt > 1, (
        "Multi-agent training requires --rollout.n_samples_per_prompt > 1 "
        f"(got {args.rollout.n_samples_per_prompt})."
    )

    # Disable critic for non-GAE estimators.
    args.critic.model_name_or_path = None

    # Disable reward model and remote_rm_url: rewards are computed in the generator.
    args.reward.model_name_or_path = None
    args.reward.remote_url = None
    args.train.agent_func_path = None

    # C1: Force-disable apply_chat_template. The MultiAgentSamplesGenerator
    # applies role-specific chat templates internally via _build_role_prompt().
    # If the dataset also applies the chat template (PromptDataset with
    # apply_chat_template=True), the already-formatted string gets wrapped
    # again as a new user message, corrupting all role prompts.
    if args.data.apply_chat_template:
        print(
            "[Warning] --data.apply_chat_template is force-disabled in multi-agent "
            "mode. The generator applies role-specific chat templates internally."
        )
        args.data.apply_chat_template = False

    # M2: Evaluation is not supported. The base generate_eval_samples uses
    # single-turn rollout producing rewards=None, which crashes
    # make_experience_batch when reward_model_group is None.
    assert args.eval.steps <= 0, (
        "Multi-agent training does not support mid-training evaluation "
        "(--eval.steps). The base eval path assumes a reward model or "
        "remote_rm_url, both of which are disabled here. Set --eval.steps -1."
    )

    # m3: partial_rollout requires async_enable.
    if args.train.partial_rollout_enable:
        assert args.train.async_enable, (
            "--train.partial_rollout_enable requires --train.async_enable."
        )

    # Dynamic filtering is not supported in multi-agent mode.
    assert not args.algo.dynamic_filtering_enable, (
        "Dynamic filtering is not supported in multi-agent mode."
    )

    # VLM constraints.
    if args.data.max_images_per_prompt > 0:
        assert not args.ds.packing_samples, "VLM training does not support --packing_samples."

    if args.data.input_template and "{}" not in args.data.input_template:
        print("[Warning] '{}' not in args.data.input_template, set to None")
        args.data.input_template = None

    if args.ds.ring_attn_size > 1:
        if not args.ds.packing_samples:
            print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
            args.ds.packing_samples = True

    if args.train.dynamic_batch_enable:
        if not args.ds.packing_samples:
            print("[Warning] Please --packing_samples to accelerate when --use_dynamic_batch is enabled.")
            args.ds.packing_samples = True
        if args.rollout.max_tokens_per_gpu is None:
            args.rollout.max_tokens_per_gpu = args.train.max_tokens_per_gpu

    if args.ds.packing_samples:
        if "flash_attention" not in args.ds.attn_implementation:
            print("[Warning] Please use --attn_implementation with flash_attention when --packing_samples is enabled.")
            args.ds.attn_implementation = "flash_attention_2"
        assert args.vllm.num_engines > 0, "Only support `--packing_samples` with vLLM."

    if args.vllm.enable_sleep and not args.train.colocate_all:
        print("Set args.vllm.enable_sleep to False when args.train.colocate_all is disabled.")
        args.vllm.enable_sleep = False

    if args.train.async_enable:
        assert not args.vllm.enable_sleep, "Async RLHF is not supported with --vllm.enable_sleep."

    if not args.rollout.vllm_generate_batch_size:
        args.rollout.vllm_generate_batch_size = args.rollout.batch_size

    # m3: vllm_generate_batch_size > batch_size requires async_enable.
    if args.rollout.vllm_generate_batch_size > args.rollout.batch_size:
        assert args.train.async_enable, (
            "--rollout.vllm_generate_batch_size > --rollout.batch_size requires "
            "--train.async_enable."
        )

    # Multi-agent generates 3x the samples per step (thinker + executor + verifier).
    # Ensure the train batch size can absorb the combined rollout.
    combined_rollout = args.rollout.batch_size * args.rollout.n_samples_per_prompt * 3
    assert combined_rollout >= args.train.batch_size, (
        f"Combined rollout size ({combined_rollout}) must be >= train batch size "
        f"({args.train.batch_size}). Reduce --train.batch_size or increase "
        f"--rollout.batch_size * --rollout.n_samples_per_prompt."
    )

    # Warn if max_len is too small for multi-stage prompts. The Executor prompt
    # includes the Thinker's response, and the Verifier prompt includes both.
    # A rough lower bound: prompt_overhead + thinker_max + executor_max.
    min_required_len = (
        args.multi_agent.thinker_max_response
        + args.multi_agent.executor_max_response
        + 512  # system prompt + problem text overhead estimate
    )
    if args.data.max_len < min_required_len:
        print(
            f"[Warning] --data.max_len ({args.data.max_len}) is smaller than the "
            f"estimated minimum for multi-agent ({min_required_len}). Executor/Verifier "
            f"sequences may be truncated, causing unparseable answers. Consider "
            f"increasing --data.max_len to at least {min_required_len}."
        )

    if args.algo.kl.use_loss:
        if args.algo.kl.estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.algo.kl.estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.algo.kl.estimator not in ["k1"]:
            print(f"Recommend setting {args.algo.kl.estimator} to 'k1' when not using KL as a loss.")

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    return args


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    args = _validate_args(args)
    train(args)
