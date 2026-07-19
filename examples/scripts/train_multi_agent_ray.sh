#!/bin/bash
# Multi-agent (Thinker + Executor + Verifier) GRPO training with OpenRLHF.
#
# One model plays three roles via different system prompts. All three roles
# are trained simultaneously every step using a combined batch.
#
# Pipeline per training step:
#   Stage 1 (Thinker):   problem       -> thought (reasoning, no final answer)
#   Stage 2 (Executor):  problem+thought -> answer (\\boxed{...})
#   Stage 3 (Verifier):  problem+thought+answer -> verdict (CORRECT/INCORRECT)
#
# Rewards:
#   Thinker  : gt_correct + weight * verifier_agreement
#   Executor : gt_correct + weight * verifier_agreement
#   Verifier : verifier_agreement (did it agree with ground truth?)
#
# Usage:
#   ray job submit --address="http://127.0.0.1:8265" \
#     -- bash examples/scripts/train_multi_agent_ray.sh
#
# Or run directly:
#   bash examples/scripts/train_multi_agent_ray.sh

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

set -x

MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
DATASET_PATH="zhuzilin/dapo-math-17k"
SAVE_PATH="${WORK_DIR}/exp/Qwen3-4B-MultiAgent"

CKPT_ARGS=(
   --actor.model_name_or_path ${MODEL_PATH}
   --ckpt.load_enable
   --ckpt.output_dir ${SAVE_PATH}
   --ckpt.path "${SAVE_PATH}/ckpt"
   --ckpt.save_hf
   --ckpt.max_num 3
   --ckpt.save_steps 10
)

ROLLOUT_ARGS=(
   --data.prompt_dataset ${DATASET_PATH}
   --data.input_key prompt
   --data.label_key label
   --data.max_len 8192
   --ds.packing_samples

   --rollout.batch_size 32
   --rollout.n_samples_per_prompt 8
   --train.batch_size 96
   --train.micro_batch_size 1
   --rollout.micro_batch_size 8
   --data.max_samples 128000
   --train.max_epochs 1
   --train.num_episodes 1
)

ENGINE_ARGS=(
   --ref.num_nodes 1
   --ref.num_gpus_per_node 4
   --actor.num_nodes 1
   --actor.num_gpus_per_node 4
   --vllm.num_engines 2
   --vllm.tensor_parallel_size 2
   --vllm.gpu_memory_utilization 0.7
   --train.colocate_all
   --ds.enable_sleep
   --vllm.sync_backend nccl
   --vllm.enforce_eager

   --ds.zero_stage 3
   --actor.gradient_checkpointing_enable
   --ds.ring_attn_size 2
   --ds.ring_attn_head_stride 2
   --ds.param_dtype bf16
)

OPTIMIZER_ARGS=(
   --algo.advantage.estimator group_norm
   --actor.adam.lr 5e-7
   --actor.entropy_coef 0.0
   --algo.kl.init_coef 1e-5
   --algo.kl.use_loss
   --algo.kl.estimator k2
)

# ── Multi-agent specific ──
MULTI_AGENT_ARGS=(
   --multi_agent.thinker_max_response 4096
   --multi_agent.executor_max_response 6144
   --multi_agent.verifier_max_response 1024
   --multi_agent.verifier_reward_weight 0.3
)

LOG_ARGS=(
   --logger.tensorboard_dir ${SAVE_PATH}/runs
   --logger.logging_steps 1
   --eval.steps -1
)

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_multi_agent_ray \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${ENGINE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${MULTI_AGENT_ARGS[@]} \
   ${LOG_ARGS[@]}
