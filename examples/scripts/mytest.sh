source /data/shenboyang/envs/openrlhf/bin/activate

export CUDA_VISIBLE_DEVICES=1,3
python3 -m openrlhf.cli.train_ppo_ray \
   --actor.model_name_or_path /memory/shenboyang/myCache/huggingface/hub/Qwen--Qwen3.5-0.8B \
   --reward.remote_url examples/python/math_reward_func.py \
   --data.prompt_dataset /memory/shenboyang/myCache/huggingface/hub/zhuzilin--dapo-math-17k \
   --data.input_key prompt \
   --data.label_key label \
   --data.apply_chat_template \
   --ds.packing_samples \
   \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 2 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 2 \
   --vllm.num_engines 2 \
   --vllm.tensor_parallel_size 1 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.2 \
   --vllm.enable_sleep \
   --ds.enable_sleep \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   \
   --algo.advantage.estimator reinforce_baseline \
   --algo.kl.use_loss \
   --algo.kl.estimator k2 \
   --algo.kl.init_coef 1e-5 \
   --actor.entropy_coef 0.0 \
   --algo.advantage.is_correction_enable \
   --algo.advantage.is_correction_type icepop \
   \
   --rollout.batch_size 2 \
   --rollout.n_samples_per_prompt 4 \
   --train.batch_size 2 \
   --algo.dynamic_filtering_enable \
   --algo.dynamic_filtering_range 0.0 1.0 \
   --train.dynamic_batch_enable \
   --data.max_len 4096 \
   --rollout.max_new_tokens 1024 \
   \
   --ds.zero_stage 2 \
   --ds.param_dtype bf16 \
   --actor.gradient_checkpointing_enable \
   --actor.adam.lr 5e-7 \
   --ckpt.output_dir shenboyang/outputs/train/multi_agent/Qwen3.5-0.8B
