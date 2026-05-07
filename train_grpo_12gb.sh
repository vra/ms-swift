#!/bin/bash
# GRPO Math Training Script for 12GB VRAM (RTX 4070 Ti)

source /home/ws/miniforge3/etc/profile.d/conda.sh
conda activate grpo

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

MAX_PROMPT_LENGTH=2048
MAX_COMPLETION_LENGTH=2048
VLLM_MAX_MODEL_LEN=$(($MAX_COMPLETION_LENGTH + $MAX_PROMPT_LENGTH))

NUM_GENERATIONS=8
GENERATION_BATCH_SIZE=8

CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29600 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-1.7B \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --use_hf true \
    --external_plugins custom_reward_plugin.py \
    --reward_funcs accuracy conditional_math \
    --reward_weights 1.0 1.0 \
    --enable_thinking false \
    --epsilon 0.2 \
    --beta 0.04 \
    --num_generations $NUM_GENERATIONS \
    --generation_batch_size $GENERATION_BATCH_SIZE \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len $VLLM_MAX_MODEL_LEN \
    --sleep_level 1 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --max_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --save_steps 50 \
    --save_total_limit 10 \
    --max_steps 400 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --log_completions true \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --disable_tqdm true \
    # --resume_from_checkpoint output/grpo_math_qwen3_1.7b_v12/v1-20260507-110656/checkpoint-250 \
    --output_dir output/grpo_math_qwen3_1.7b_v12
