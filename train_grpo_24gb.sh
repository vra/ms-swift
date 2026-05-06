#!/bin/bash
# GRPO Math Training Script for 24GB VRAM
# Optimized for larger batch size, num_generations, and max_completion_length

source ~/miniforge3/etc/profile.d/conda.sh
conda activate grpo

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

MAX_PROMPT_LENGTH=2048
MAX_COMPLETION_LENGTH=4096
VLLM_MAX_MODEL_LEN=$(($MAX_COMPLETION_LENGTH + $MAX_PROMPT_LENGTH))

NUM_GENERATIONS=16
GENERATION_BATCH_SIZE=16

CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29600 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-1.7B \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --use_hf true \
    --reward_funcs accuracy \
    --enable_thinking false \
    --epsilon 0.2 \
    --beta 0.04 \
    --num_generations $NUM_GENERATIONS \
    --generation_batch_size $GENERATION_BATCH_SIZE \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.7 \
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
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --save_steps 100 \
    --save_total_limit 10 \
    --max_steps 1000 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --log_completions true \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --disable_tqdm true \
    --output_dir output/grpo_math_qwen3_1.7b
