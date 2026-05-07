#!/bin/bash
# Evaluation script for trained model

ADAPTER_PATH=$1

if [ -z "$ADAPTER_PATH" ]; then
    echo "Usage: ./eval_model.sh <adapter_path>"
    echo "Example: ./eval_model.sh output/grpo_math_qwen3_1.7b/checkpoint-100"
    exit 1
fi

swift eval \
    --model Qwen/Qwen3-1.7B \
    --adapters $ADAPTER_PATH \
    --merge_lora true \
    --enable_thinking false \
    --eval_dataset math_500 \
    --eval_backend Native \
    --infer_backend vllm \
    --port 8000 \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
