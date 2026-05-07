#!/usr/bin/env python3
"""Evaluate Qwen3-1.7B (+ LoRA) on MATH-500 using vLLM + math_verify."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import parse, verify
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    boxed2 = re.search(r'\\boxed\s*\{?\s*([^}]+)\s*\}?', text)
    if boxed2:
        return boxed2.group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


def format_prompt(problem: str, system: str, tokenizer) -> str:
    """Format prompt using chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def evaluate_vllm(llm, tokenizer, dataset, system: str, adapter_path: str = None, max_tokens: int = 4096):
    """Evaluate using vLLM."""
    correct = 0
    total = len(dataset)
    results = []

    # Prepare prompts
    prompts = []
    for problem in dataset['problem']:
        prompts.append(format_prompt(problem, system, tokenizer))

    # Sampling params: greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=42,
    )

    # Generate
    lora_request = None
    if adapter_path:
        lora_request = LoRARequest("adapter", 1, adapter_path)

    print(f"Generating {total} responses...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )

    # Evaluate
    for i, output in enumerate(tqdm(outputs, desc="Verifying")):
        response_text = output.outputs[0].text
        pred_answer = extract_answer(response_text)
        gt_answer = dataset[i]['answer']

        try:
            pred_parsed = parse(pred_answer)
            gt_parsed = parse(gt_answer)
            is_correct = verify(pred_parsed, gt_parsed)
        except Exception:
            is_correct = pred_answer.strip() == gt_answer.strip()

        if is_correct:
            correct += 1

        results.append({
            'id': dataset[i]['unique_id'],
            'subject': dataset[i]['subject'],
            'level': dataset[i]['level'],
            'gt_answer': gt_answer,
            'pred_answer': pred_answer,
            'response': response_text[:800],
            'correct': is_correct,
        })

    accuracy = correct / total
    return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=str, default=None, help='Path to LoRA adapter')
    parser.add_argument('--output', type=str, default='eval_result.json', help='Output JSON path')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--limit', type=int, default=None, help='Limit examples for quick test')
    parser.add_argument('--gpu_util', type=float, default=0.85)
    args = parser.parse_args()

    BASE_MODEL = 'Qwen/Qwen3-1.7B'
    SYSTEM = 'You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.'

    # Load dataset
    print("Loading MATH-500 dataset...")
    dataset = load_dataset('HuggingFaceH4/MATH-500', split='test')
    if args.limit:
        dataset = dataset.select(range(args.limit))
    print(f"Dataset size: {len(dataset)}")

    # Load model with vLLM
    print(f"Loading model with vLLM (gpu_util={args.gpu_util})...")
    llm = LLM(
        model=BASE_MODEL,
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_util,
        max_model_len=8192,
        trust_remote_code=True,
        enable_lora=args.adapter is not None,
        max_lora_rank=64,
    )
    tokenizer = llm.get_tokenizer()

    # Evaluate
    print(f"Evaluating (max_tokens={args.max_tokens})...")
    accuracy, results = evaluate_vllm(
        llm, tokenizer, dataset, SYSTEM,
        adapter_path=args.adapter, max_tokens=args.max_tokens
    )

    # Report
    print(f"\n{'='*50}")
    print(f"Adapter: {args.adapter if args.adapter else 'Baseline (no adapter)'}")
    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(dataset))}/{len(dataset)})")
    print(f"{'='*50}")

    # Subject breakdown
    subjects = {}
    for r in results:
        sub = r['subject']
        if sub not in subjects:
            subjects[sub] = {'correct': 0, 'total': 0}
        subjects[sub]['total'] += 1
        if r['correct']:
            subjects[sub]['correct'] += 1

    print("\nPer-subject accuracy:")
    for sub, stats in sorted(subjects.items()):
        acc = stats['correct'] / stats['total']
        print(f"  {sub:20s}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    # Save
    with open(args.output, 'w') as f:
        json.dump({
            'adapter': args.adapter,
            'accuracy': accuracy,
            'correct': int(accuracy * len(dataset)),
            'total': len(dataset),
            'subjects': {k: {'correct': v['correct'], 'total': v['total'], 'accuracy': v['correct']/v['total']} for k, v in subjects.items()},
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
