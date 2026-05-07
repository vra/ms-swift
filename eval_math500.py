#!/usr/bin/env python3
"""Evaluate Qwen3-1.7B (+ LoRA) on MATH-500 using math_verify."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import parse, verify
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text: str) -> str:
    """Extract answer from model output, preferring \boxed{} content."""
    # Try \boxed{...}
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    # Try boxed without braces (sometimes malformed)
    boxed2 = re.search(r'\\boxed\s*\{?\s*([^}]+)\s*\}?', text)
    if boxed2:
        return boxed2.group(1).strip()
    # Fallback: last line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def load_model_and_tokenizer(base_model: str, adapter_path: str = None):
    """Load model and optional LoRA adapter."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("Adapter merged.")

    model.eval()
    return model, tokenizer


def build_prompt(problem: str, system: str) -> str:
    """Build chat prompt for Qwen3."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": problem},
    ]
    return messages


def evaluate(model, tokenizer, dataset, system: str, batch_size: int = 8, max_new_tokens: int = 8192):
    """Evaluate model on MATH-500 dataset."""
    correct = 0
    total = len(dataset)
    results = []

    for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch = dataset[i:i + batch_size]
        problems = batch['problem']
        gt_answers = batch['answer']
        ids = batch['unique_id']

        # Build prompts
        prompts = []
        for problem in problems:
            messages = build_prompt(problem, system)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

        # Tokenize
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                top_k=1,
                top_p=1.0,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        for j, output in enumerate(outputs):
            # Remove prompt tokens
            prompt_len = inputs['input_ids'][j].shape[0]
            # Find actual prompt length (non-padding)
            prompt_mask = inputs['attention_mask'][j].bool()
            actual_prompt_len = prompt_mask.sum().item()

            response_ids = output[actual_prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            pred_answer = extract_answer(response_text)
            gt_answer = gt_answers[j]

            # Verify with math_verify
            try:
                pred_parsed = parse(pred_answer)
                gt_parsed = parse(gt_answer)
                is_correct = verify(pred_parsed, gt_parsed)
            except Exception:
                is_correct = pred_answer.strip() == gt_answer.strip()

            if is_correct:
                correct += 1

            results.append({
                'id': ids[j],
                'problem': problems[j][:200],
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'response': response_text[:500],
                'correct': is_correct,
            })

    accuracy = correct / total
    return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=str, default=None, help='Path to LoRA adapter (None for baseline)')
    parser.add_argument('--output', type=str, default='eval_result.json', help='Output JSON path')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples for quick test')
    args = parser.parse_args()

    BASE_MODEL = 'Qwen/Qwen3-1.7B'
    SYSTEM = 'You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.'

    # Load dataset
    print("Loading MATH-500 dataset...")
    dataset = load_dataset('HuggingFaceH4/MATH-500', split='test')
    if args.limit:
        dataset = dataset.select(range(args.limit))
    print(f"Dataset size: {len(dataset)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, args.adapter)

    # Evaluate
    print(f"Starting evaluation (batch_size={args.batch_size})...")
    accuracy, results = evaluate(
        model, tokenizer, dataset, SYSTEM,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens
    )

    # Report
    print(f"\n{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(dataset))}/{len(dataset)})")
    print(f"{'='*50}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'adapter': args.adapter,
            'accuracy': accuracy,
            'correct': int(accuracy * len(dataset)),
            'total': len(dataset),
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
