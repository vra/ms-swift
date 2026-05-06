#!/usr/bin/env python3
"""Evaluate Qwen3-1.7B (+ LoRA) on MATH-500 using vLLM."""
import argparse, json, re, sys
from pathlib import Path
from datasets import load_dataset
from math_verify import parse, verify
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def extract_answer(text):
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    boxed2 = re.search(r'\\boxed\s*\{?\s*([^}]+)\s*\}?', text)
    if boxed2:
        return boxed2.group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()

def format_prompt(problem, system, tokenizer):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=str, default=None)
    parser.add_argument('--output', type=str, default='eval_result.json')
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--max_model_len', type=int, default=12288)
    parser.add_argument('--gpu_util', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    BASE_MODEL = 'Qwen/Qwen3-1.7B'
    SYSTEM = 'You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}.'

    print("Loading dataset...", flush=True)
    dataset = load_dataset('HuggingFaceH4/MATH-500', split='test')
    if args.limit:
        dataset = dataset.select(range(args.limit))
    total = len(dataset)
    print(f"Dataset size: {total}", flush=True)

    print(f"Loading vLLM (gpu_util={args.gpu_util}, max_model_len={args.max_model_len})...", flush=True)
    llm = LLM(
        model=BASE_MODEL,
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=args.adapter is not None,
        max_lora_rank=64,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0,
        max_tokens=args.max_tokens, seed=42,
    )

    lora_request = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    name = "Baseline" if args.adapter is None else Path(args.adapter).name
    print(f"\nEvaluating: {name}", flush=True)

    correct = 0
    results = []
    num_batches = (total + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, total)
        batch = dataset[start:end]
        prompts = [format_prompt(p, SYSTEM, tokenizer) for p in batch['problem']]

        print(f"  Batch {batch_idx+1}/{num_batches} ({start}-{end}) generating...", flush=True)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

        for i, output in enumerate(outputs):
            idx = start + i
            response_text = output.outputs[0].text
            pred_answer = extract_answer(response_text)
            gt_answer = batch['answer'][i]

            try:
                is_correct = verify(parse(pred_answer), parse(gt_answer))
            except Exception:
                is_correct = pred_answer.strip() == gt_answer.strip()

            if is_correct:
                correct += 1

            results.append({
                'id': batch['unique_id'][i],
                'subject': batch['subject'][i],
                'level': batch['level'][i],
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'correct': is_correct,
            })

        print(f"  Batch {batch_idx+1}/{num_batches} done. Running acc: {correct}/{end} = {correct/end:.4f}", flush=True)

    accuracy = correct / total
    print(f"\n{'='*50}", flush=True)
    print(f"{name}: Accuracy = {accuracy:.4f} ({correct}/{total})", flush=True)
    print(f"{'='*50}", flush=True)

    subjects = {}
    for r in results:
        sub = r['subject']
        if sub not in subjects:
            subjects[sub] = {'correct': 0, 'total': 0}
        subjects[sub]['total'] += 1
        if r['correct']:
            subjects[sub]['correct'] += 1

    print("\nPer-subject:", flush=True)
    for sub, stats in sorted(subjects.items()):
        acc = stats['correct'] / stats['total']
        print(f"  {sub:25s}: {acc:.4f} ({stats['correct']}/{stats['total']})", flush=True)

    with open(args.output, 'w') as f:
        json.dump({
            'name': name, 'adapter': args.adapter,
            'accuracy': accuracy, 'correct': correct, 'total': total,
            'subjects': {k: {'correct': v['correct'], 'total': v['total'], 'accuracy': v['correct']/v['total']} for k, v in subjects.items()},
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}", flush=True)

if __name__ == '__main__':
    main()
