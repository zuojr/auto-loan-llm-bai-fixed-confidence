#!/usr/bin/env python
"""Generate pred_prob for paired scoring rows using a fine-tuned Qwen LoRA adapter."""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.auto_loan_utils import make_prompt_for_acceptance, parse_probability


def read_done(path: Path):
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)['record_id'].astype(str))
    except Exception:
        return set()


def append_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['record_id','pred_prob'])
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--adapter-dir', default='qwen_finetune/qwen_lora_auto_loan')
    ap.add_argument('--scoring', default='api_scoring/scoring_full_paired.csv')
    ap.add_argument('--out', default='predictions/qwen_ft_predictions.csv')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-4bit', action='store_true')
    args = ap.parse_args()
    scoring = Path(args.scoring)
    if not scoring.is_absolute(): scoring = ROOT/scoring
    out = Path(args.out)
    if not out.is_absolute(): out = ROOT/out

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, bnb_4bit_quant_type='nf4')
    base = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant, device_map='auto', trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    df = pd.read_csv(scoring)
    if args.limit: df = df.head(args.limit)
    df['record_id'] = df['record_id'].astype(str)
    done = read_done(out)
    df = df[~df['record_id'].isin(done)].reset_index(drop=True)
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start+args.batch_size]
        prompts = []
        for _, r in batch.iterrows():
            messages = [
                {'role':'system','content':'You are a calibrated auto-loan offer acceptance predictor. Return only valid JSON.'},
                {'role':'user','content': make_prompt_for_acceptance(r, include_policy=True)},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        outputs = tokenizer.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        rows=[]
        for rid, text in zip(batch['record_id'], outputs):
            p = parse_probability(text, default=0.18)
            rows.append({'record_id': rid, 'pred_prob': p})
        append_rows(out, rows)
        print(f'wrote {min(start+args.batch_size,len(df))}/{len(df)} to {out}')

if __name__ == '__main__':
    main()
