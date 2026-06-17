#!/usr/bin/env python
"""Generate pred_prob for paired scoring rows with a local open-weight Qwen model, no fine-tuning.

This is the same-base ICL comparator for Qwen LoRA. Use the same base model
(e.g. Qwen/Qwen2.5-7B-Instruct) for this script and for 07/08 LoRA training.

Example on AutoDL:
  python scripts/09_predict_qwen_local_icl_autodl.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --scoring api_scoring/scoring_full_paired.csv \
    --out predictions/qwen_local_icl_predictions.csv \
    --rows-per-call 6 --n-examples 12
"""
from __future__ import annotations

import argparse, csv, json, re, sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.auto_loan_utils import make_api_user_batch, make_calibration_examples_text, parse_probability

SYSTEM_PROMPT = (ROOT / 'api_scoring' / 'llm_scoring_system_prompt.md').read_text()


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)['record_id'].astype(str))
    except Exception:
        return set()


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['record_id', 'pred_prob'])
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def clip_prob(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    if not (v == v):
        v = default
    return max(0.01, min(0.99, v))


def extract_json_array(text: str) -> Any | None:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    # Remove common markdown fences if present.
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    candidates = [text]
    # Find first bracketed array.
    m = re.search(r'\[[\s\S]*\]', text)
    if m:
        candidates.insert(0, m.group(0))
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None


def parse_batch_output(text: str, input_ids: list[str], default_prob: float) -> list[dict[str, Any]]:
    """Parse model output into rows, filling missing ids with a conservative default."""
    wanted = [str(x) for x in input_ids]
    found: dict[str, float] = {}
    obj = extract_json_array(text)
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            rid = str(item.get('record_id', '')).strip()
            if not rid:
                continue
            # Accept pred_prob, probability, p, prob as fallbacks.
            val = item.get('pred_prob', item.get('probability', item.get('prob', item.get('p'))))
            found[rid] = clip_prob(val, default_prob)
    # Regex fallback: useful if the model emits CSV or semi-structured text.
    if len(found) < len(wanted):
        for rid in wanted:
            if rid in found:
                continue
            pat = re.escape(rid) + r'[^0-9]{0,80}([01](?:\.\d+)?)'
            m = re.search(pat, text)
            if m:
                found[rid] = clip_prob(m.group(1), default_prob)
    # Last fallback: parse probabilities in order, but only if enough numbers appear.
    if len(found) < len(wanted):
        nums = [float(x) for x in re.findall(r'(?<![A-Za-z0-9_])(?:0?\.\d+|1\.0+|1|0)(?![A-Za-z0-9_])', text)]
        nums = [x for x in nums if 0 <= x <= 1]
        if len(nums) >= len(wanted):
            for rid, val in zip(wanted, nums[-len(wanted):]):
                found.setdefault(rid, clip_prob(val, default_prob))
    return [{'record_id': rid, 'pred_prob': found.get(rid, default_prob)} for rid in wanted]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--scoring', default='api_scoring/scoring_full_paired.csv')
    ap.add_argument('--calibration', default='api_scoring/calibration_examples_balanced.csv')
    ap.add_argument('--global-stats', default='api_scoring/calibration_global_stats.json')
    ap.add_argument('--out', default='predictions/qwen_local_icl_predictions.csv')
    ap.add_argument('--rows-per-call', type=int, default=6)
    ap.add_argument('--n-examples', type=int, default=12)
    ap.add_argument('--max-new-tokens', type=int, default=1536)
    ap.add_argument('--max-input-tokens', type=int, default=8192)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-4bit', action='store_true')
    args = ap.parse_args()

    scoring = Path(args.scoring)
    if not scoring.is_absolute(): scoring = ROOT / scoring
    calibration = Path(args.calibration)
    if not calibration.is_absolute(): calibration = ROOT / calibration
    global_stats_path = Path(args.global_stats)
    if not global_stats_path.is_absolute(): global_stats_path = ROOT / global_stats_path
    out = Path(args.out)
    if not out.is_absolute(): out = ROOT / out

    df = pd.read_csv(scoring)
    if args.limit:
        df = df.head(args.limit)
    df['record_id'] = df['record_id'].astype(str)
    done = read_done(out)
    df = df[~df['record_id'].isin(done)].reset_index(drop=True)

    global_stats = json.loads(global_stats_path.read_text())
    default_prob = clip_prob(global_stats.get('train_apply_rate', global_stats.get('apply_rate', 0.18)), 0.18)
    examples = pd.read_csv(calibration)
    examples_text = make_calibration_examples_text(examples, max_rows=args.n_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    total = len(df)
    for start in range(0, total, args.rows_per_call):
        batch = df.iloc[start:start + args.rows_per_call]
        user_prompt = make_api_user_batch(batch, global_stats, examples_text)
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True, max_length=args.max_input_tokens).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        ids = batch['record_id'].astype(str).tolist()
        rows = parse_batch_output(text, ids, default_prob=default_prob)
        append_rows(out, rows)
        print(f'wrote {min(start + args.rows_per_call, total)}/{total} to {out}')


if __name__ == '__main__':
    main()
