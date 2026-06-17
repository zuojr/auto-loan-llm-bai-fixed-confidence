#!/usr/bin/env python
"""Score paired auto-loan rows with a Qwen model through Alibaba DashScope OpenAI-compatible API.

Environment variables:
  DASHSCOPE_API_KEY  required
  QWEN_API_MODEL     optional, default qwen-plus
  DASHSCOPE_BASE_URL optional; default Singapore compatible endpoint.
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from pathlib import Path
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.auto_loan_utils import parse_probability, make_api_user_batch, make_calibration_examples_text


def make_user_batch(rows: pd.DataFrame, global_stats: dict, examples_text: str) -> str:
    return make_api_user_batch(rows, global_stats, examples_text)

def parse_json_array(text: str):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r'(\[.*\])', text, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(1))
    if isinstance(obj, dict) and 'predictions' in obj:
        obj = obj['predictions']
    return obj


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
    ap.add_argument('--scoring', default=str(ROOT/'api_scoring/scoring_pilot_1000.csv'))
    ap.add_argument('--out', default=str(ROOT/'predictions/qwen_api_predictions.csv'))
    ap.add_argument('--model', default=os.getenv('QWEN_API_MODEL', 'qwen-plus'))
    ap.add_argument('--base-url', default=os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'))
    ap.add_argument('--rows-per-call', type=int, default=10)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--sleep', type=float, default=0.2)
    ap.add_argument('--n-examples', type=int, default=16)
    args = ap.parse_args()
    if not os.getenv('DASHSCOPE_API_KEY'):
        raise RuntimeError('Set DASHSCOPE_API_KEY before running this script.')
    client = OpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=args.base_url)
    system_prompt = (ROOT/'api_scoring/llm_scoring_system_prompt.md').read_text()
    global_stats = json.loads((ROOT/'api_scoring/calibration_global_stats.json').read_text())
    ex = pd.read_csv(ROOT/'api_scoring/calibration_examples_balanced.csv')
    examples_text = make_calibration_examples_text(ex, max_rows=args.n_examples)
    df = pd.read_csv(args.scoring)
    if args.limit:
        df = df.head(args.limit)
    df['record_id'] = df['record_id'].astype(str)
    done = read_done(Path(args.out))
    df = df[~df['record_id'].isin(done)].reset_index(drop=True)
    for start in range(0, len(df), args.rows_per_call):
        batch = df.iloc[start:start+args.rows_per_call]
        user = make_user_batch(batch, global_stats, examples_text)
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {'role':'system','content':system_prompt},
                {'role':'user','content':user},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content
        try:
            arr = parse_json_array(text)
            byid = {str(x.get('record_id')): x for x in arr}
            out_rows = []
            for _, r in batch.iterrows():
                obj = byid.get(str(r['record_id']), {})
                p = parse_probability(obj.get('pred_prob', obj), default=None)
                if p is None or p != p:
                    p = 0.18
                out_rows.append({'record_id': r['record_id'], 'pred_prob': p})
        except Exception:
            fail_dir = Path(args.out).parent/'raw_failures'
            fail_dir.mkdir(exist_ok=True)
            (fail_dir/f'qwen_failure_{start}.txt').write_text(text)
            out_rows = [{'record_id': r['record_id'], 'pred_prob': 0.18} for _, r in batch.iterrows()]
        append_rows(Path(args.out), out_rows)
        print(f'wrote {min(start+args.rows_per_call, len(df))}/{len(df)} to {args.out}')
        time.sleep(args.sleep)

if __name__ == '__main__':
    main()
