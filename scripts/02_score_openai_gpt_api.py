#!/usr/bin/env python
"""Score paired auto-loan rows with an OpenAI GPT model, no fine-tuning.

Environment variables:
  OPENAI_API_KEY   required
  OPENAI_MODEL     optional, e.g. gpt-5.2-mini, gpt-4.1-mini, etc.

The script writes a CSV with columns record_id,pred_prob and supports resume.
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
    # remove fenced code if any
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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def progress_line(done: int, total: int, out_path: str | Path, started_at: float, last_batch_seconds: float) -> str:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    pct = 100.0 * done / total if total else 100.0
    avg_per_row = elapsed / max(done, 1)
    remaining_rows = max(total - done, 0)
    eta = avg_per_row * remaining_rows
    avg_row = elapsed / max((done + 1e-9), 1.0)
    return (
        f"wrote {done}/{total} ({pct:.1f}%) to {out_path} | "
        f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)} | "
        f"last_call {last_batch_seconds:.1f}s | avg_row {avg_row:.1f}s/row"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scoring', default=str(ROOT/'api_scoring/scoring_pilot_1000.csv'))
    ap.add_argument('--out', default=str(ROOT/'predictions/gpt_icl_predictions.csv'))
    ap.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'gpt-5.2-mini'))
    ap.add_argument('--rows-per-call', type=int, default=10)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--sleep', type=float, default=0.2)
    ap.add_argument('--n-examples', type=int, default=16)
    args = ap.parse_args()
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY before running this script.')

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
    client = OpenAI()
    started_at = time.monotonic()
    total = len(df)
    if total == 0:
        print(f'No remaining rows to score for {args.out}')
        return
    print(f'Scoring {total} remaining rows with model {args.model}, rows_per_call={args.rows_per_call}')
    for start in range(0, len(df), args.rows_per_call):
        batch_started = time.monotonic()
        batch = df.iloc[start:start+args.rows_per_call]
        user = make_user_batch(batch, global_stats, examples_text)
        resp = client.responses.create(
            model=args.model,
            input=[
                {'role':'system','content':system_prompt},
                {'role':'user','content':user},
            ],
        )
        text = resp.output_text
        try:
            arr = parse_json_array(text)
            out_rows = []
            byid = {str(x.get('record_id')): x for x in arr}
            for _, r in batch.iterrows():
                obj = byid.get(str(r['record_id']), {})
                p = parse_probability(obj.get('pred_prob', obj), default=None)
                if p is None or p != p:
                    p = 0.18
                out_rows.append({'record_id': r['record_id'], 'pred_prob': p})
        except Exception as e:
            # Save raw failure for debugging and continue with conservative base-rate fallback.
            fail_dir = Path(args.out).parent/'raw_failures'
            fail_dir.mkdir(exist_ok=True)
            (fail_dir/f'gpt_failure_{start}.txt').write_text(text)
            out_rows = [{'record_id': r['record_id'], 'pred_prob': 0.18} for _, r in batch.iterrows()]
        append_rows(Path(args.out), out_rows)
        done_now = min(start + args.rows_per_call, total)
        last_batch_seconds = time.monotonic() - batch_started
        print(progress_line(done_now, total, args.out, started_at, last_batch_seconds), flush=True)
        time.sleep(args.sleep)

if __name__ == '__main__':
    main()
