#!/usr/bin/env python
"""Score rows with a yes/no LoRA adapter by reading next-token logits.

Outputs raw probabilities; calibration is handled by
scripts/12_calibrate_qwen_lora_cls.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.qwen_lora_cls_utils import CLS_SYSTEM_PROMPT, make_binary_acceptance_prompt, yes_no_probability_from_logits


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_done(path: Path, id_col: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)[id_col].astype(str))
    except Exception:
        return set()


def _append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def _single_token_ids(tokenizer, candidates: list[str]) -> list[int]:
    ids: set[int] = set()
    for text in candidates:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return sorted(ids)


def _label_from_assistant(content: str) -> int:
    s = str(content).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return 1
    if s in {"no", "n", "false", "0"}:
        return 0
    obj = json.loads(content)
    return 1 if float(obj["pred_prob"]) > 0.5 else 0


def _load_scoring_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    df = pd.read_csv(path)
    df["record_id"] = df["record_id"].astype(str)
    rows = []
    for _, r in df.iterrows():
        messages = [
            {"role": "system", "content": CLS_SYSTEM_PROMPT},
            {"role": "user", "content": make_binary_acceptance_prompt(r)},
        ]
        rows.append({"record_id": str(r["record_id"]), "messages": messages})
    return "record_id", rows


def _load_val_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = obj["messages"]
            y = _label_from_assistant(messages[-1]["content"])
            rows.append({"val_id": str(i), "y": y, "messages": messages[:-1]})
    return "val_id", rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict raw yes/no logit probabilities with a Qwen LoRA classifier.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-dir", default="qwen_finetune/qwen_lora_cls_auto_loan")
    ap.add_argument("--scoring", default=None, help="CSV with scoring rows. Mutually exclusive with --val-jsonl.")
    ap.add_argument("--val-jsonl", default=None, help="Classification JSONL validation set.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if bool(args.scoring) == bool(args.val_jsonl):
        raise ValueError("Pass exactly one of --scoring or --val-jsonl.")

    out = _resolve(args.out)
    if args.scoring:
        id_col, rows = _load_scoring_rows(_resolve(args.scoring))
        fieldnames = ["record_id", "raw_pred_prob"]
    else:
        id_col, rows = _load_val_rows(_resolve(args.val_jsonl))
        fieldnames = ["val_id", "y", "raw_pred_prob"]
    if args.limit:
        rows = rows[: args.limit]

    done = _read_done(out, id_col)
    rows = [r for r in rows if str(r[id_col]) not in done]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_ids = _single_token_ids(tokenizer, ["yes", "Yes", " YES", " yes", "Yes.", "yes."])
    no_ids = _single_token_ids(tokenizer, ["no", "No", " NO", " no", "No.", "no."])
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Could not find single-token yes/no ids. yes={yes_ids}, no={no_ids}")
    print(f"yes_token_ids={yes_ids}")
    print(f"no_token_ids={no_ids}")

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    total = len(rows)
    for start in range(0, total, args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].detach().float().cpu().numpy()
        out_rows = []
        for row, logit in zip(batch, logits):
            raw_prob = yes_no_probability_from_logits(logit, yes_ids, no_ids)
            if args.scoring:
                out_rows.append({"record_id": row["record_id"], "raw_pred_prob": raw_prob})
            else:
                out_rows.append({"val_id": row["val_id"], "y": row["y"], "raw_pred_prob": raw_prob})
        _append_rows(out, out_rows, fieldnames)
        print(f"wrote {min(start + args.batch_size, total)}/{total} to {out}")


if __name__ == "__main__":
    main()
