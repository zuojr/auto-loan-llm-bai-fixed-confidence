#!/usr/bin/env python
"""AutoDL-friendly Qwen LoRA/QLoRA fine-tuning for acceptance-probability generation.

Example on AutoDL:
  python scripts/07_train_qwen_lora_autodl.py \
      --train-jsonl qwen_finetune/train_full.jsonl \
      --eval-jsonl qwen_finetune/val_full.jsonl \
      --model-name Qwen/Qwen2.5-7B-Instruct \
      --out-dir qwen_finetune/qwen_lora_auto_loan \
      --epochs 1 --batch-size 1 --grad-accum 16
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-jsonl', default='qwen_finetune/train_full.jsonl')
    ap.add_argument('--eval-jsonl', default='qwen_finetune/val_full.jsonl')
    ap.add_argument('--model-name', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--out-dir', default='qwen_finetune/qwen_lora_auto_loan')
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--no-4bit', action='store_true')
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data_files = {'train': args.train_jsonl}
    if args.eval_jsonl and Path(args.eval_jsonl).exists():
        data_files['validation'] = args.eval_jsonl
    ds = load_dataset('json', data_files=data_files)

    def tokenize(ex):
        messages = ex['messages']
        # Mask prompt tokens and train only assistant completion.
        prompt_messages = messages[:-1]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full = tokenizer(full_text, truncation=True, max_length=args.max_length, padding=False)
        prompt = tokenizer(prompt_text, truncation=True, max_length=args.max_length, padding=False)
        labels = full['input_ids'].copy()
        prompt_len = min(len(prompt['input_ids']), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        full['labels'] = labels
        return full

    tok = ds.map(tokenize, remove_columns=ds['train'].column_names)

    def collate(features):
        max_len = max(len(f['input_ids']) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [tokenizer.pad_token_id]*pad)
            attention_mask.append(f['attention_mask'] + [0]*pad)
            labels.append(f['labels'] + [-100]*pad)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    train_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        eval_strategy='steps' if 'validation' in tok else 'no',
        eval_steps=500,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim='paged_adamw_8bit' if not args.no_4bit else 'adamw_torch',
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tok['train'],
        eval_dataset=tok.get('validation'),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print('Saved adapter/tokenizer to', args.out_dir)

if __name__ == '__main__':
    main()
