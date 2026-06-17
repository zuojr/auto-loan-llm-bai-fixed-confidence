# Qwen Fine-tuning Directory

Qwen training JSONL files and LoRA adapter checkpoints are generated artifacts and are not tracked in git.

Generate the classification-format training files with:

```bash
python scripts/10_make_qwen_lora_cls_data.py
```

Run the reported raw classification LoRA pipeline with:

```bash
bash scripts/run_qwen_lora_cls_raw_autodl_fixed_confidence.sh
```

