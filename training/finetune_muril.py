"""
Fine-tune google/muril-base-cased for 3-class unsafe-content detection.

Label schema:
    0 = safe
    1 = toxic      (abusive / hateful language)
    2 = jailbreak  (prompt-injection / role-play evasion)

Inputs: train.csv, val.csv, test.csv produced by prepare_data.py
Output: models/muril-guardrail/ (HuggingFace-format checkpoint)

Usage (M3 Mac, smoke test):
    python training/finetune_muril.py \\
        --train data/training/train.csv \\
        --val   data/training/val.csv   \\
        --test  data/training/test.csv  \\
        --epochs 1 --batch-size 8 --lr 2e-5 --max-length 128

Usage (DGX A100, full run):
    python training/finetune_muril.py \\
        --train data/training/train.csv \\
        --val   data/training/val.csv   \\
        --test  data/training/test.csv  \\
        --epochs 5 --batch-size 32 --lr 2e-5 --max-length 256
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

LABEL2ID = {"safe": 0, "toxic": 1, "jailbreak": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_csv(path: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenise(texts: list[str], tokenizer, max_length: int):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score  # type: ignore
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import classification_report  # type: ignore

    device = _pick_device()
    print(f"Device: {device}")
    print(f"Loading tokenizer and model from {args.base_model!r}...")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Move model to device manually (Trainer will also do this, but explicit is clearer)
    model.to(device)

    # Load data
    print("Loading data...")
    train_texts, train_labels = _load_csv(args.train)
    val_texts,   val_labels   = _load_csv(args.val)
    test_texts,  test_labels  = _load_csv(args.test)
    print(f"  train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # Tokenise
    print(f"Tokenising (max_length={args.max_length})...")
    train_enc = _tokenise(train_texts, tokenizer, args.max_length)
    val_enc   = _tokenise(val_texts,   tokenizer, args.max_length)
    test_enc  = _tokenise(test_texts,  tokenizer, args.max_length)

    train_ds = _TextDataset(train_enc, train_labels)
    val_ds   = _TextDataset(val_enc,   val_labels)
    test_ds  = _TextDataset(test_enc,  test_labels)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fp16 only on CUDA — MPS and CPU don't support it
    use_fp16 = (device == "cuda")

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=10,
        report_to="none",   # no wandb / tensorboard unless user sets up
        dataloader_pin_memory=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )

    print(f"\nFine-tuning for {args.epochs} epoch(s)...")
    trainer.train()

    # Final test evaluation
    print("\n--- Test set evaluation ---")
    test_preds_out = trainer.predict(test_ds)
    test_preds = np.argmax(test_preds_out.predictions, axis=-1)
    print(classification_report(
        test_labels, test_preds,
        target_names=["safe", "toxic", "jailbreak"],
        digits=3,
    ))

    # Save final model + tokenizer
    print(f"\nSaving checkpoint to {out_dir}...")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Checkpoint saved: {out_dir}")
    print("\nTo use this checkpoint, set in config.py:")
    print(f'  MODELS["toxicity"]  = "{out_dir}"')
    print(f'  MODELS["jailbreak"] = "{out_dir}"')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fine-tune MuRIL for Hinglish guardrails")
    ap.add_argument("--base-model", default="google/muril-base-cased",
                    help="HuggingFace model ID or local path to start from")
    ap.add_argument("--train",  default="data/training/train.csv")
    ap.add_argument("--val",    default="data/training/val.csv")
    ap.add_argument("--test",   default="data/training/test.csv")
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--batch-size", type=int,   default=16)
    ap.add_argument("--lr",         type=float, default=2e-5)
    ap.add_argument("--max-length", type=int,   default=128)
    ap.add_argument("--output", default="models/muril-guardrail",
                    help="directory to save the fine-tuned checkpoint")
    args = ap.parse_args()

    for p in (args.train, args.val, args.test):
        if not Path(p).exists():
            sys.exit(f"Data file not found: {p}\nRun: python training/prepare_data.py")

    train(args)


if __name__ == "__main__":
    main()
