from __future__ import annotations

import argparse
from typing import List

from rich.console import Console

console = Console()


def train_token_labeler(args: argparse.Namespace) -> None:
    """Skeleton token labeling trainer for BIO tags: STEP, CONDITION, ARG_NAME, ARG_VALUE."""
    try:
        from datasets import Dataset
        from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
        import numpy as np

        model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tiny dummy dataset
        texts = [
            "Walk to Machine A and read pressure_gauge_A",
            "Toggle valve A_inlet to open",
        ]
        labels = [
            # O O O STEP O O O O ARG_VALUE
            [0, 0, 0, 1, 0, 0, 0, 0, 4],
            [1, 2, 3, 0, 0],
        ]

        label2id = {"O": 0, "B-STEP": 1, "I-STEP": 2, "B-ARG_NAME": 3, "B-ARG_VALUE": 4, "B-CONDITION": 5}
        id2label = {v: k for k, v in label2id.items()}

        def tokenize_and_align(_texts, _labels):
            enc = tokenizer(_texts, truncation=True, padding=True)
            # For skeleton, truncate/pad labels to input ids length
            out_labels = []
            for ids, lab in zip(enc["input_ids"], _labels):
                arr = lab[: len(ids)] + [0] * max(0, len(ids) - len(lab))
                out_labels.append(arr)
            enc["labels"] = out_labels
            return enc

        ds = Dataset.from_dict(tokenize_and_align(texts, labels))

        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )
        args_tr = TrainingArguments(
            output_dir=args.out_dir,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",
        )

        def compute_metrics(eval_pred):
            # Dummy F1-like metric
            logits, labs = eval_pred
            preds = logits.argmax(-1)
            acc = (preds == labs).mean().item()
            return {"f1": float(acc)}

        trainer = Trainer(model=model, args=args_tr, train_dataset=ds, eval_dataset=ds, compute_metrics=compute_metrics)
        trainer.train()
        console.log(f"Saved token labeler to {args.out_dir}")
    except Exception as e:
        console.log(f"[yellow]Skipping token labeling due to: {e}[/yellow]")


def build_parser():
    p = argparse.ArgumentParser(description="Optional BIO token labeling trainer (skeleton).")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--out_dir", type=str, default="artifacts/ner_token")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_token_labeler(args)


