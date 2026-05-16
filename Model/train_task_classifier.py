"""Train one task-specific Turkish transformer classifier on v2 splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data Analysis" / "outputs" / "splits_score"
DEFAULT_OUT = HERE / "runs" / "v2_task_berturk_score"
DEFAULT_MODEL = "dbmdz/bert-base-turkish-cased"
TASKS = ["intent", "emotion", "diagnosis_stage"]
SEED = 42

AUG_REPLACEMENTS = [
    (r"\bhemen\b", "derhal"),
    (r"\bgerekli\b", "lazım"),
    (r"\bolabilir\b", "mümkün"),
    (r"\byapın\b", "uygulayın"),
    (r"\başlayın\b", "başlatın"),
    (r"\bisteyin\b", "talep edin"),
    (r"\bkötüleşiyor\b", "ağırlaşıyor"),
]


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_map: dict[str, int], text_col: str, task: str, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.text_col = text_col
        self.task = task
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row.get(self.text_col) or ""),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_map[str(row[self.task])], dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.loss_fn.weight is not None and self.loss_fn.weight.device != logits.device:
            self.loss_fn.weight = self.loss_fn.weight.to(logits.device)
        loss = self.loss_fn(logits, labels.to(logits.device))
        return (loss, outputs) if return_outputs else loss


def sample_train_for_task(
    train: pd.DataFrame,
    task: str,
    max_majority: int = 1500,
    target_ratio: float = 0.40,
    max_multiplier: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    counts = train[task].value_counts()
    majority_cap = min(max_majority, int(counts.max()))
    target_min = int(majority_cap * target_ratio)
    pieces = []
    for label, count in counts.items():
        sub = train[train[task] == label]
        if count > max_majority:
            pieces.append(sub.sample(n=max_majority, random_state=SEED))
            continue
        pieces.append(sub)
        if count < target_min:
            desired = min(target_min, count * max_multiplier)
            extra_n = max(0, desired - count)
            if extra_n:
                sampled_idx = rng.choice(sub.index.to_numpy(), size=extra_n, replace=True)
                pieces.append(train.loc[sampled_idx])
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def augment_minority_rows(train: pd.DataFrame, task: str, text_col: str) -> pd.DataFrame:
    counts = train[task].value_counts()
    median_count = int(counts.median())
    rows = []
    for _, row in train.iterrows():
        if counts[row[task]] >= median_count:
            continue
        text = str(row[text_col])
        augmented = text
        for pattern, repl in AUG_REPLACEMENTS:
            candidate = re.sub(pattern, repl, augmented, count=1, flags=re.IGNORECASE)
            if candidate != augmented:
                augmented = candidate
                break
        if augmented == text:
            augmented = text + " [AUG=minority]"
        new_row = row.copy()
        new_row[text_col] = augmented
        new_row["is_augmented"] = True
        rows.append(new_row)
    if not rows:
        return train
    return pd.concat([train, pd.DataFrame(rows)], ignore_index=True).sample(frac=1.0, random_state=SEED)


def build_label_map(train: pd.DataFrame, task: str) -> dict[str, int]:
    labels = sorted(str(x) for x in train[task].dropna().unique())
    return {label: i for i, label in enumerate(labels)}


def compute_class_weights(train: pd.DataFrame, task: str, label_map: dict[str, int]) -> torch.Tensor:
    counts = np.ones(len(label_map), dtype=np.float64)
    for value in train[task].astype(str):
        counts[label_map[value]] += 1.0
    total = counts.sum()
    weights = total / (len(label_map) * counts)
    weights = np.minimum(weights, 3.0 * weights.mean())
    return torch.tensor(weights, dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--text-col", default="feature_text")
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = parser.parse_args()

    set_seed(args.seed)
    run_dir = args.out_dir / args.task
    run_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv", encoding="utf-8")
    val = pd.read_csv(args.data_dir / "val.csv", encoding="utf-8")
    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8")
    train["is_augmented"] = False
    if args.oversample:
        train = sample_train_for_task(train, args.task)
    if args.augment:
        train = augment_minority_rows(train, args.task, args.text_col)

    label_map = build_label_map(train, args.task)
    inv_map = {v: k for k, v in label_map.items()}
    (run_dir / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "model_config.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "model_name": args.model_name,
                "text_col": args.text_col,
                "max_len": args.max_len,
                "oversample": args.oversample,
                "augment": args.augment,
                "class_weight": args.class_weight,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_map),
        id2label={i: label for label, i in label_map.items()},
        label2id=label_map,
    )

    train_ds = TextDataset(train, tokenizer, label_map, args.text_col, args.task, args.max_len)
    val_ds = TextDataset(val, tokenizer, label_map, args.text_col, args.task, args.max_len)
    test_ds = TextDataset(test, tokenizer, label_map, args.text_col, args.task, args.max_len)

    class_weights = compute_class_weights(train, args.task, label_map) if args.class_weight == "balanced" else None

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    y_true = pred_output.label_ids
    y_pred = logits.argmax(axis=-1)
    target_names = [inv_map[i] for i in range(len(inv_map))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    (run_dir / "test_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "label": [inv_map[int(x)] for x in y_true],
            "prediction": [inv_map[int(x)] for x in y_pred],
            "text": test["text"].to_numpy(),
            "feature_text": test[args.text_col].to_numpy(),
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8")

    final_dir = run_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
