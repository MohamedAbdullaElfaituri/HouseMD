"""Multi-task Turkish encoder classifier for House MD Turkish dialogue.

Three heads on a shared Turkish encoder:
  - intent          (≤13 classes)
  - emotion         (≤12 classes)
  - diagnosis_stage (≤9 classes)

Key fixes vs v1:
  - No aggressive 1/sqrt(freq) class weights (caused grad_norm = 32000 explosions).
    Optional --class-weight balanced uses milder sklearn-style 1/N weights.
  - No freeze-then-unfreeze two-stage (broke optimizer continuity).
  - max_grad_norm 5.0 (was 1.0, was clipping useful gradient).
  - label smoothing 0.1.
  - Drops classes with <min_class_count training samples (default 10) → merge to "diğer".

Usage:
    python Model/train_multitask.py [--epochs 8] [--batch-size 32] [--class-weight none|balanced]

RTX PRO 6000 / A100 / T4 all supported. fp16 default on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_SPLITS = PROJECT_ROOT / "Data Analysis" / "outputs" / "splits"
DEFAULT_OUT = HERE / "runs" / "multitask_berturk_nosarcasm_balanced"

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
TASKS = ["intent", "emotion", "diagnosis_stage"]
TASK_WEIGHTS = {"intent": 1.0, "emotion": 1.0, "diagnosis_stage": 1.0}
RARE_CLASS_FALLBACK = "diğer"


# -----------------------------------------------------------------------------
# Rare-class filtering
# -----------------------------------------------------------------------------
def consolidate_rare_classes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Merge classes seen fewer than `min_count` times in train into RARE_CLASS_FALLBACK.
    Applies the same mapping to val/test. Returns kept-class sets per task."""
    kept: dict[str, set[str]] = {}
    for task in TASKS:
        counts = train_df[task].astype(str).fillna("").value_counts()
        keep = set(counts[counts >= min_count].index)
        keep.discard("")
        keep.add(RARE_CLASS_FALLBACK)  # always allow merging into this bucket
        kept[task] = keep

    def remap(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for task in TASKS:
            df[task] = df[task].astype(str).where(df[task].astype(str).isin(kept[task]), RARE_CLASS_FALLBACK)
            df.loc[df[task] == "nan", task] = pd.NA
        return df

    return remap(train_df), remap(val_df), remap(test_df), kept


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class HouseMDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_maps: dict[str, dict[str, int]], max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row.get("text") or "")
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        for task in TASKS:
            val = row.get(task)
            lm = self.label_maps[task]
            if pd.isna(val) or str(val).strip() in ("", "nan"):
                item[f"label_{task}"] = -100
                continue
            key = str(val).strip()
            item[f"label_{task}"] = lm.get(key, -100)
        return item


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MultiTaskBerturk(nn.Module):
    def __init__(self, model_name: str, n_classes: dict[str, int], dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, n in n_classes.items()})

    def forward(self, input_ids, attention_mask, **_labels) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return {task: head(cls) for task, head in self.heads.items()}


# -----------------------------------------------------------------------------
# Trainer with multi-task loss
# -----------------------------------------------------------------------------
class MultiTaskTrainer(Trainer):
    def __init__(self, class_weights: dict[str, torch.Tensor] | None = None,
                 label_smoothing: float = 0.0, **kw):
        super().__init__(**kw)
        self.task_loss_fns = {}
        for task in TASKS:
            w = class_weights.get(task) if class_weights else None
            self.task_loss_fns[task] = nn.CrossEntropyLoss(
                weight=w, ignore_index=-100, label_smoothing=label_smoothing
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = {task: inputs.pop(f"label_{task}") for task in TASKS}
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        total = None
        for task in TASKS:
            logits = outputs[task]
            tgt = labels[task].to(logits.device)
            fn = self.task_loss_fns[task]
            if fn.weight is not None and fn.weight.device != logits.device:
                fn.weight = fn.weight.to(logits.device)
            loss_t = fn(logits, tgt) * TASK_WEIGHTS[task]
            total = loss_t if total is None else total + loss_t
        return (total, outputs) if return_outputs else total

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = {task: inputs.pop(f"label_{task}") for task in TASKS}
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            preds = torch.stack([outputs[task].argmax(dim=-1) for task in TASKS], dim=1)
            labels_tensor = torch.stack([labels[task] for task in TASKS], dim=1)
        return (None, preds, labels_tensor)


def compute_metrics(eval_pred) -> dict[str, float]:
    preds, labels = eval_pred
    out = {}
    macros = []
    for i, task in enumerate(TASKS):
        p, t = preds[:, i], labels[:, i]
        mask = t != -100
        if mask.sum() == 0:
            out[f"f1_macro_{task}"] = float("nan")
            continue
        f1 = f1_score(t[mask], p[mask], average="macro", zero_division=0)
        out[f"f1_macro_{task}"] = float(f1)
        macros.append(f1)
    out["f1_macro_overall"] = float(np.mean(macros)) if macros else 0.0
    return out


# -----------------------------------------------------------------------------
# Label maps + (optional) class weights
# -----------------------------------------------------------------------------
def build_label_maps(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    label_maps: dict[str, dict[str, int]] = {}
    for task in TASKS:
        series = train_df[task].dropna()
        vals = sorted({str(v).strip() for v in series.unique() if str(v).strip() not in ("", "nan")})
        label_maps[task] = {v: i for i, v in enumerate(vals)}
    return label_maps


def compute_class_weights_balanced(train_df: pd.DataFrame, label_maps: dict[str, dict[str, int]]) -> dict[str, torch.Tensor]:
    """sklearn-style 'balanced' weights: N / (K * count_c). Mild compared to 1/sqrt(count)."""
    weights: dict[str, torch.Tensor] = {}
    for task in TASKS:
        lm = label_maps[task]
        n_classes = len(lm)
        counts = np.ones(n_classes, dtype=np.float64)  # smoothing
        for v in train_df[task].dropna():
            key = str(v).strip()
            idx = lm.get(key)
            if idx is not None:
                counts[idx] += 1
        n_total = counts.sum()
        w = n_total / (n_classes * counts)
        # cap to avoid explosion: max weight ≤ 5x mean
        w = np.minimum(w, 5.0 * w.mean())
        weights[task] = torch.tensor(w, dtype=torch.float32)
    return weights


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-class-count", type=int, default=10,
                        help="Classes seen fewer than this in train → merged to 'diğer'.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    train_df = pd.read_csv(args.data_dir / "train.csv")
    val_df = pd.read_csv(args.data_dir / "val.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    for df in (train_df, val_df, test_df):
        df["text"] = df["text"].fillna("").astype(str)
    print(f"  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    print(f"Consolidating classes seen <{args.min_class_count}× in train → '{RARE_CLASS_FALLBACK}'...")
    train_df, val_df, test_df, kept = consolidate_rare_classes(train_df, val_df, test_df, args.min_class_count)
    for task in TASKS:
        kept_top = sorted(kept[task] - {RARE_CLASS_FALLBACK})
        print(f"  {task}: kept {len(kept_top)} classes + '{RARE_CLASS_FALLBACK}' bucket")

    print("Building label maps (from train split)...")
    label_maps = build_label_maps(train_df)
    for task in TASKS:
        print(f"  {task}: {len(label_maps[task])} classes → {sorted(label_maps[task].keys())[:5]}...")
    (args.out_dir / "label_maps.json").write_text(
        json.dumps(label_maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "tasks": TASKS,
                "max_len": args.max_len,
                "class_weight": args.class_weight,
                "min_class_count": args.min_class_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    class_weights = None
    if args.class_weight == "balanced":
        print("Computing balanced class weights (capped at 5x mean)...")
        class_weights = compute_class_weights_balanced(train_df, label_maps)
        for task in TASKS:
            w = class_weights[task].numpy()
            print(f"  {task}: weights min={w.min():.2f} max={w.max():.2f} mean={w.mean():.2f}")

    print(f"Loading tokenizer + model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    n_classes = {task: len(label_maps[task]) for task in TASKS}
    model = MultiTaskBerturk(args.model_name, n_classes)

    train_ds = HouseMDDataset(train_df, tokenizer, label_maps, max_len=args.max_len)
    val_ds = HouseMDDataset(val_df, tokenizer, label_maps, max_len=args.max_len)
    test_ds = HouseMDDataset(test_df, tokenizer, label_maps, max_len=args.max_len)

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
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
        metric_for_best_model="f1_macro_overall",
        greater_is_better=True,
        save_total_limit=2,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2 if os.name != "nt" else 0,
    )

    trainer = MultiTaskTrainer(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\n=== Training ===")
    trainer.train()

    print("\n=== Final eval on TEST split ===")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    # Per-class detailed report
    preds_output = trainer.predict(test_ds)
    preds, labels = preds_output.predictions, preds_output.label_ids
    reports = {}
    for i, task in enumerate(TASKS):
        p, t = preds[:, i], labels[:, i]
        mask = t != -100
        if mask.sum() == 0:
            continue
        inv_lm = {v: k for k, v in label_maps[task].items()}
        present = sorted(set(t[mask].tolist()) | set(p[mask].tolist()))
        target_names = [inv_lm[i] for i in present]
        rep = classification_report(
            t[mask], p[mask], labels=present, target_names=target_names,
            output_dict=True, zero_division=0,
        )
        reports[task] = rep
    (args.out_dir / "test_classification_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save final model
    final_dir = args.out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.bin")
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel + tokenizer saved to {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
