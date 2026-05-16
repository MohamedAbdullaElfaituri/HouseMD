"""Frozen transformer embeddings + linear classifiers for HouseMD v2.

This is a middle ground between TF-IDF and full fine-tuning:
use a pretrained encoder only to produce mean-pooled embeddings, then train
small linear classifiers.  It is useful for noisy/small datasets because it is
fast to iterate and less prone to overfitting than full fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data Analysis" / "outputs"
DEFAULT_OUT = HERE.parent / "runs" / "v2_embedding_linear"
TASKS = ["intent", "emotion", "diagnosis_stage"]
SEED = 42


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def encode_texts(
    texts: list[str],
    model_name: str,
    cache_path: Path,
    batch_size: int,
    max_len: int,
    fp16: bool,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if fp16 and device == "cuda":
        model = model.half()
    loader = DataLoader(TextDataset(texts, tokenizer, max_len), batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            pooled = mean_pool(out.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled.float(), p=2, dim=1)
            chunks.append(pooled.cpu().numpy())
    arr = np.vstack(chunks).astype("float32")
    np.save(cache_path, arr)
    return arr


def sample_indices(y: pd.Series, max_majority: int = 1400, target_ratio: float = 0.45, max_multiplier: int = 6) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    counts = y.value_counts()
    cap = min(max_majority, int(counts.max()))
    target_min = int(cap * target_ratio)
    pieces = []
    for label, count in counts.items():
        idx = np.where(y.to_numpy() == label)[0]
        if count > max_majority:
            pieces.append(rng.choice(idx, size=max_majority, replace=False))
        else:
            pieces.append(idx)
            if count < target_min:
                desired = min(target_min, int(count * max_multiplier))
                extra = max(0, desired - count)
                if extra:
                    pieces.append(rng.choice(idx, size=extra, replace=True))
    out = np.concatenate(pieces)
    rng.shuffle(out)
    return out


def candidates():
    return {
        "logreg_c1_bal": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2500, solver="lbfgs", random_state=SEED),
        "logreg_c2_bal": LogisticRegression(C=2.0, class_weight="balanced", max_iter=2500, solver="lbfgs", random_state=SEED),
        "svc_c05_bal": LinearSVC(C=0.5, class_weight="balanced", dual="auto", random_state=SEED),
        "svc_c1_bal": LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=SEED),
        "ridge_a1_bal": RidgeClassifier(alpha=1.0, class_weight="balanced"),
        "ridge_a2_bal": RidgeClassifier(alpha=2.0, class_weight="balanced"),
    }


def evaluate(y_true, pred, labels, out_dir: Path, task: str, prefix: str) -> dict:
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_{prefix}_{task}.csv", encoding="utf-8")
    (out_dir / f"report_{prefix}_{task}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "task": task,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "accuracy": float(report.get("accuracy", 0.0)),
    }


def run_family(args, family: str) -> pd.DataFrame:
    split_dir = args.data_root / f"splits_{family}"
    out_dir = args.out_dir / family / args.text_col.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(split_dir / "train.csv", encoding="utf-8")
    val = pd.read_csv(split_dir / "val.csv", encoding="utf-8")
    test = pd.read_csv(split_dir / "test.csv", encoding="utf-8")
    all_df = pd.concat([train, val, test], ignore_index=True)
    texts = all_df[args.text_col].fillna("").astype(str).tolist()
    cache_name = f"{family}_{args.text_col}_{args.model_name.replace('/', '__')}_len{args.max_len}.npy"
    emb = encode_texts(texts, args.model_name, args.out_dir / "cache" / cache_name, args.batch_size, args.max_len, args.fp16)
    n_train, n_val, n_test = len(train), len(val), len(test)
    x_train = emb[:n_train]
    x_val = emb[n_train : n_train + n_val]
    x_test = emb[n_train + n_val :]
    x_train_val = emb[: n_train + n_val]
    train_val = pd.concat([train, val], ignore_index=True)

    rows = []
    search_rows = []
    for task in TASKS:
        y_train = train[task].astype(str)
        y_val = val[task].astype(str)
        idx = sample_indices(y_train)
        best = None
        for name, clf in candidates().items():
            model = clf
            model.fit(x_train[idx], y_train.iloc[idx])
            pred = model.predict(x_val)
            row = {
                "task": task,
                "candidate": name,
                "val_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
                "val_weighted_f1": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
            }
            search_rows.append(row)
            if best is None or row["val_macro_f1"] > best["val_macro_f1"]:
                best = row
        assert best is not None
        final_model = candidates()[best["candidate"]]
        y_train_val = train_val[task].astype(str)
        idx_tv = sample_indices(y_train_val)
        final_model.fit(x_train_val[idx_tv], y_train_val.iloc[idx_tv])
        pred_test = final_model.predict(x_test)
        labels = sorted(y_train_val.unique())
        metrics = evaluate(test[task].astype(str), pred_test, labels, out_dir, task, "selected")
        metrics.update(best)
        rows.append(metrics)
        joblib.dump(
            {
                "model": final_model,
                "labels": labels,
                "task": task,
                "text_col": args.text_col,
                "model_name": args.model_name,
                "max_len": args.max_len,
                "embedding": "mean_pool_l2",
            },
            out_dir / f"selected_{task}.joblib",
        )
    pd.DataFrame(search_rows).to_csv(out_dir / "search_results.csv", index=False, encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "selected_task_results.csv", index=False, encoding="utf-8")
    overall = {
        "split_family": family,
        "text_col": args.text_col,
        "model_name": args.model_name,
        "macro_f1_avg": float(summary["macro_f1"].mean()),
        "weighted_f1_avg": float(summary["weighted_f1"].mean()),
        "accuracy_avg": float(summary["accuracy"].mean()),
        "tasks": rows,
    }
    (out_dir / "selected_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary.assign(split_family=family, text_col=args.text_col, model_name=args.model_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-family", choices=["score", "fair", "both"], default="score")
    parser.add_argument("--text-col", default="feature_text")
    parser.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    families = ["score", "fair"] if args.split_family == "both" else [args.split_family]
    rows = [run_family(args, family) for family in families]
    pd.concat(rows, ignore_index=True).to_csv(args.out_dir / "all_selected_task_results.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
