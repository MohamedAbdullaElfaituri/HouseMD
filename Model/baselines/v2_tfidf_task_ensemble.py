"""Task-specific TF-IDF baselines and soft-vote ensembles for v2 data.

Runs fast ablations on both split families:
  - taxonomy_v2_text
  - feature_text
  - feature_text_sampled
  - feature_text_sampled_augmented
  - feature_text_soft_vote

Outputs reports under Model/runs/v2_tfidf_task_ensemble/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data Analysis" / "outputs"
DEFAULT_OUT = HERE.parent / "runs" / "v2_tfidf_task_ensemble"
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


def word_model(class_weight: str | dict | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=2,
                    ngram_range=(1, 2),
                    max_features=90000,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    class_weight=class_weight,
                    max_iter=2500,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def char_model(class_weight: str | dict | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    min_df=2,
                    ngram_range=(3, 5),
                    max_features=90000,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=3.0,
                    class_weight=class_weight,
                    max_iter=2500,
                    solver="lbfgs",
                ),
            ),
        ]
    )


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


def proba_aligned(model: Pipeline, x: pd.Series, labels: list[str]) -> np.ndarray:
    probs = model.predict_proba(x)
    aligned = np.zeros((len(x), len(labels)), dtype=np.float64)
    class_to_col = {label: i for i, label in enumerate(model.named_steps["clf"].classes_)}
    for j, label in enumerate(labels):
        aligned[:, j] = probs[:, class_to_col[label]]
    return aligned


def evaluate_predictions(y_true: pd.Series, labels: list[str], pred: np.ndarray, out_dir: Path, task: str, variant: str) -> dict:
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_{variant}_{task}.csv", encoding="utf-8")
    (out_dir / f"report_{variant}_{task}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "variant": variant,
        "task": task,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "accuracy": float(report.get("accuracy", 0.0)),
    }


def run_variant(train: pd.DataFrame, test: pd.DataFrame, task: str, variant: str, text_col: str, sampled: bool, augmented: bool,
                soft_vote: bool, out_dir: Path, save_models: bool) -> dict:
    task_train = train.copy()
    if sampled:
        task_train = sample_train_for_task(task_train, task)
    if augmented:
        task_train = augment_minority_rows(task_train, task, text_col)

    labels = sorted(task_train[task].astype(str).unique())
    x_train = task_train[text_col].fillna("").astype(str)
    y_train = task_train[task].astype(str)
    x_test = test[text_col].fillna("").astype(str)
    y_test = test[task].astype(str)

    w_model = word_model(class_weight="balanced")
    w_model.fit(x_train, y_train)

    if soft_vote:
        c_model = char_model(class_weight="balanced")
        c_model.fit(x_train, y_train)
        probs = (proba_aligned(w_model, x_test, labels) + proba_aligned(c_model, x_test, labels)) / 2.0
        pred = np.array(labels, dtype=object)[probs.argmax(axis=1)]
        if save_models:
            joblib.dump({"word": w_model, "char": c_model, "labels": labels, "text_col": text_col}, out_dir / f"model_{variant}_{task}.joblib")
    else:
        pred = w_model.predict(x_test)
        if save_models:
            joblib.dump({"word": w_model, "labels": labels, "text_col": text_col}, out_dir / f"model_{variant}_{task}.joblib")

    pred_df = pd.DataFrame(
        {
            "row_id": test.index,
            "task": task,
            "label": y_test.to_numpy(),
            "prediction": pred,
            "text": test["text"].to_numpy(),
        }
    )
    pred_df.to_csv(out_dir / f"predictions_{variant}_{task}.csv", index=False, encoding="utf-8")
    result = evaluate_predictions(y_test, labels, pred, out_dir, task, variant)
    result.update({"train_rows": int(len(task_train)), "test_rows": int(len(test)), "labels": len(labels)})
    return result


def run_split_family(data_root: Path, split_family: str, out_root: Path, save_models: bool) -> pd.DataFrame:
    split_dir = data_root / f"splits_{split_family}"
    out_dir = out_root / split_family
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(split_dir / "train.csv", encoding="utf-8")
    val = pd.read_csv(split_dir / "val.csv", encoding="utf-8")
    test = pd.read_csv(split_dir / "test.csv", encoding="utf-8")
    train_val = pd.concat([train, val], ignore_index=True)

    variants = [
        ("taxonomy_v2_text", "text", False, False, False),
        ("feature_text", "feature_text", False, False, False),
        ("feature_text_sampled", "feature_text", True, False, False),
        ("feature_text_sampled_augmented", "feature_text", True, True, False),
        ("feature_text_soft_vote", "feature_text", True, True, True),
    ]

    rows = []
    for variant, text_col, sampled, augmented, soft_vote in variants:
        for task in TASKS:
            print(f"{split_family} / {variant} / {task}")
            rows.append(
                run_variant(
                    train_val,
                    test,
                    task,
                    variant,
                    text_col,
                    sampled,
                    augmented,
                    soft_vote,
                    out_dir,
                    save_models=save_models and variant == "feature_text_soft_vote",
                )
            )

    results = pd.DataFrame(rows)
    overall = (
        results.groupby("variant", as_index=False)
        .agg(macro_f1_avg=("macro_f1", "mean"), weighted_f1_avg=("weighted_f1", "mean"), accuracy_avg=("accuracy", "mean"))
        .sort_values("macro_f1_avg", ascending=False)
    )
    results.to_csv(out_dir / "ablation_task_results.csv", index=False, encoding="utf-8")
    overall.to_csv(out_dir / "ablation_summary.csv", index=False, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "split_family": split_family,
                "best_variant": overall.iloc[0].to_dict(),
                "task_results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return overall.assign(split_family=split_family)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-family", choices=["score", "fair", "both"], default="both")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    split_families = ["score", "fair"] if args.split_family == "both" else [args.split_family]
    summaries = []
    for split_family in split_families:
        summaries.append(run_split_family(args.data_root, split_family, args.out_dir, args.save_models))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.out_dir / "all_ablation_summary.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
