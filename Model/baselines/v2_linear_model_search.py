"""Broader sparse linear model search for HouseMD v2 labels.

This script is intentionally classical and fast.  With ~7k rows, stronger
TF-IDF linear baselines often beat under-trained transformers, especially for
noisy labels.

Workflow per split family:
  1. Train candidates on train and select per task by val macro-F1.
  2. Retrain the selected candidate on train+val.
  3. Report test metrics and save the selected model bundle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data Analysis" / "outputs"
DEFAULT_OUT = HERE.parent / "runs" / "v2_linear_model_search"
TASKS = ["intent", "emotion", "diagnosis_stage"]
SEED = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    text_col: str
    pipeline_factory: Callable[[], Pipeline]


def normalize_text(s: object) -> str:
    return "" if pd.isna(s) else str(s)


def build_text_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    speaker = out.get("speaker", pd.Series(["unknown"] * len(out))).fillna("unknown").astype(str)
    text = out["text"].fillna("").astype(str)
    prev1 = out.get("prev_text_1", pd.Series([""] * len(out))).fillna("").astype(str)
    prev2 = out.get("prev_text_2", pd.Series([""] * len(out))).fillna("").astype(str)
    symptom = out.get("symptom_list", pd.Series([""] * len(out))).fillna("").astype(str)
    test = out.get("test_list", pd.Series([""] * len(out))).fillna("").astype(str)
    drug = out.get("drug_list", pd.Series([""] * len(out))).fillna("").astype(str)
    procedure = out.get("procedure_list", pd.Series([""] * len(out))).fillna("").astype(str)
    organ = out.get("organ_list", pd.Series([""] * len(out))).fillna("").astype(str)

    out["text_only"] = text
    out["speaker_text"] = "[SPEAKER=" + speaker + "] [TEXT=" + text + "]"
    out["context_text"] = "[PREV=" + prev2 + " || " + prev1 + "] [TEXT=" + text + "]"
    out["entity_text"] = (
        "[SYMPTOM=" + symptom + "] [TEST=" + test + "] [DRUG=" + drug + "] "
        "[PROCEDURE=" + procedure + "] [ORGAN=" + organ + "] [TEXT=" + text + "]"
    )
    out["compact_feature_text"] = (
        "[SPEAKER=" + speaker + "] [SYMPTOM=" + symptom + "] [TEST=" + test + "] "
        "[DRUG=" + drug + "] [PROCEDURE=" + procedure + "] [ORGAN=" + organ + "] "
        "[TEXT=" + text + "]"
    )
    if "feature_text" not in out.columns:
        out["feature_text"] = out["compact_feature_text"]
    return out


def sample_train_for_task(
    train: pd.DataFrame,
    task: str,
    max_majority: int = 1400,
    target_ratio: float = 0.45,
    max_multiplier: int = 6,
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
            desired = min(target_min, int(count * max_multiplier))
            extra_n = max(0, desired - count)
            if extra_n:
                idx = rng.choice(sub.index.to_numpy(), size=extra_n, replace=True)
                pieces.append(train.loc[idx])
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def word_vec(min_df: int = 1, max_features: int = 160_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        min_df=min_df,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        strip_accents=None,
    )


def char_vec(min_df: int = 1, max_features: int = 180_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        min_df=min_df,
        ngram_range=(3, 6),
        max_features=max_features,
        sublinear_tf=True,
    )


def union_vec(min_df: int = 1) -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", word_vec(min_df=min_df, max_features=120_000)),
            ("char", char_vec(min_df=min_df, max_features=120_000)),
        ]
    )


def pipe(vec: BaseEstimator, clf: BaseEstimator) -> Pipeline:
    return Pipeline([("tfidf", vec), ("clf", clf)])


def make_candidates() -> list[Candidate]:
    text_cols = ["text_only", "speaker_text", "context_text", "entity_text", "compact_feature_text", "feature_text"]
    model_factories: list[tuple[str, Callable[[], Pipeline]]] = [
        (
            "word_svc_c05",
            lambda: pipe(word_vec(1), LinearSVC(C=0.5, class_weight="balanced", random_state=SEED, dual="auto")),
        ),
        (
            "word_svc_c1",
            lambda: pipe(word_vec(1), LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, dual="auto")),
        ),
        (
            "char_svc_c1",
            lambda: pipe(char_vec(1), LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, dual="auto")),
        ),
        (
            "union_svc_c05",
            lambda: pipe(union_vec(1), LinearSVC(C=0.5, class_weight="balanced", random_state=SEED, dual="auto")),
        ),
        (
            "union_svc_c1",
            lambda: pipe(union_vec(1), LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, dual="auto")),
        ),
        (
            "union_ridge_a1",
            lambda: pipe(union_vec(1), RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ),
        (
            "union_ridge_a2",
            lambda: pipe(union_vec(1), RidgeClassifier(alpha=2.0, class_weight="balanced")),
        ),
        (
            "union_logreg_c2",
            lambda: pipe(
                union_vec(1),
                LogisticRegression(C=2.0, class_weight="balanced", max_iter=2500, solver="saga", random_state=SEED),
            ),
        ),
        (
            "word_logreg_c4",
            lambda: pipe(
                word_vec(1),
                LogisticRegression(C=4.0, class_weight="balanced", max_iter=2500, solver="lbfgs", random_state=SEED),
            ),
        ),
        (
            "word_sgd_huber",
            lambda: pipe(
                word_vec(1),
                SGDClassifier(
                    loss="modified_huber",
                    alpha=1e-5,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                    tol=1e-4,
                ),
            ),
        ),
    ]

    candidates: list[Candidate] = []
    for text_col in text_cols:
        for model_name, factory in model_factories:
            candidates.append(Candidate(f"{text_col}__{model_name}", text_col, factory))
    return candidates


def fit_predict(model: Pipeline, x_train: pd.Series, y_train: pd.Series, x_eval: pd.Series) -> np.ndarray:
    model.fit(x_train, y_train)
    return model.predict(x_eval)


def decision_scores(model: Pipeline, x: pd.Series, labels: list[str]) -> np.ndarray:
    clf = model.named_steps["clf"]
    classes = list(clf.classes_)
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x)
    else:
        raw = model.decision_function(x)
        if raw.ndim == 1:
            raw = np.vstack([-raw, raw]).T
    aligned = np.zeros((len(x), len(labels)), dtype=np.float64)
    for i, label in enumerate(labels):
        if label in classes:
            aligned[:, i] = raw[:, classes.index(label)]
    return aligned


def evaluate(y_true: pd.Series, pred: np.ndarray, labels: list[str], out_dir: Path, prefix: str, task: str) -> dict:
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_{prefix}_{task}.csv", encoding="utf-8")
    (out_dir / f"report_{prefix}_{task}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "task": task,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "accuracy": float(report.get("accuracy", 0.0)),
    }


def save_predictions(test: pd.DataFrame, task: str, pred: np.ndarray, out_dir: Path, prefix: str) -> None:
    pd.DataFrame(
        {
            "row_id": test.index,
            "label": test[task].astype(str).to_numpy(),
            "prediction": pred,
            "text": test["text"].fillna("").astype(str).to_numpy(),
        }
    ).to_csv(out_dir / f"predictions_{prefix}_{task}.csv", index=False, encoding="utf-8")


def run_task_search(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, task: str, out_dir: Path) -> dict:
    rows = []
    candidates = make_candidates()
    sampled_train = sample_train_for_task(train, task)
    y_train = sampled_train[task].astype(str)
    y_val = val[task].astype(str)

    for cand in candidates:
        x_train = sampled_train[cand.text_col].fillna("").astype(str)
        x_val = val[cand.text_col].fillna("").astype(str)
        model = cand.pipeline_factory()
        pred = fit_predict(model, x_train, y_train, x_val)
        rows.append(
            {
                "task": task,
                "candidate": cand.name,
                "text_col": cand.text_col,
                "val_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
                "val_weighted_f1": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
            }
        )

    search_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    search_df.to_csv(out_dir / f"search_{task}.csv", index=False, encoding="utf-8")
    top = search_df.iloc[0].to_dict()
    selected = next(c for c in candidates if c.name == top["candidate"])

    train_val = pd.concat([train, val], ignore_index=True)
    sampled_train_val = sample_train_for_task(train_val, task)
    labels = sorted(train_val[task].astype(str).unique())

    final_model = selected.pipeline_factory()
    final_model.fit(sampled_train_val[selected.text_col].fillna("").astype(str), sampled_train_val[task].astype(str))
    pred_test = final_model.predict(test[selected.text_col].fillna("").astype(str))
    metrics = evaluate(test[task].astype(str), pred_test, labels, out_dir, "selected", task)
    save_predictions(test, task, pred_test, out_dir, "selected")
    joblib.dump(
        {
            "model": final_model,
            "labels": labels,
            "text_col": selected.text_col,
            "candidate": selected.name,
            "uses_score": hasattr(final_model.named_steps["clf"], "decision_function"),
        },
        out_dir / f"selected_{task}.joblib",
    )
    metrics.update(
        {
            "selected_candidate": selected.name,
            "selected_text_col": selected.text_col,
            "val_macro_f1": float(top["val_macro_f1"]),
            "val_weighted_f1": float(top["val_weighted_f1"]),
            "train_rows_after_sampling": int(len(sampled_train_val)),
        }
    )
    return metrics


def run_family(data_root: Path, split_family: str, out_root: Path) -> pd.DataFrame:
    split_dir = data_root / f"splits_{split_family}"
    out_dir = out_root / split_family
    out_dir.mkdir(parents=True, exist_ok=True)

    train = build_text_variants(pd.read_csv(split_dir / "train.csv", encoding="utf-8"))
    val = build_text_variants(pd.read_csv(split_dir / "val.csv", encoding="utf-8"))
    test = build_text_variants(pd.read_csv(split_dir / "test.csv", encoding="utf-8"))

    rows = []
    for task in TASKS:
        print(f"{split_family} / searching {task}")
        rows.append(run_task_search(train, val, test, task, out_dir))

    summary = pd.DataFrame(rows)
    overall = {
        "split_family": split_family,
        "macro_f1_avg": float(summary["macro_f1"].mean()),
        "weighted_f1_avg": float(summary["weighted_f1"].mean()),
        "accuracy_avg": float(summary["accuracy"].mean()),
    }
    summary.to_csv(out_dir / "selected_task_results.csv", index=False, encoding="utf-8")
    (out_dir / "selected_summary.json").write_text(
        json.dumps({"overall": overall, "tasks": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary.assign(**overall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-family", choices=["score", "fair", "both"], default="both")
    args = parser.parse_args()

    families = ["score", "fair"] if args.split_family == "both" else [args.split_family]
    all_rows = []
    for family in families:
        all_rows.append(run_family(args.data_root, family, args.out_dir))
    pd.concat(all_rows, ignore_index=True).to_csv(args.out_dir / "all_selected_task_results.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
