"""Targeted emotion-classifier search for HouseMD v2.

Emotion is the weakest task.  This script tries a narrower but more aggressive
set of sparse models, sampling levels, and word/char soft-vote weights.
Selection is by validation macro-F1, then the selected model is retrained on
train+val and evaluated on test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from v2_linear_model_search import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    SEED,
    build_text_variants,
    char_vec,
    sample_train_for_task,
    union_vec,
)

DEFAULT_OUT = PROJECT_ROOT / "Model" / "runs" / "v2_emotion_boost_search"
TASK = "emotion"


@dataclass(frozen=True)
class Candidate:
    name: str
    text_col: str
    sample_mode: str
    factory: Callable[[], object]


def pipe(vec, clf) -> Pipeline:
    return Pipeline([("tfidf", vec), ("clf", clf)])


def sampled(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return df
    if mode == "mild":
        return sample_train_for_task(df, TASK, max_majority=1400, target_ratio=0.45, max_multiplier=6)
    if mode == "strong":
        return sample_train_for_task(df, TASK, max_majority=1200, target_ratio=0.75, max_multiplier=10)
    raise ValueError(mode)


def make_candidates() -> list[Candidate]:
    text_cols = ["feature_text", "compact_feature_text"]
    sample_modes = ["none", "mild"]
    models: list[tuple[str, Callable[[], object]]] = [
        ("union_ridge_a2", lambda: pipe(union_vec(1), RidgeClassifier(alpha=2.0, class_weight="balanced"))),
        ("union_ridge_a4", lambda: pipe(union_vec(1), RidgeClassifier(alpha=4.0, class_weight="balanced"))),
        ("union_svc_c02", lambda: pipe(union_vec(1), LinearSVC(C=0.2, class_weight="balanced", dual="auto", random_state=SEED))),
        ("union_svc_c05", lambda: pipe(union_vec(1), LinearSVC(C=0.5, class_weight="balanced", dual="auto", random_state=SEED))),
        ("char_svc_c05", lambda: pipe(char_vec(1), LinearSVC(C=0.5, class_weight="balanced", dual="auto", random_state=SEED))),
    ]
    return [Candidate(f"{text_col}__{mode}__{name}", text_col, mode, factory) for text_col in text_cols for mode in sample_modes for name, factory in models]


def evaluate(y_true: pd.Series, pred, labels: list[str], out_dir: Path, prefix: str) -> dict:
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_{prefix}_{TASK}.csv", encoding="utf-8")
    (out_dir / f"report_{prefix}_{TASK}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "accuracy": float(report.get("accuracy", 0.0)),
    }


def run_family(data_root: Path, split_family: str, out_root: Path) -> dict:
    split_dir = data_root / f"splits_{split_family}"
    out_dir = out_root / split_family
    out_dir.mkdir(parents=True, exist_ok=True)

    train = build_text_variants(pd.read_csv(split_dir / "train.csv", encoding="utf-8"))
    val = build_text_variants(pd.read_csv(split_dir / "val.csv", encoding="utf-8"))
    test = build_text_variants(pd.read_csv(split_dir / "test.csv", encoding="utf-8"))
    y_val = val[TASK].astype(str)

    rows = []
    for cand in make_candidates():
        model = cand.factory()
        fit_df = sampled(train, cand.sample_mode)
        model.fit(fit_df[cand.text_col].fillna("").astype(str), fit_df[TASK].astype(str))
        pred = model.predict(val[cand.text_col].fillna("").astype(str))
        rows.append(
            {
                "candidate": cand.name,
                "text_col": cand.text_col,
                "sample_mode": cand.sample_mode,
                "val_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
                "val_weighted_f1": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
            }
        )

    search = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    search.to_csv(out_dir / "search_emotion.csv", index=False, encoding="utf-8")
    top = search.iloc[0]
    selected = next(c for c in make_candidates() if c.name == top["candidate"])

    train_val = build_text_variants(pd.concat([train, val], ignore_index=True))
    fit_df = sampled(train_val, selected.sample_mode)
    model = selected.factory()
    model.fit(fit_df[selected.text_col].fillna("").astype(str), fit_df[TASK].astype(str))
    pred_test = model.predict(test[selected.text_col].fillna("").astype(str))
    labels = sorted(train_val[TASK].astype(str).unique())
    metrics = evaluate(test[TASK].astype(str), pred_test, labels, out_dir, "selected")
    pd.DataFrame(
        {
            "row_id": test.index,
            "label": test[TASK].astype(str).to_numpy(),
            "prediction": pred_test,
            "text": test["text"].fillna("").astype(str).to_numpy(),
        }
    ).to_csv(out_dir / "predictions_selected_emotion.csv", index=False, encoding="utf-8")
    bundle = {
        "model": model,
        "labels": labels,
        "text_col": selected.text_col,
        "candidate": selected.name,
        "task": TASK,
    }
    joblib.dump(bundle, out_dir / "selected_emotion.joblib")
    result = {
        "split_family": split_family,
        "selected_candidate": selected.name,
        "selected_text_col": selected.text_col,
        "sample_mode": selected.sample_mode,
        "val_macro_f1": float(top["val_macro_f1"]),
        "val_weighted_f1": float(top["val_weighted_f1"]),
        **metrics,
    }
    (out_dir / "selected_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-family", choices=["score", "fair", "both"], default="score")
    args = parser.parse_args()

    families = ["score", "fair"] if args.split_family == "both" else [args.split_family]
    rows = [run_family(args.data_root, family, args.out_dir) for family in families]
    pd.DataFrame(rows).to_csv(args.out_dir / "all_selected_emotion_results.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
