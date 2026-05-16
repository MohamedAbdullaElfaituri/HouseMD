"""TF-IDF + LinearSVC baseline, one model per task. Lower bound benchmark.

Usage:
    python Model/baselines/tfidf_svc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SPLITS = PROJECT_ROOT / "Data Analysis" / "outputs" / "splits"
OUT = HERE.parent / "runs" / "tfidf_svc"
TASKS = ["intent", "emotion", "diagnosis_stage", "sarcasm"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SPLITS / "train.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)

    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000, lowercase=True)
    X_train = vec.fit_transform(train["text"])
    X_test = vec.transform(test["text"])

    results: dict[str, dict] = {}
    for task in TASKS:
        y_train = train[task].astype(str)
        y_test = test[task].astype(str)
        mask_tr = y_train.notna() & (y_train != "nan")
        mask_te = y_test.notna() & (y_test != "nan")
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(X_train[mask_tr.values], y_train[mask_tr])
        pred = clf.predict(X_test[mask_te.values])
        f1 = f1_score(y_test[mask_te], pred, average="macro", zero_division=0)
        rep = classification_report(y_test[mask_te], pred, output_dict=True, zero_division=0)
        results[task] = {"f1_macro": float(f1), "n_test": int(mask_te.sum())}
        print(f"{task}: macro-F1 = {f1:.4f}  (n_test={int(mask_te.sum())})")
        (OUT / f"report_{task}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    overall = float(np.mean([r["f1_macro"] for r in results.values()]))
    results["overall_avg_f1_macro"] = overall
    print(f"\nOVERALL avg macro-F1: {overall:.4f}")
    (OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
