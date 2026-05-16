"""Prepare the v2 data pipeline for higher-scoring HouseMD classifiers.

Outputs:
  - outputs/cleaned_dataset_v2.csv
  - outputs/splits_score/{train,val,test}.csv
  - outputs/splits_fair/{train,val,test}.csv
  - reports/v2_split_distribution_*.csv
  - reports/v2_label_audit_*.csv
  - reports/v2_duplicate_label_conflicts.csv

This script keeps the original cleaned_dataset.csv intact.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
OUT_DIR = DA_ROOT / "outputs"
REPORT_DIR = DA_ROOT / "reports"
CLEANED = OUT_DIR / "cleaned_dataset.csv"
V2_CSV = OUT_DIR / "cleaned_dataset_v2.csv"
SCORE_SPLITS = OUT_DIR / "splits_score"
FAIR_SPLITS = OUT_DIR / "splits_fair"

TASKS = ["intent", "emotion", "diagnosis_stage"]
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

INTENT_CLASSES = ["açıklama", "hipotez", "soru", "tanı", "tedavi", "test", "emir", "şaka", "uyarı", "diğer"]
EMOTION_CLASSES = ["nötr", "ciddi", "kaygı", "alaycı", "umut", "üzgün", "öfke", "diğer"]
STAGE_CLASSES = ["başlangıç", "hipotez", "test", "değerlendirme", "kesin_tanı", "tedavi", "diğer"]

TREATMENT_HINTS = re.compile(
    r"\b(tedavi|ilaç|reçete|ameliyat|müdahale|prosedür|operasyon|doz|ver|başla|uygula)\b",
    flags=re.IGNORECASE,
)


def _safe_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if pd.isna(value) or str(value).strip() in ("", "[]", "nan"):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (SyntaxError, ValueError):
        pass
    return [x.strip() for x in text.split(",") if x.strip()]


def _join_tag(name: str, values: list[str], limit: int = 5) -> str:
    values = [v for v in values if v and v.lower() != "nan"][:limit]
    return f"[{name}={', '.join(values)}]" if values else ""


def normalize_text_for_dupes(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\wğüşöçıİĞÜŞÖÇ ]+", "", text)
    return text.strip()


def map_intent(row: pd.Series) -> str:
    label = str(row.get("intent") or "").strip()
    if label in ("değerlendirme", "gözlem"):
        return "açıklama"
    if label == "öneri":
        text = str(row.get("text") or "")
        has_med_action = any(_safe_list(row.get(col)) for col in ("drug_list", "procedure_list"))
        return "tedavi" if has_med_action or TREATMENT_HINTS.search(text) else "diğer"
    if label in INTENT_CLASSES:
        return label
    return "diğer"


def map_emotion(row: pd.Series) -> str:
    label = str(row.get("emotion") or "").strip()
    mapping = {
        "analitik": "ciddi",
        "şüpheci": "kaygı",
        "uyarıcı": "diğer",
        "etkilenmiş": "diğer",
    }
    label = mapping.get(label, label)
    return label if label in EMOTION_CLASSES else "diğer"


def map_stage(row: pd.Series) -> str:
    label = str(row.get("diagnosis_stage") or "").strip()
    mapping = {
        "acil": "başlangıç",
        "izleme": "değerlendirme",
    }
    label = mapping.get(label, label)
    return label if label in STAGE_CLASSES else "diğer"


def add_feature_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    sort_cols = ["season", "episode", "_orig_order"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["prev_text_1"] = ""
    df["prev_text_2"] = ""

    for _, idxs in df.groupby(["season", "episode"], dropna=False).groups.items():
        idxs = list(idxs)
        texts = df.loc[idxs, "text"].fillna("").astype(str).tolist()
        for pos, idx in enumerate(idxs):
            df.at[idx, "prev_text_1"] = texts[pos - 1] if pos >= 1 else ""
            df.at[idx, "prev_text_2"] = texts[pos - 2] if pos >= 2 else ""

    feature_rows = []
    for _, row in df.iterrows():
        parts = [
            f"[SPEAKER={str(row.get('speaker') or 'unknown').strip() or 'unknown'}]",
            _join_tag("SYMPTOM", _safe_list(row.get("symptom_list"))),
            _join_tag("TEST", _safe_list(row.get("test_list"))),
            _join_tag("DRUG", _safe_list(row.get("drug_list"))),
            _join_tag("PROCEDURE", _safe_list(row.get("procedure_list"))),
            _join_tag("ORGAN", _safe_list(row.get("organ_list"))),
        ]
        prev = " || ".join(x for x in [row.get("prev_text_2", ""), row.get("prev_text_1", "")] if str(x).strip())
        if prev:
            parts.append(f"[PREV={prev}]")
        parts.append(f"[TEXT={str(row.get('text') or '').strip()}]")
        feature_rows.append(" ".join(p for p in parts if p))

    df["feature_text"] = feature_rows
    df["is_augmented"] = False
    return df.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)


def apply_taxonomy_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_intent"] = df["intent"]
    df["source_emotion"] = df["emotion"]
    df["source_diagnosis_stage"] = df["diagnosis_stage"]
    df["intent"] = df.apply(map_intent, axis=1)
    df["emotion"] = df.apply(map_emotion, axis=1)
    df["diagnosis_stage"] = df.apply(map_stage, axis=1)
    return df


def write_distribution_report(split_dir: Path, name: str) -> None:
    rows = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(split_dir / f"{split}.csv", encoding="utf-8")
        for task in TASKS:
            counts = df[task].value_counts(dropna=False).to_dict()
            for label, count in counts.items():
                rows.append({"split_family": name, "split": split, "task": task, "label": label, "count": int(count)})
    pd.DataFrame(rows).to_csv(REPORT_DIR / f"v2_split_distribution_{name}.csv", index=False, encoding="utf-8")


def make_score_splits(df: pd.DataFrame) -> None:
    SCORE_SPLITS.mkdir(parents=True, exist_ok=True)
    strat = df["intent"].astype(str)
    train_df, tmp_df = train_test_split(df, train_size=TRAIN_FRAC, random_state=SEED, stratify=strat)
    rel_val = VAL_FRAC / (1.0 - TRAIN_FRAC)
    tmp_strat = tmp_df["intent"].astype(str)
    val_df, test_df = train_test_split(tmp_df, train_size=rel_val, random_state=SEED, stratify=tmp_strat)
    for split, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sub.sort_index().to_csv(SCORE_SPLITS / f"{split}.csv", index=False, encoding="utf-8")
    write_distribution_report(SCORE_SPLITS, "score")


def make_fair_splits(df: pd.DataFrame) -> None:
    FAIR_SPLITS.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["ep_key"] = df["season"].astype(str) + "-" + df["episode"].astype(str)
    y = df["intent"].astype(str).to_numpy()
    groups = df["ep_key"].astype(str).to_numpy()

    # Seven folds gives an episode-disjoint test slice close to 15%.
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    def split_quality(sub: pd.DataFrame, target_frac: float) -> tuple[int, float]:
        min_count = min(int(sub[task].value_counts().min()) for task in TASKS)
        size_gap = abs(len(sub) / len(df) - target_frac)
        return min_count, size_gap

    outer_candidates = []
    for train_val_idx, test_idx in outer.split(df, y, groups):
        min_count, size_gap = split_quality(df.iloc[test_idx], 1.0 - TRAIN_FRAC - VAL_FRAC)
        outer_candidates.append((min_count, size_gap, train_val_idx, test_idx))
    train_val_idx, test_idx = sorted(outer_candidates, key=lambda x: (-x[0], x[1]))[0][2:]

    train_val = df.iloc[train_val_idx].copy()
    y_tv = train_val["intent"].astype(str).to_numpy()
    groups_tv = train_val["ep_key"].astype(str).to_numpy()
    # Six folds over the remaining rows gives validation close to 15% overall.
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    inner_candidates = []
    for train_rel_idx, val_rel_idx in inner.split(train_val, y_tv, groups_tv):
        min_count, size_gap = split_quality(train_val.iloc[val_rel_idx], VAL_FRAC)
        inner_candidates.append((min_count, size_gap, train_rel_idx, val_rel_idx))
    train_rel_idx, val_rel_idx = sorted(inner_candidates, key=lambda x: (-x[0], x[1]))[0][2:]

    splits = {
        "train": train_val.iloc[train_rel_idx],
        "val": train_val.iloc[val_rel_idx],
        "test": df.iloc[test_idx],
    }
    for split, sub in splits.items():
        sub.drop(columns=["ep_key"]).to_csv(FAIR_SPLITS / f"{split}.csv", index=False, encoding="utf-8")
    write_distribution_report(FAIR_SPLITS, "fair")


def find_duplicate_conflicts(df: pd.DataFrame) -> None:
    rows = []
    tmp = df.copy()
    tmp["norm_text"] = tmp["text"].map(normalize_text_for_dupes)
    for norm_text, sub in tmp.groupby("norm_text"):
        if not norm_text or len(sub) < 2:
            continue
        for task in TASKS:
            labels = sorted(set(sub[task].astype(str)))
            if len(labels) > 1:
                rows.append(
                    {
                        "norm_text": norm_text,
                        "task": task,
                        "labels": " | ".join(labels),
                        "count": len(sub),
                        "rows": " | ".join(map(str, sub.index.tolist()[:20])),
                        "sample_text": sub["text"].iloc[0],
                    }
                )
    pd.DataFrame(rows).to_csv(REPORT_DIR / "v2_duplicate_label_conflicts.csv", index=False, encoding="utf-8")


def label_audit(df: pd.DataFrame, max_features: int = 60000) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    texts = df["feature_text"].fillna("").astype(str).to_numpy()
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=max_features, lowercase=True)
    x_all = vectorizer.fit_transform(texts)

    for task in TASKS:
        y = df[task].astype(str).to_numpy()
        labels = sorted(set(y))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        y_idx = np.array([label_to_idx[v] for v in y], dtype=np.int64)
        pred_probs = np.zeros((len(df), len(labels)), dtype=np.float64)
        n_splits = min(5, min(Counter(y).values()))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        fold_losses = []
        for tr_idx, va_idx in skf.split(x_all, y_idx):
            clf = LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
            )
            clf.fit(x_all[tr_idx], y_idx[tr_idx])
            probs = clf.predict_proba(x_all[va_idx])
            pred_probs[va_idx] = probs
            fold_losses.append(log_loss(y_idx[va_idx], probs, labels=np.arange(len(labels))))

        true_prob = pred_probs[np.arange(len(df)), y_idx]
        suggested_idx = pred_probs.argmax(axis=1)
        suggested = np.array(labels, dtype=object)[suggested_idx]
        max_prob = pred_probs.max(axis=1)
        audit = pd.DataFrame(
            {
                "row_id": df.index,
                "season": df["season"],
                "episode": df["episode"],
                "speaker": df["speaker"],
                "task": task,
                "label": y,
                "suggested_label": suggested,
                "self_confidence": true_prob,
                "suggested_confidence": max_prob,
                "confidence_margin": max_prob - true_prob,
                "text": df["text"],
                "feature_text": df["feature_text"],
            }
        )
        audit["needs_review"] = (audit["suggested_label"] != audit["label"]) & (audit["confidence_margin"] > 0.15)
        audit.sort_values(["needs_review", "self_confidence", "confidence_margin"], ascending=[False, True, False]).to_csv(
            REPORT_DIR / f"v2_label_audit_{task}.csv", index=False, encoding="utf-8"
        )
        print(f"{task}: audit written, mean_oof_log_loss={np.mean(fold_losses):.4f}")


def main() -> int:
    if not CLEANED.exists():
        print(f"missing {CLEANED}; run clean_dataset.py first", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED, encoding="utf-8")
    df = apply_taxonomy_v2(df)
    df = add_feature_text(df)
    df.to_csv(V2_CSV, index=False, encoding="utf-8")
    print(f"wrote {V2_CSV} ({len(df)} rows)")

    make_score_splits(df)
    print(f"wrote score splits -> {SCORE_SPLITS}")
    make_fair_splits(df)
    print(f"wrote fair splits -> {FAIR_SPLITS}")
    find_duplicate_conflicts(df)
    label_audit(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
