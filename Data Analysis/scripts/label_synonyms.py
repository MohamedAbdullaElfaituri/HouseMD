"""Auto-discover synonym clusters in label columns.

For each label column (intent, emotion, diagnosis_stage, organ + symptom/test/drug/procedure
seen as bags-of-tags), enumerate distinct surface forms then group them by fuzzy similarity.
Produces:
  reports/label_values_<col>.csv     — value_counts of every distinct value
  reports/label_clusters_<col>.csv   — one row per cluster, columns: cluster_id, suggested_canonical, members, total_count

Team manually reviews `label_clusters_*.csv`, picks a canonical form per cluster, edits
`canonical_labels.yaml`, then reruns `clean_dataset.py`.

Run:
    python "Data Analysis/scripts/label_synonyms.py"
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clean_dataset import read_raw  # reuse the robust raw parser

DA_ROOT = HERE.parent
PROJECT_ROOT = DA_ROOT.parent
RAW_CSV = PROJECT_ROOT / "DATASET" / "Last_HouseMD_DataSet(Sayfa1).csv"
REPORT_DIR = DA_ROOT / "reports"

LABEL_COLS = ["intent", "emotion", "diagnosis_stage", "organ"]
BAG_COLS = ["symptom", "test", "drug", "procedure"]

CLUSTER_THRESHOLD = 85  # rapidfuzz token_set_ratio


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _enumerate_values(df: pd.DataFrame, col: str, bag: bool) -> Counter[str]:
    counter: Counter[str] = Counter()
    series = df[col].dropna().astype(str)
    if bag:
        for s in series:
            for part in re.split(r"\s*,\s*", s):
                p = _normalize(part)
                if p:
                    counter[p] += 1
    else:
        for s in series:
            p = _normalize(s)
            if p:
                counter[p] += 1
    return counter


def _cluster_values(values: list[str], threshold: int = CLUSTER_THRESHOLD) -> list[set[str]]:
    """Greedy clustering: for each value, attach to the first existing cluster with any
    member above the similarity threshold; otherwise start a new cluster."""
    clusters: list[set[str]] = []
    representatives: list[str] = []
    for v in values:
        attached = False
        if representatives:
            # find best-matching representative
            best = rf_process.extractOne(v, representatives, scorer=fuzz.token_set_ratio)
            if best and best[1] >= threshold:
                idx = best[2]
                clusters[idx].add(v)
                attached = True
        if not attached:
            clusters.append({v})
            representatives.append(v)
    return clusters


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = read_raw()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    for col in LABEL_COLS + BAG_COLS:
        if col not in df.columns:
            print(f"skip {col} — not in df")
            continue
        bag = col in BAG_COLS or col == "organ"
        counter = _enumerate_values(df, col, bag=bag)
        if not counter:
            continue

        # value_counts dump
        values_df = pd.DataFrame(
            counter.most_common(), columns=["value", "count"]
        )
        values_df.to_csv(REPORT_DIR / f"label_values_{col}.csv", index=False, encoding="utf-8")

        # cluster
        sorted_vals = [v for v, _ in counter.most_common()]
        clusters = _cluster_values(sorted_vals)

        rows = []
        for cid, members in enumerate(clusters, start=1):
            members_sorted = sorted(members, key=lambda v: -counter[v])
            suggested = members_sorted[0]  # most frequent surface form
            total = sum(counter[m] for m in members_sorted)
            rows.append({
                "cluster_id": cid,
                "suggested_canonical": suggested,
                "n_members": len(members_sorted),
                "total_count": total,
                "members": " | ".join(members_sorted),
            })
        clusters_df = pd.DataFrame(rows).sort_values("total_count", ascending=False)
        clusters_df.to_csv(REPORT_DIR / f"label_clusters_{col}.csv", index=False, encoding="utf-8")
        print(
            f"{col}: {len(counter)} distinct values → {len(clusters)} clusters "
            f"(top cluster size: {clusters_df['n_members'].iloc[0]})"
        )

    print(f"\nDone. Review {REPORT_DIR}/label_clusters_*.csv and edit canonical_labels.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
