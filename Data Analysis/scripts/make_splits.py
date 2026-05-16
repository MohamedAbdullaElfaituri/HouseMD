"""Episode-disjoint train/val/test splits.

We never split within an episode — preventing dialogue leakage between train and test.
Within each split we stratify by `intent` so per-class distribution is roughly preserved.

Output: outputs/splits/{train,val,test}.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
CLEANED_CSV = DA_ROOT / "outputs" / "cleaned_dataset.csv"
SPLITS_DIR = DA_ROOT / "outputs" / "splits"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test gets the remainder


def main() -> int:
    if not CLEANED_CSV.exists():
        print(f"missing {CLEANED_CSV} — run clean_dataset.py first", file=sys.stderr)
        return 1
    df = pd.read_csv(CLEANED_CSV, encoding="utf-8")

    # episode-disjoint: shuffle (season, episode) tuples once, split groups
    df["ep_key"] = df["season"].astype(str) + "-" + df["episode"].astype(str)
    rng = np.random.default_rng(SEED)
    episodes = sorted(df["ep_key"].dropna().unique())
    rng.shuffle(episodes)

    n = len(episodes)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_eps = set(episodes[:n_train])
    val_eps = set(episodes[n_train:n_train + n_val])
    test_eps = set(episodes[n_train + n_val:])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, eps in (("train", train_eps), ("val", val_eps), ("test", test_eps)):
        sub = df[df["ep_key"].isin(eps)].drop(columns=["ep_key"])
        out = SPLITS_DIR / f"{name}.csv"
        sub.to_csv(out, index=False, encoding="utf-8")
        print(f"{name}: {len(eps)} episodes, {len(sub)} rows → {out}")

    # safety assertion
    assert train_eps.isdisjoint(val_eps) and train_eps.isdisjoint(test_eps) and val_eps.isdisjoint(test_eps)
    print("OK: splits are episode-disjoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
