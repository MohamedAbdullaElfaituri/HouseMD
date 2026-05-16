"""Helpers for parsing comma-separated multi-value cells.

Used by both `clean_dataset.py` and notebook 03/04 for unified tokenization.
"""

from __future__ import annotations

import re
from typing import Iterable


_DELIM = re.compile(r"\s*,\s*")


def split_multivalue(s: object) -> list[str]:
    """Split a CSV cell with comma-separated values, trimmed and de-empty'd."""
    if s is None or (isinstance(s, float) and s != s):  # NaN check
        return []
    return [t.strip() for t in _DELIM.split(str(s)) if t and t.strip()]


def explode_multivalue(df, columns: Iterable[str]):
    """Yield (row_idx, column, value) triples for every value in multi-value cells.
    Useful for building long-format frames for co-occurrence analysis."""
    for idx, row in df.iterrows():
        for col in columns:
            for v in split_multivalue(row.get(col)):
                yield idx, col, v
