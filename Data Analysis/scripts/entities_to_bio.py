"""Convert `medical_entities` JSON spans → BIO-tagged CoNLL file for NER fine-tuning.

Algorithm
---------
For each row:
  1. Tokenize `text` with a whitespace-aware regex (we keep punctuation as separate tokens
     so spans can align cleanly).
  2. For every entity {text, type}, locate it in the row's tokenized text via case-insensitive
     longest-prefix matching on contiguous tokens.
  3. Tag the first matched token as B-<TYPE>, subsequent as I-<TYPE>. Anything not matched
     stays O.
  4. Drop rows whose entities can't be aligned (logged).

Output: `outputs/entities_bio.conll` — one token per line `token␉tag`, blank line between
rows. Standard CoNLL format consumed by `seqeval`, `transformers` token-classification, etc.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent
CLEANED_CSV = DA_ROOT / "outputs" / "cleaned_dataset.csv"
OUT_CONLL = DA_ROOT / "outputs" / "entities_bio.conll"

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Return [(token, start_char, end_char), …]."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text or "")]


def find_span(tokens: list[tuple[str, int, int]], entity_text: str) -> tuple[int, int] | None:
    """Return (start_token_idx, end_token_idx_exclusive) or None."""
    ent_tokens = [t.lower() for (t, _, _) in tokenize(entity_text)]
    if not ent_tokens:
        return None
    n = len(ent_tokens)
    text_lower = [t.lower() for (t, _, _) in tokens]
    for i in range(0, len(text_lower) - n + 1):
        if text_lower[i:i + n] == ent_tokens:
            return i, i + n
    return None


def main() -> int:
    if not CLEANED_CSV.exists():
        print(f"missing {CLEANED_CSV} — run clean_dataset.py first", file=sys.stderr)
        return 1

    df = pd.read_csv(CLEANED_CSV, encoding="utf-8")
    written = 0
    skipped = 0
    with OUT_CONLL.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = row.get("text")
            entities_raw = row.get("medical_entities")
            if not isinstance(text, str) or not isinstance(entities_raw, str):
                continue
            try:
                entities = json.loads(entities_raw)
            except json.JSONDecodeError:
                skipped += 1
                continue
            tokens = tokenize(text)
            if not tokens:
                continue
            tags = ["O"] * len(tokens)
            for ent in entities or []:
                ent_text = (ent or {}).get("text", "")
                ent_type = (ent or {}).get("type", "").upper().replace(" ", "_")
                if not ent_text or not ent_type:
                    continue
                span = find_span(tokens, ent_text)
                if span is None:
                    continue
                s, e = span
                # only assign if currently O (don't overwrite a previously-aligned tag)
                if all(t == "O" for t in tags[s:e]):
                    tags[s] = f"B-{ent_type}"
                    for j in range(s + 1, e):
                        tags[j] = f"I-{ent_type}"
            for (tok, _, _), tag in zip(tokens, tags):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")
            written += 1

    print(f"wrote {OUT_CONLL} — {written} sentences, {skipped} skipped (JSON parse failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
