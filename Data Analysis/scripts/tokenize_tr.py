"""Turkish tokenization helpers.

For EDA only (length stats, type/token ratio). For model input use
the BERTurk tokenizer (`transformers.AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')`).
"""

from __future__ import annotations

import re

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})


def tr_lower(s: str) -> str:
    """Turkish-aware lowercase. Handles dotted/dotless İ/I correctly."""
    return s.translate(_TR_LOWER).lower()


_WORD = re.compile(r"[A-Za-zÇĞİıÖŞÜçğıöşü']+", re.UNICODE)


def words(s: str) -> list[str]:
    return _WORD.findall(s or "")


def word_count(s: str) -> int:
    return len(words(s))


def char_count(s: str) -> int:
    return len(s or "")
