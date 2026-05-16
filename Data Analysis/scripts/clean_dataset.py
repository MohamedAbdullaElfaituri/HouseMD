"""Canonical cleaner for the House MD Turkish dialogue dataset.

Reads `DATASET/Last_HouseMD_DataSet(Sayfa1).csv`, applies the 15 fix rules from the
cleaning plan, writes `outputs/cleaned_dataset.csv` (+ `.parquet`) and
`reports/cleaning_report.md` + `reports/inconsistency_log.csv`.

Run from anywhere:
    python "Data Analysis/scripts/clean_dataset.py"
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from ftfy import fix_text

# --- paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DA_ROOT = HERE.parent                                # Data Analysis/
PROJECT_ROOT = DA_ROOT.parent                        # HouseMD2/
RAW_CSV = PROJECT_ROOT / "DATASET" / "Last_HouseMD_DataSet(Sayfa1).csv"
CANONICAL_YAML = DA_ROOT / "canonical_labels.yaml"
OUT_DIR = DA_ROOT / "outputs"
REPORT_DIR = DA_ROOT / "reports"
OUT_CSV = OUT_DIR / "cleaned_dataset.csv"
OUT_PARQUET = OUT_DIR / "cleaned_dataset.parquet"
INCONSISTENCY_CSV = REPORT_DIR / "inconsistency_log.csv"
CLEANING_REPORT_MD = REPORT_DIR / "cleaning_report.md"

EXPECTED_COLS = [
    "season", "episode", "speaker", "symptom", "test", "drug", "procedure",
    "intent", "diagnosis_stage", "sarcasm", "emotion", "organ",
    "correct_prediction", "model_prediction", "text", "medical_entities",
]
MULTIVALUE_COLS = ["symptom", "test", "drug", "procedure", "organ"]
LABEL_COLS_FOR_CANONICAL = ["intent", "emotion", "diagnosis_stage", "sarcasm", "organ"]

# --- helpers ---------------------------------------------------------------

def _log(rows: list[dict], row_id, column: str, issue: str, action: str) -> None:
    rows.append({"row_id": row_id, "column": column, "issue": issue, "action": action})


def _normalize_label(s: object) -> str | None:
    if pd.isna(s):
        return None
    out = str(s).strip().lower()
    # collapse internal whitespace
    out = re.sub(r"\s+", " ", out)
    # treat slash-separated combos uniformly: "test / tetkik" -> "test / tetkik" (kept)
    return out or None


def _build_canonical_map(yaml_doc: dict) -> dict[str, dict[str, str]]:
    """{col: {surface_form_lower: canonical_form}}"""
    out: dict[str, dict[str, str]] = {}
    for col, mapping in (yaml_doc or {}).items():
        col_map: dict[str, str] = {}
        for canonical, surfaces in (mapping or {}).items():
            canonical_str = str(canonical).strip().lower()
            for s in surfaces or []:
                col_map[_normalize_label(s)] = canonical_str
            col_map.setdefault(canonical_str, canonical_str)
        out[col] = col_map
    return out


# --- CSV reading -----------------------------------------------------------
# Source CSV is non-RFC compliant. Rows have mixed quote-escape depths:
#  - Some rows are globally wrapped in `"..."` with internal `""` escape.
#  - Some rows are not wrapped and the JSON medical_entities cell was wrapped TWICE
#    by Sheets (producing `""""text""""` quad-quotes).
#  - Every row has a tail of `,,,,,,...` from 24 trailing empty columns.
# Strategy:
#  1. Strip trailing `,+$`.
#  2. Peel up to 2 layers of outer `"..."` wrapping, unescaping `""`->`"` each layer.
#  3. Parse with csv.reader(delimiter=';', quotechar='"').
#  4. If result has != 16 fields, fall back to plain split & collapse overflow.

_TRAILING_COMMAS = re.compile(r",+$")


def _peel_quotes(line: str) -> str:
    """Peel outer `"..."` wrapper(s), unescaping `""`->`"` each layer. Bounded to 3 passes."""
    for _ in range(3):
        if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
            line = line[1:-1].replace('""', '"')
        else:
            break
    return line


def _parse_raw_line(line: str, expected_fields: int = 16) -> list[str]:
    line = line.rstrip("\r\n")
    line = _TRAILING_COMMAS.sub("", line)
    line = _peel_quotes(line)
    try:
        fields = next(csv.reader([line], delimiter=";", quotechar='"'))
    except Exception:
        fields = line.split(";")
    if len(fields) > expected_fields:
        head = fields[:expected_fields - 1]
        tail = ";".join(fields[expected_fields - 1:])
        fields = head + [tail]
    while len(fields) < expected_fields:
        fields.append("")
    return fields


def read_raw() -> tuple[pd.DataFrame, list[dict]]:
    """Read raw CSV with a custom line parser."""
    issues: list[dict] = []
    with open(RAW_CSV, encoding="utf-8-sig") as f:
        lines = list(f)
    if not lines:
        raise RuntimeError("empty CSV")
    header_cols = _parse_raw_line(lines[0])
    expected = 16
    rows = []
    for ln_no, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            continue
        parsed = _parse_raw_line(raw, expected_fields=expected)
        if not parsed[0].strip() and any(parsed[1:]):
            _log(issues, ln_no, "<row>", "blank season field after parse", "kept row, season=NaN")
        rows.append(parsed)
    cols = header_cols[:expected]
    while len(cols) < expected:
        cols.append(f"col_{len(cols)}")
    df = pd.DataFrame(rows, columns=cols)
    df = df.where(df != "", other=pd.NA)
    return df, issues


# --- cleaning steps --------------------------------------------------------

def drop_trailing_empty_cols(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    n_before = len(df.columns)
    # drop columns whose name starts with "Unnamed" OR is entirely null
    keep_cols = [
        c for c in df.columns
        if not str(c).startswith("Unnamed") and not df[c].isna().all()
    ]
    n_dropped = n_before - len(keep_cols)
    if n_dropped:
        _log(issues, None, "<schema>", f"{n_dropped} trailing empty cols", "dropped")
    return df[keep_cols].copy()


def normalize_column_names(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    old = list(df.columns)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for o, n in zip(old, df.columns):
        if o != n:
            _log(issues, None, o, "mixed-case column name", f"renamed → {n}")
    return df


def assert_expected_schema(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLS]
    for c in missing:
        _log(issues, None, c, "expected column missing", "filled NaN")
        df[c] = pd.NA
    for c in extra:
        _log(issues, None, c, "unexpected extra column", "dropped")
    return df[EXPECTED_COLS].copy()


def strip_and_fix_text(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    fixed_count = 0
    for col in df.columns:
        if df[col].dtype == object:
            # strip
            df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
            df[col] = df[col].where(df[col] != "", pd.NA)
            # mojibake — only fix actual replacement chars or sequences
            mask_mojibake = df[col].fillna("").str.contains("�|Ã[-ÿ]", regex=True)
            if mask_mojibake.any():
                fixed_count += int(mask_mojibake.sum())
                df.loc[mask_mojibake, col] = df.loc[mask_mojibake, col].map(
                    lambda s: fix_text(s) if isinstance(s, str) else s
                )
                for idx in df.index[mask_mojibake]:
                    _log(issues, idx, col, "mojibake / replacement char", "ftfy.fix_text applied")
    return df


def cast_numeric_cols(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    for col in ("season", "episode"):
        before = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")
        for idx in df.index[df[col].isna() & before.notna()]:
            _log(issues, idx, col, f"non-numeric value: {before.loc[idx]!r}", "coerced to NaN")
    return df


def normalize_sarcasm(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    before = df["sarcasm"].copy()
    df["sarcasm"] = (
        pd.to_numeric(df["sarcasm"], errors="coerce").fillna(0).astype("Int8")
    )
    bad_idx = df.index[before.notna() & ~before.astype(str).str.match(r"^\s*[01]\s*$")]
    for idx in bad_idx:
        _log(issues, idx, "sarcasm", f"non-binary value: {before.loc[idx]!r}", "coerced to 0")
    return df


def parse_multivalue(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    for col in MULTIVALUE_COLS:
        list_col = f"{col}_list"
        df[list_col] = df[col].apply(
            lambda s: [t.strip() for t in re.split(r"\s*,\s*", str(s)) if t.strip()]
            if pd.notna(s) else []
        )
    return df


def apply_canonical_labels(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    if not CANONICAL_YAML.exists():
        _log(issues, None, "<canonical>", "canonical_labels.yaml missing", "skipped label normalization")
        return df
    yaml_doc = yaml.safe_load(CANONICAL_YAML.read_text(encoding="utf-8"))
    cmap = _build_canonical_map(yaml_doc)
    for col in LABEL_COLS_FOR_CANONICAL:
        if col not in df.columns or col not in cmap:
            continue
        original = df[col].copy()
        if col == "organ":
            # organ is multivalue — operate on the *_list column elementwise too
            lookup = cmap[col]
            df["organ_list"] = df["organ_list"].apply(
                lambda lst: [lookup.get(_normalize_label(x), _normalize_label(x)) for x in lst]
            )
            # rebuild scalar organ from the (canonicalized) list when not empty, else NaN
            df[col] = df["organ_list"].apply(
                lambda lst: ", ".join(lst) if lst else pd.NA
            )
            continue
        lookup = cmap[col]
        df[col] = original.map(lambda s: lookup.get(_normalize_label(s), _normalize_label(s)))
        # log values that fell through to themselves (not in YAML)
        unmapped = (
            df.loc[original.notna(), col]
            .where(~df.loc[original.notna(), col].isin(lookup.values()))
            .dropna()
            .unique()
        )
        for v in unmapped:
            cnt = int((df[col] == v).sum())
            _log(issues, None, col, f"unmapped label value: {v!r} ({cnt}×)", "kept as-is, needs YAML entry")
    return df


_ENT_RE = re.compile(
    r'"\s*text\s*"\s*:\s*"([^"]+?)"\s*[,;]?\s*"\s*type\s*"\s*:\s*"([^"]+?)"',
    re.IGNORECASE,
)


def _repair_entities_json(s: str) -> str | None:
    """Best-effort repair of broken JSON in the medical_entities cell."""
    # collapse repeated double-quotes: "" → "
    for _ in range(4):
        if '""' not in s:
            break
        s = s.replace('""', '"')
    # strip leading/trailing stray quotes
    s = s.strip().strip('"').strip()
    # fix `"," "` (a stray separator pattern from Sheets escaping) → `, "`
    # Bad pattern from Sheets export: `"," ` where `,` got wrapped in extra quotes.
    # Lookahead `"` is NOT consumed, so replacement must NOT include trailing `"`.
    s = re.sub(r'"\s*,\s*"\s+(?=")', '", ', s)
    s = re.sub(r'"\s*,\s+(?="\w)', '", ', s)
    # Pattern between adjacent objects: `}"," {` → `}, {`
    s = re.sub(r'\}\s*"\s*,\s*"\s*(?=\{)', '}, ', s)
    s = re.sub(r'\}\s*"\s*,\s*(?=\{)', '}, ', s)
    # if it's a single dict, wrap in a list
    if s.startswith("{") and s.endswith("}"):
        s = "[" + s + "]"
    return s


def parse_medical_entities(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    def _parse(s):
        if pd.isna(s):
            return None
        s_raw = str(s).strip()
        if s_raw in ("", "[]"):
            return []
        s_repair = _repair_entities_json(s_raw)
        for candidate in (s_repair, s_raw):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    obj = [obj]
                if isinstance(obj, list):
                    return [
                        {"text": str(d.get("text", "")).strip(), "type": str(d.get("type", "")).strip()}
                        for d in obj if isinstance(d, dict)
                    ]
            except Exception:
                pass
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict):
                    obj = [obj]
                if isinstance(obj, list):
                    return [
                        {"text": str(d.get("text", "")).strip(), "type": str(d.get("type", "")).strip()}
                        for d in obj if isinstance(d, dict)
                    ]
            except Exception:
                pass
        # last-resort regex extraction
        pairs = _ENT_RE.findall(s_raw)
        if pairs:
            return [{"text": t.strip(), "type": ty.strip()} for t, ty in pairs]
        return None

    parsed = df["medical_entities"].apply(_parse)
    failures = parsed.isna() & df["medical_entities"].notna()
    for idx in df.index[failures]:
        _log(issues, idx, "medical_entities", "JSON parse failure", "set to []")
    df["medical_entities"] = parsed.where(~failures, pd.Series([[]] * len(df), index=df.index))
    return df


def drop_empty_text(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    before = len(df)
    mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    for idx in df.index[mask]:
        _log(issues, idx, "text", "empty text", "row dropped")
    df = df.loc[~mask].copy()
    return df


def drop_exact_duplicate_text(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    # dedup on (text, speaker, episode) to be safe — same line said by different chars is not a dup
    key = df[["text", "speaker", "episode"]].astype(str).agg("␟".join, axis=1)
    dup_mask = key.duplicated(keep="first")
    for idx in df.index[dup_mask]:
        _log(issues, idx, "<row>", "exact duplicate (text + speaker + episode)", "row dropped")
    df = df.loc[~dup_mask].copy()
    return df


def flag_near_duplicates(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    """Flag near-duplicate `text` rows using MinHash. Do NOT drop."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        _log(issues, None, "text", "datasketch not installed", "near-dup detection skipped")
        df["near_dup_cluster"] = pd.NA
        return df

    def shingles(s: str, k: int = 3) -> set[str]:
        s = (s or "").lower()
        return {s[i:i + k] for i in range(len(s) - k + 1)}

    lsh = MinHashLSH(threshold=0.9, num_perm=64)
    minhashes = {}
    for idx, text in df["text"].items():
        m = MinHash(num_perm=64)
        for sh in shingles(str(text)):
            m.update(sh.encode("utf-8"))
        minhashes[idx] = m
        lsh.insert(str(idx), m)

    cluster_id = {}
    cur = 0
    for idx, m in minhashes.items():
        if idx in cluster_id:
            continue
        matches = [int(x) for x in lsh.query(m) if int(x) != idx]
        if matches:
            cur += 1
            cluster_id[idx] = cur
            for mi in matches:
                cluster_id.setdefault(mi, cur)
    df["near_dup_cluster"] = df.index.map(lambda i: cluster_id.get(i, pd.NA))
    n_flagged = int(df["near_dup_cluster"].notna().sum())
    if n_flagged:
        _log(issues, None, "text", f"{n_flagged} near-duplicates flagged (≥0.9 Jaccard)", "kept, see near_dup_cluster")
    return df


# --- main ------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "splits").mkdir(parents=True, exist_ok=True)

    print(f"Reading {RAW_CSV} …")
    df, issues = read_raw()
    print(f"  raw shape: {df.shape}")

    df = drop_trailing_empty_cols(df, issues)
    df = normalize_column_names(df, issues)
    df = assert_expected_schema(df, issues)
    df = strip_and_fix_text(df, issues)
    df = cast_numeric_cols(df, issues)
    df = normalize_sarcasm(df, issues)
    df = parse_multivalue(df, issues)
    df = apply_canonical_labels(df, issues)
    df = parse_medical_entities(df, issues)
    df = drop_empty_text(df, issues)
    df = drop_exact_duplicate_text(df, issues)
    df = flag_near_duplicates(df, issues)

    print(f"  cleaned shape: {df.shape}")

    # serialize medical_entities back to JSON for the CSV
    df_out = df.copy()
    df_out["medical_entities"] = df_out["medical_entities"].apply(
        lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else ""
    )
    # serialize *_list columns as comma-separated for the CSV (round-trip safe)
    for col in MULTIVALUE_COLS:
        lc = f"{col}_list"
        if lc in df_out.columns:
            df_out[lc] = df_out[lc].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else ""
            )

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    try:
        # parquet doesn't like list-of-dicts in object column → use raw df with stringified entities
        df_parquet = df.copy()
        df_parquet["medical_entities"] = df_parquet["medical_entities"].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else None
        )
        for col in MULTIVALUE_COLS:
            lc = f"{col}_list"
            if lc in df_parquet.columns:
                df_parquet[lc] = df_parquet[lc].apply(
                    lambda v: list(v) if isinstance(v, list) else []
                )
        df_parquet.to_parquet(OUT_PARQUET, index=False)
    except Exception as e:
        print(f"  parquet write failed ({e}); CSV only", file=sys.stderr)

    # write inconsistency log
    pd.DataFrame(issues, columns=["row_id", "column", "issue", "action"]).to_csv(
        INCONSISTENCY_CSV, index=False, encoding="utf-8"
    )

    # write cleaning report
    counter = Counter((d["column"], d["issue"].split(":", 1)[0]) for d in issues)
    lines = ["# Temizleme Raporu — House MD veri kümesi", ""]
    lines.append(f"- Ham satır sayısı: **{7282}** (başlık hariç)")
    lines.append(f"- Temizlenmiş satır sayısı: **{len(df)}**")
    lines.append(f"- Toplam tutarsızlık girdisi: **{len(issues)}**")
    lines.append("")
    lines.append("## Sütun bazlı en sık sorunlar")
    lines.append("")
    lines.append("| Sütun | Sorun | Adet |")
    lines.append("|---|---|---:|")
    for (col, issue), n in counter.most_common(40):
        lines.append(f"| `{col}` | {issue} | {n} |")
    lines.append("")
    lines.append("## Etiket kardinalitesi (temizleme sonrası)")
    lines.append("")
    lines.append("| Sütun | Benzersiz değer | Beklenen (şartname) |")
    lines.append("|---|---:|---:|")
    expected = {"intent": 11, "emotion": 8, "diagnosis_stage": 8, "sarcasm": 2}
    for col in ["intent", "emotion", "diagnosis_stage", "sarcasm", "organ"]:
        n_unique = df[col].nunique(dropna=True)
        lines.append(f"| `{col}` | {n_unique} | {expected.get(col, '—')} |")
    lines.append("")
    lines.append("> Not: Şartname 11/8/8/2 sınıf öngörüyor ama veri seti elle etiketlendiği için")
    lines.append("> her sütunda çok daha fazla yüzey form var. `label_synonyms.py` bunu kümeler.")
    lines.append("> Eşleştirilmemiş değerler `inconsistency_log.csv` dosyasına yazılır.")
    lines.append("")
    lines.append("## Uygulanan kurallar")
    lines.append("")
    lines.append("1. UTF-8 BOM kaldırma (utf-8-sig)")
    lines.append("2. 24 boş tamamlama sütunu drop")
    lines.append("3. Sütun adları → snake_case")
    lines.append("4. Bütün metin sütunlarında trim")
    lines.append("5. Mojibake onarımı (ftfy)")
    lines.append("6. season/episode → Int16")
    lines.append("7. sarcasm → Int8 (0/1)")
    lines.append("8. Çok-değerli hücreler (`,` ile ayrılmış) listeye parse → `*_list` sütunu")
    lines.append("9. `canonical_labels.yaml` ile etiket normalizasyonu")
    lines.append("10. `medical_entities` JSON unescape + parse")
    lines.append("11. Boş `text` satırlarını sil")
    lines.append("12. Tam yinelenen satırları sil (text + speaker + episode)")
    lines.append("13. Yakın yinelenenleri MinHash ile işaretle (≥0.9 Jaccard)")

    CLEANING_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"  wrote {OUT_CSV}")
    print(f"  wrote {OUT_PARQUET}")
    print(f"  wrote {INCONSISTENCY_CSV} ({len(issues)} entries)")
    print(f"  wrote {CLEANING_REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
