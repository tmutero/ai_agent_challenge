"""
Generic, structure-aware ingestion for user-uploaded files.

    CSV / TSV / XLSX       -> one table dataset per file (per sheet for Excel)
    JSON                   -> the main list of records becomes a table; nested lists
                              of objects become child tables; top-level scalars are
                              kept as document context
    PDF                    -> every table (stitched across page breaks) + prose paragraphs
    DOCX                   -> every table + prose paragraphs
    TXT / MD               -> prose paragraphs (markdown headings become sections)

Each file is turned into

  1. ``Document`` objects for the vector pipeline -- one per *record* (table row,
     JSON object, prose paragraph), never a blind character split across rows, and
  2. pandas ``DataFrame``s over the same records, for aggregate questions
     ("cheapest", "average", "how many") that top-k vector search cannot answer.

Nothing here knows about any particular dataset: key columns, display titles and
filterable columns are inferred from the data. Every record carries the same
provenance in both views -- ``source`` file, ``record_ref`` (``row 3``,
``Table 2, row 5``, ``paragraph 4``) and the ``chunk_id`` the vector store assigns.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pandas as pd

from rag_core.pipeline.types import Document

# Folder loaded at start-up; uploads are added on top. Point RAG_DATA_DIR at your own documents.
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR") or Path(__file__).parent.parent / "sample_data")
SUPPORTED = (".csv", ".tsv", ".xlsx", ".json", ".pdf", ".docx", ".txt", ".md")
MAX_FILE_BYTES = 20 * 1024 * 1024
MISSING = "not reported"
MAX_FILTER_VALUES = 30          # text columns with at most this many distinct values are filterable
MAX_PARAGRAPH_CHARS = 1500      # longer prose blocks are split on sentence boundaries

# Readable labels embed (and read) better than snake_case column names.
_TOKENS = {"pct": "%", "num": "number", "no": "number", "qty": "quantity", "amt": "amount",
           "min": "minimum", "max": "maximum", "avg": "average", "id": "ID"}
_CURRENCIES = {"usd", "eur", "gbp", "zar", "jpy", "cny", "inr", "aud", "cad", "chf", "nzd", "brl", "kes", "ngn",
               "bwp", "nad", "sgd", "hkd", "aed", "sek", "nok", "dkk", "mxn"}


def field_key(column: str) -> str:
    """Payload-safe key for a column ("Client risk profile" -> "client_risk_profile")."""
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") or "col"


def _phrase(column: str) -> str:
    """A label for use mid-sentence: 'Minimum lump sum (ZAR)' -> 'minimum lump sum (ZAR)', 'TER %' unchanged."""
    text = label(column)
    return text[:1].lower() + text[1:] if text[1:2].islower() else text


def label(column: str) -> str:
    """'return_5y_pct' -> 'Return 5-year %', 'min_lump_sum_zar' -> 'Minimum lump sum (ZAR)'; already-readable names unchanged."""
    col = str(column)
    if " " in col or not re.search(r"[_a-z]", col):
        return col
    words = [re.sub(r"^(\d+)y$", r"\1-year", _TOKENS.get(w.lower(), w)) for w in re.split(r"[_\-.]+", col) if w]
    if len(words) > 1 and words[-1].lower() in _CURRENCIES:   # trailing currency code is a unit: "price_usd"
        words[-1] = f"({words[-1].upper()})"
    text = " ".join(words)
    return text[:1].upper() + text[1:]


# --- model ---------------------------------------------------------------------------

@dataclass
class SourceFile:
    name: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class Dataset:
    """One logical table, available to both the retriever and the aggregator."""

    name: str              # id used by the query planner, e.g. "funds" or "investment_policy_t2"
    source: str            # file the rows came from
    description: str
    frame: pd.DataFrame    # includes _source / _record_ref / _chunk_id columns
    key_column: str
    filterable: dict[str, list[str]] = field(default_factory=dict)  # col -> allowed values
    category: str = ""     # payload category the rows are indexed under (child tables share the parent's)
    title_column: str | None = None   # human-readable name column, if one was found

    @property
    def numeric_columns(self) -> list[str]:
        """Columns where nearly every reported value is a number (after stripping % and thousands separators)."""
        out = []
        for c in self.frame.columns:
            if str(c).startswith("_") or c == self.key_column:
                continue
            reported = self.frame[c][~self.frame[c].map(_blank)]
            if len(reported) and to_number(reported).notna().mean() >= 0.8:
                out.append(c)
        return out


@dataclass
class TextSource:
    name: str
    source: str
    description: str
    paragraphs: int


@dataclass
class LoadReport:
    source: str
    ok: bool
    records: int = 0
    datasets: list[str] = field(default_factory=list)
    text_paragraphs: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class KnowledgeBase:
    documents: list[Document]
    datasets: dict[str, Dataset]
    text_sources: dict[str, TextSource] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return sorted({d.source for d in self.documents})

    @property
    def empty(self) -> bool:
        return not self.documents

    def catalog(self) -> str:
        """Compact description of every dataset for the query planner's prompt."""
        lines = []
        for d in self.datasets.values():
            cols = [c for c in d.frame.columns if not str(c).startswith("_")]
            lines.append(f"- {d.name} ({d.source}): {d.description}. Columns: {', '.join(map(str, cols))}")
            for col, values in d.filterable.items():
                lines.append(f"    {col} allowed values: {values}")
        for t in self.text_sources.values():
            lines.append(f"- {t.name} ({t.source}): {t.description}")
        return "\n".join(lines) or "(empty)"

    def example_questions(self, limit: int = 6) -> list[str]:
        """Starter questions built from whatever is loaded. Cycles through question
        kinds (lookup, ranking, filter, summary) so the list shows what the system can do."""
        lookups, rankings, filters = [], [], []
        for d in self.datasets.values():
            if d.category != d.name or d.frame.empty:   # nested child tables are covered by their parent
                continue
            numeric = d.numeric_columns
            if d.title_column and numeric:
                lookups.append(f"What is the {_phrase(numeric[0])} of {_fmt(d.frame[d.title_column].iloc[0])}?")
            if numeric:
                rankings.append(f"In {d.source}, which entries have the lowest {_phrase(numeric[-1])}?")
            # A filter example needs a column whose values repeat (not an ID or a name).
            for col, values in d.filterable.items():
                if col not in (d.key_column, d.title_column) and d.frame[col].nunique() < len(d.frame):
                    filters.append(f"Which entries in {d.source} have {_phrase(col)} '{values[0]}'?")
                    break
        summaries = [f"Summarise the main points of {t.source}" for t in self.text_sources.values()]
        ordered = [q for group in zip_longest(lookups, rankings, filters, summaries) for q in group if q]
        return ordered[:limit - 1] + ["What documents and datasets are loaded?"]


# --- shared record helpers ---------------------------------------------------------------

def _chunk_id(doc_id: str) -> str:
    # Matches AtomicChunker's id scheme: one chunk per document, index 0.
    return f"{doc_id}::0"


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or (isinstance(value, str) and not value.strip())


def to_number(series: pd.Series) -> pd.Series:
    """'1,250' -> 1250, '0.55%' -> 0.55; anything else -> NaN."""
    text = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce")


def _fmt(value: Any) -> str:
    if _blank(value):
        return MISSING
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "doc"


def _unique_name(base: str, taken: set[str]) -> str:
    name, n = base, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    taken.add(name)
    return name


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() or f"column_{i + 1}" for i, c in enumerate(df.columns)]
    for c in df.columns:
        if _is_text(df[c]):
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df.reset_index(drop=True)


def _is_text(series: pd.Series) -> bool:
    # pandas 3 infers a dedicated string dtype; older versions use object.
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _infer_key(df: pd.DataFrame) -> str | None:
    """First column whose values are unique and present in every row (prefer text)."""
    candidates = [c for c in df.columns if df[c].notna().all() and df[c].astype(str).is_unique]
    text = [c for c in candidates if _is_text(df[c])]
    return (text or candidates or [None])[0]


def _infer_title(df: pd.DataFrame, key: str | None) -> str | None:
    """A human-readable name column: mostly-unique text, not too long (e.g. fund_name)."""
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if _is_text(df[c]) and len(s) and s.nunique() >= 0.9 * len(df) and s.str.len().mean() <= 60 \
                and not s.str.fullmatch(r"[A-Z0-9\-_]+").all():
            return c
    return key


def _filterable(df: pd.DataFrame) -> dict[str, list[str]]:
    out = {}
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if _is_text(df[c]) and 1 < s.nunique() <= MAX_FILTER_VALUES and s.str.len().mean() <= 60:
            out[c] = sorted(s.unique().tolist())
    return out


def table_dataset(df: pd.DataFrame, *, source: str, name: str, caption: str, ref_prefix: str,
                  doc_type: str, doc_prefix: str = "", context: str = "",
                  skip_blank: bool = False) -> tuple[list[Document], Dataset]:
    """Any DataFrame -> one Document per row + a Dataset with provenance columns.
    ``skip_blank`` omits empty fields from the text (JSON records with optional
    keys); otherwise an empty cell is stated explicitly as "not reported"."""
    df = _clean_frame(df)
    key = _infer_key(df)
    title_col = _infer_title(df, key)
    docs, refs, ids, seen = [], [], [], set()
    for i, row in enumerate(df.to_dict("records"), start=1):
        ref = f"{ref_prefix}row {i}"
        raw_key = _fmt(row[key]) if key else f"row{i}"
        doc_id = f"{source}#{doc_prefix}{raw_key}"
        if doc_id in seen:
            doc_id = f"{doc_id}~{i}"
        seen.add(doc_id)
        prefix = f"{caption} - " if caption and caption not in ref else ""
        heading = f"{source} - {prefix}{ref}" + (f": {_fmt(row[title_col])}" if title_col else "")
        body = "\n".join(f"{label(k)}: {_fmt(v)}" for k, v in row.items() if not (skip_blank and _blank(v)))
        docs.append(Document(
            doc_id=doc_id, text="\n".join(x for x in (heading, context, body) if x), source=source, doc_type=doc_type,
            metadata={"category": name, "section": ref, "record_ref": ref,
                      "fields": {field_key(k): _fmt(v) for k, v in row.items()}}))
        refs.append(ref)
        ids.append(doc_id)
    frame = df.copy()
    frame["_source"], frame["_record_ref"], frame["_chunk_id"] = source, refs, [_chunk_id(d) for d in ids]
    desc = f"{caption + ': ' if caption else ''}table with {len(df)} rows" + (f", one per {label(key)}" if key else "")
    return docs, Dataset(name, source, desc, frame, key or df.columns[0], _filterable(df), name, title_col)


def text_documents(paragraphs: list[str], *, source: str, name: str, doc_type: str,
                   sections: list[str | None] | None = None) -> list[Document]:
    docs, k = [], 0
    for idx, para in enumerate(paragraphs):
        for piece in _split_long(para):
            k += 1
            sec = sections[idx] if sections else None
            ref = f"paragraph {k}" + (f" ({sec})" if sec else "")
            docs.append(Document(doc_id=f"{source}#p{k}", text=piece, source=source, doc_type=doc_type,
                                 metadata={"category": name, "section": ref, "record_ref": ref}))
    return docs


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_PARAGRAPH_CHARS:
        return [text]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(sentence) > MAX_PARAGRAPH_CHARS:
            out.append(cur.strip())
            cur = ""
        cur += sentence + " "
    return out + ([cur.strip()] if cur.strip() else [])


# --- per-format loaders ------------------------------------------------------------------
# Each returns (documents, datasets, text_sources, warnings).

def _load_delimited(f: SourceFile, stem: str, taken: set[str]):
    text = _decode(f.data)
    sep = "\t" if f.name.lower().endswith(".tsv") else None
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=None)
    docs, ds = table_dataset(df, source=f.name, name=_unique_name(stem, taken), caption="", ref_prefix="",
                             doc_type="csv_row")
    return docs, [ds], [], []


def _load_excel(f: SourceFile, stem: str, taken: set[str]):
    sheets = pd.read_excel(io.BytesIO(f.data), sheet_name=None)
    docs, datasets, warnings = [], [], []
    for sheet, df in sheets.items():
        if df.dropna(how="all").empty:
            warnings.append(f"sheet '{sheet}' is empty - skipped")
            continue
        many = len(sheets) > 1
        d, ds = table_dataset(df, source=f.name, name=_unique_name(f"{stem}_{_slug(sheet)}" if many else stem, taken),
                              caption=f"sheet {sheet}" if many else "", ref_prefix=f"{sheet}, " if many else "",
                              doc_type="xlsx_row", doc_prefix=f"{_slug(sheet)}-" if many else "")
        docs += d
        datasets.append(ds)
    return docs, datasets, [], warnings


def _scalar(v: Any) -> bool:
    return not isinstance(v, (dict, list))


def _flatten(obj: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, list) and all(_scalar(x) for x in v):
            out[key] = "; ".join(map(str, v))
        else:
            out[key] = v
    return out


def _record_lists(obj: Any, path: str = "") -> list[tuple[str, list[dict]]]:
    """All lists of objects in a JSON document, shallowest first."""
    found = []
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        found.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += _record_lists(v, f"{path}.{k}" if path else k)
    return found


def _load_json(f: SourceFile, stem: str, taken: set[str]):
    data = json.loads(_decode(f.data))
    lists = _record_lists(data)
    if not lists:   # no records: treat the document as text, one line per leaf value
        flat = _flatten(data) if isinstance(data, dict) else {"value": data}
        paras = [f"{label(k)}: {_fmt(v)}" for k, v in flat.items()]
        name = _unique_name(f"{stem}_text", taken)
        return text_documents(["\n".join(paras)], source=f.name, name=name, doc_type="json_text"), [], \
            [TextSource(name, f.name, "JSON document (no record lists)", 1)], []

    path, records = max(lists, key=lambda t: (-t[0].count("."), len(t[1])))   # shallowest, then longest
    context = ""
    if isinstance(data, dict):
        ctx = {k: v for k, v in data.items() if _scalar(v)}
        context = "; ".join(f"{label(k)}: {v}" for k, v in ctx.items())

    parent_name = _unique_name(stem, taken)
    rows, children = [], {}
    for rec in records:
        flat = {}
        for k, v in rec.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                children.setdefault(k, []).append(v)
                flat[k] = " | ".join(", ".join(f"{label(a)}: {_fmt(b)}" for a, b in _flatten(x).items() if not _blank(b))
                                     for x in v)
            elif isinstance(v, dict):
                flat.update(_flatten(v, f"{k}."))
            elif isinstance(v, list):
                flat[k] = "; ".join(map(str, v))
            else:
                flat[k] = v
        rows.append(flat)
    df = pd.DataFrame(rows)
    docs, parent = table_dataset(df, source=f.name, name=parent_name, caption=path, ref_prefix=f"{path}, ",
                                 doc_type="json_record", context=context, skip_blank=True)
    datasets = [parent]
    # Nested lists of objects (e.g. fee tiers) also become a flat child table for maths;
    # their rows cite the parent record's chunk, where the whole list is kept together.
    for child, per_parent in children.items():
        crow, refs, cids = [], [], []
        for (_, prow), items in zip(parent.frame.iterrows(), per_parent):
            for item in items:
                crow.append({parent.key_column: prow[parent.key_column], **_flatten(item)})
                refs.append(prow["_record_ref"])
                cids.append(prow["_chunk_id"])
        cdf = _clean_frame(pd.DataFrame(crow))
        cdf["_source"], cdf["_record_ref"], cdf["_chunk_id"] = f.name, refs, cids
        datasets.append(Dataset(_unique_name(f"{parent_name}_{_slug(child)}", taken), f.name,
                                f"'{child}' entries nested in each {path} record ({len(cdf)} rows)", cdf,
                                parent.key_column, {}, parent_name))
    return docs, datasets, [], []


def _paragraphs_from_lines(lines: list[dict]) -> list[str]:
    """Group PDF text lines into paragraphs: a vertical gap larger than half a line
    height starts a new paragraph; wrapped lines are re-joined."""
    paras: list[list[str]] = []
    prev_bottom = None
    for ln in lines:
        height = ln["bottom"] - ln["top"]
        if prev_bottom is None or ln["top"] - prev_bottom > 0.5 * height:
            paras.append([])
        paras[-1].append(ln["text"].strip())
        prev_bottom = ln["bottom"]
    return [" ".join(p) for p in paras]


def _table_rows_to_frame(rows: list[list[str]]) -> pd.DataFrame:
    header = [h or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=header)


def _load_pdf(f: SourceFile, stem: str, taken: set[str]):
    """Tables are stitched when a page starts with a table that has the same column
    count as the table ending the previous page (a repeated header row is dropped).
    Captions come from the text line just above a table."""
    import pdfplumber

    tables: list[dict] = []        # {rows, caption, page, bottom_ratio}
    prose: list[str] = []
    with pdfplumber.open(io.BytesIO(f.data)) as pdf:
        for pno, page in enumerate(pdf.pages):
            found = page.find_tables()
            lines = page.extract_text_lines()
            for ti, t in enumerate(found):
                rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in t.extract()]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2 and not tables:
                    continue
                prev = tables[-1] if tables else None
                at_top = t.bbox[1] < 0.2 * page.height
                if prev and ti == 0 and at_top and prev["page"] == pno - 1 and prev["bottom_ratio"] > 0.6 \
                        and len(rows[0]) == len(prev["rows"][0]):
                    prev["rows"].extend(rows[1:] if rows[0] == prev["rows"][0] else rows)
                    prev["page"], prev["bottom_ratio"] = pno, t.bbox[3] / page.height
                    continue
                above = [ln for ln in lines if 0 < t.bbox[1] - ln["bottom"] < 40]
                caption = above[-1]["text"].strip() if above else ""
                tables.append({"rows": rows, "caption": caption, "page": pno, "bottom_ratio": t.bbox[3] / page.height})
            boxes = [t.bbox for t in found]
            outside = page.filter(lambda obj: obj.get("object_type") != "char" or not any(
                b[0] <= obj["x0"] <= b[2] and b[1] <= obj["top"] <= b[3] for b in boxes))
            prose.extend(_paragraphs_from_lines(outside.extract_text_lines()))

    docs, datasets, warnings = [], [], []
    captions = {t["caption"] for t in tables}
    for n, t in enumerate(tables, start=1):
        if len(t["rows"]) < 2:
            continue
        caption = t["caption"] or f"Table {n}"
        d, ds = table_dataset(_table_rows_to_frame(t["rows"]), source=f.name, name=_unique_name(f"{stem}_t{n}", taken),
                              caption=caption, ref_prefix=f"Table {n}, ", doc_type="pdf_table_row", doc_prefix=f"t{n}-")
        docs += d
        datasets.append(ds)
    paragraphs = [p for p in prose if len(p) > 60 and p not in captions]
    texts = []
    if paragraphs:
        name = _unique_name(f"{stem}_text", taken)
        docs += text_documents(paragraphs, source=f.name, name=name, doc_type="pdf_text")
        texts.append(TextSource(name, f.name, f"prose paragraphs of {f.name}", len(paragraphs)))
    if not docs:
        warnings.append("no extractable text or tables (scanned PDF? OCR is not supported)")
    return docs, datasets, texts, warnings


def _load_docx(f: SourceFile, stem: str, taken: set[str]):
    import docx

    d = docx.Document(io.BytesIO(f.data))
    docs, datasets = [], []
    for n, table in enumerate(d.tables, start=1):
        rows = [[c.text.strip() for c in r.cells] for r in table.rows]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue
        dd, ds = table_dataset(_table_rows_to_frame(rows), source=f.name, name=_unique_name(f"{stem}_t{n}", taken),
                               caption=f"Table {n}", ref_prefix=f"Table {n}, ", doc_type="docx_table_row",
                               doc_prefix=f"t{n}-")
        docs += dd
        datasets.append(ds)
    paras, sections, heading = [], [], None
    for p in d.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if p.style is not None and p.style.name.lower().startswith(("heading", "title")):
            heading = text
            continue
        paras.append(text)
        sections.append(heading)
    texts = []
    if paras:
        name = _unique_name(f"{stem}_text", taken)
        docs += text_documents(paras, source=f.name, name=name, doc_type="docx_text", sections=sections)
        texts.append(TextSource(name, f.name, f"prose paragraphs of {f.name}", len(paras)))
    return docs, datasets, texts, [] if docs else ["document has no text"]


def _load_text(f: SourceFile, stem: str, taken: set[str]):
    paras, sections, heading = [], [], None
    for block in re.split(r"\n\s*\n", _decode(f.data)):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if lines[0].lstrip().startswith("#"):
            heading = lines[0].lstrip("# ").strip()
            block = "\n".join(lines[1:]).strip()
            if not block:
                continue
        paras.append(re.sub(r"\s*\n\s*", " ", block))
        sections.append(heading)
    if not paras:
        return [], [], [], ["file is empty"]
    name = _unique_name(f"{stem}_text", taken)
    return (text_documents(paras, source=f.name, name=name, doc_type="text", sections=sections), [],
            [TextSource(name, f.name, f"text paragraphs of {f.name}", len(paras))], [])


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


_LOADERS = {".csv": _load_delimited, ".tsv": _load_delimited, ".xlsx": _load_excel, ".json": _load_json,
            ".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_text, ".md": _load_text}


# --- knowledge base -----------------------------------------------------------------------

def _catalog_doc(ds: Dataset) -> Document:
    cols = [c for c in ds.frame.columns if not str(c).startswith("_")]
    text = (f"Dataset overview: {ds.source} ({ds.name}) - {ds.description}. "
            f"{len(ds.frame)} records. Columns: {', '.join(label(c) for c in cols)}.")
    return Document(doc_id=f"{ds.source}#overview-{ds.name}", text=text, source=ds.source,
                    doc_type="catalog", metadata={"category": "catalog", "section": "dataset overview",
                                                  "record_ref": "dataset overview"})


def build_knowledge_base(files: list[SourceFile]) -> tuple[KnowledgeBase, list[LoadReport]]:
    """Parse every file independently: a bad file is reported, never fatal."""
    docs: list[Document] = []
    datasets: list[Dataset] = []
    texts: list[TextSource] = []
    reports: list[LoadReport] = []
    taken: set[str] = {"catalog"}
    for f in files:
        ext = Path(f.name).suffix.lower()
        loader = _LOADERS.get(ext)
        if loader is None:
            reports.append(LoadReport(f.name, False, error=f"unsupported file type '{ext}' (supported: {', '.join(SUPPORTED)})"))
            continue
        if len(f.data) > MAX_FILE_BYTES:
            reports.append(LoadReport(f.name, False, error=f"file larger than {MAX_FILE_BYTES // 2**20} MB"))
            continue
        try:
            d, dss, txt, warnings = loader(f, _slug(Path(f.name).stem), taken)
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            reports.append(LoadReport(f.name, False, error=f"could not parse: {type(exc).__name__}: {exc}"))
            continue
        docs += d
        datasets += dss
        texts += txt
        reports.append(LoadReport(f.name, bool(d), len(d), [x.name for x in dss], sum(t.paragraphs for t in txt),
                                  warnings, None if d else "nothing could be extracted"))
    docs += [_catalog_doc(x) for x in datasets]
    kb = KnowledgeBase(docs, {x.name: x for x in datasets}, {t.name: t for t in texts})
    return kb, reports


def sample_files(data_dir: Path = DATA_DIR) -> list[SourceFile]:
    if not data_dir.is_dir():   # e.g. RAG_DATA_DIR points nowhere: start empty rather than crash
        logging.getLogger(__name__).warning("Data folder %s not found - starting with no documents", data_dir)
        return []
    return [SourceFile(p.name, p.read_bytes()) for p in sorted(data_dir.iterdir())
            if p.suffix.lower() in SUPPORTED]


def load_knowledge_base(data_dir: Path = DATA_DIR) -> KnowledgeBase:
    """The bundled sample documents, loaded through the same path as uploads."""
    return build_knowledge_base(sample_files(data_dir))[0]
