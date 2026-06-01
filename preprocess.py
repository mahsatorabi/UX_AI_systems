"""
Bibliometric text preprocessing for Scopus UX × Generative AI corpus.

Pipeline: load → encode fix → field merge → cleaning → normalization →
tokenization → stopword removal (NLTK + manual) → lemmatization (spaCy) → export.

Run:
    python preprocess.py
    python preprocess.py --input data.csv --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import unicodedata

import nltk

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None  # type: ignore
import pandas as pd
import regex
from tqdm import tqdm

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:  # pragma: no cover
    _HAS_LANGDETECT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEXT_COLUMNS = ("Title", "Abstract", "Author Keywords", "Index Keywords")
DOCUMENT_SEPARATOR = " "

# Scopus / publisher boilerplate (case-insensitive)
BOILERPLATE_PATTERNS = [
    r"©\s*.*?(?:springer|elsevier|ieee|wiley|sage|taylor\s*&\s*francis|nature|mdpi|frontiers|acm|author\(s\)).*",
    r"all rights reserved\.?",
    r"under exclusive license to.*",
    r"published by.*",
    r"available (?:online|at).*",
    r"peer review under responsibility of.*",
    r"this is an open access article.*",
    r"creative commons.*",
    r"https?://\S+",
    r"www\.\S+",
    r"\bdoi:\s*\S+",
    r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",  # DOI
    r"scopus\.com/\S*",
]

# Characters to strip after regex passes
CONTROL_CHARS = regex.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Scopus placeholder when abstract is missing
NO_ABSTRACT_PATTERN = regex.compile(
    r"^\s*\[?\s*no\s+abstract\s+available\s*\]?\s*\.?\s*$",
    flags=regex.IGNORECASE,
)

# Token filters
MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 45

# NLTK resources (downloaded on first run)
NLTK_PACKAGES = (
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng",
)

# Manual stopwords: bibliometric noise + Scopus artifacts (NOT domain terms like UX/LLM)
MANUAL_STOPWORDS = frozenset(
    w.lower()
    for w in """
    abstract introduction background methodology methods method materials
    results result discussion conclusion conclusions finding findings
    study studies paper papers article articles journal volume vol issue
    pp page pages figure fig table tab appendix supplementary
    author authors et al affiliation affiliations correspondence
    copyright licensed license springer elsevier wiley ieee acm mdpi
    published publisher publishing peer reviewed review reviews
    however therefore furthermore moreover thus hence namely
    also used using use based within among across via per
    one two three four five six seven eight nine ten
    may might could would shall should must
    data dataset datasets sample samples participant participants
    shown show shows showed indicate indicates indicated suggesting
    overall respectively specifically particularly primarily
    present presented presents providing provided provide
    aim aims objective objectives purpose purposes
    related regarding concerning addressed address addresses
    new novel recent future previous prior existing
    et al etc ie eg vs
    """.split()
)

# Domain terms to preserve even if they appear in NLTK stopwords
PROTECTED_TERMS = frozenset(
    w.lower()
    for w in """
    ux user experience generative ai genai llm llms gpt chatgpt
    chatbot chatbots conversational ai artificial intelligence
    human-computer interaction hci
    """.split()
)


@dataclass
class PreprocessConfig:
    input_path: Path = Path("data.csv")
    output_dir: Path = Path("outputs")
    spacy_model: str = "en_core_web_sm"
    english_only_lemma: bool = True
    min_doc_tokens: int = 10
    save_tokens_json: bool = True
    save_parquet: bool = True
    limit: int | None = None  # debug: process first N rows only


# ---------------------------------------------------------------------------
# NLTK / spaCy bootstrap
# ---------------------------------------------------------------------------


def ensure_nltk_data() -> None:
    for pkg in NLTK_PACKAGES:
        try:
            nltk.data.find(
                f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
                if pkg in ("stopwords", "wordnet", "omw-1.4")
                else f"taggers/{pkg}"
            )
        except LookupError:
            logger.info("Downloading NLTK package: %s", pkg)
            nltk.download(pkg, quiet=True)


def load_spacy_model(model_name: str):
    import spacy

    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        logger.error(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            model_name,
            model_name,
        )
        sys.exit(1)


def get_nltk_stopwords() -> set[str]:
    from nltk.corpus import stopwords

    stops = set(stopwords.words("english"))
    # Remove protected domain terms if NLTK lists them
    return stops - PROTECTED_TERMS


def build_stopword_set() -> set[str]:
    return (get_nltk_stopwords() | MANUAL_STOPWORDS) - PROTECTED_TERMS


# ---------------------------------------------------------------------------
# Text cleaning & normalization
# ---------------------------------------------------------------------------


def fix_encoding(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    if ftfy is not None:
        text = ftfy.fix_text(text, normalization="NFKC")
    else:
        text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARS.sub("", text)
    return text


def remove_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = regex.sub(pat, " ", text, flags=regex.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    text = regex.sub(r"\s+", " ", text).strip()
    return text


def normalize_hyphens_and_quotes(text: str) -> str:
    # Unify dash types; smart quotes → ASCII
    text = regex.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = regex.sub(r"[''`´]", "'", text)
    text = regex.sub(r"[\u201c\u201d\u201e\u00ab\u00bb]", '"', text)
    return text


def expand_contractions(text: str) -> str:
    """Light contraction expansion for academic prose."""
    mapping = {
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bn't\b": " not",
        r"\b're\b": " are",
        r"\b've\b": " have",
        r"\b'll\b": " will",
        r"\b'd\b": " would",
        r"\bm\b": " am",
        r"\s+'s\b": " is",  # it's → it is (rough)
    }
    for pat, repl in mapping.items():
        text = regex.sub(pat, repl, text, flags=regex.IGNORECASE)
    return text


def clean_text(raw: str) -> str:
    """Full cleaning pass without lemmatization (keeps case for detection)."""
    if pd.isna(raw) or raw is None:
        return ""
    text = str(raw)
    text = fix_encoding(text)
    text = normalize_hyphens_and_quotes(text)
    text = remove_boilerplate(text)
    # Emails, leftover HTML entities
    text = regex.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)
    text = regex.sub(r"&[a-z]+;|&#\d+;", " ", text, flags=regex.IGNORECASE)
    text = expand_contractions(text)
    text = normalize_whitespace(text)
    return text


def merge_document_fields(row: pd.Series, columns: Iterable[str] = TEXT_COLUMNS) -> str:
    parts = []
    for col in columns:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(clean_text(str(val)))
    return DOCUMENT_SEPARATOR.join(p for p in parts if p)


def detect_language(text: str) -> str:
    if not _HAS_LANGDETECT:
        return "unknown"
    sample = text[:4000] if len(text) > 4000 else text
    if len(sample.strip()) < 40:
        return "unknown"
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def normalize_for_tokenization(text: str) -> str:
    """Lowercase and soften punctuation for token boundary stability."""
    text = text.lower()
    # Keep intra-token hyphens (e.g., human-computer); split sentence punctuation
    text = regex.sub(r"[^\w\s\-+/#]", " ", text)
    text = regex.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tokenization, stopwords, lemmatization
# ---------------------------------------------------------------------------


def filter_tokens(tokens: Iterable[str]) -> list[str]:
    out = []
    for t in tokens:
        t = t.strip("-_")
        if not t:
            continue
        if len(t) < MIN_TOKEN_LEN or len(t) > MAX_TOKEN_LEN:
            continue
        if t.isdigit():
            continue
        if regex.fullmatch(r"[\W_]+", t):
            continue
        out.append(t)
    return out


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    return [t for t in tokens if t not in stopwords or t in PROTECTED_TERMS]


def lemmatize_spacy(
    texts: list[str],
    nlp,
    stopwords: set[str],
    batch_size: int = 256,
) -> list[list[str]]:
    """Lemma + POS-filter with spaCy pipe."""
    results: list[list[str]] = []
    allowed_pos = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}

    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        desc="Lemmatizing (spaCy)",
    ):
        tokens = []
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            if tok.like_num:
                continue
            lemma = tok.lemma_.lower().strip()
            if tok.pos_ not in allowed_pos and tok.pos_ != "X":
                # Keep protected multi-word concepts handled as single tokens
                if lemma not in PROTECTED_TERMS:
                    continue
            lemma = regex.sub(r"[^\w\-+#/]", "", lemma)
            if not lemma or len(lemma) < MIN_TOKEN_LEN:
                continue
            if lemma in stopwords and lemma not in PROTECTED_TERMS:
                continue
            tokens.append(lemma)
        results.append(filter_tokens(tokens))
    return results


def process_non_english(texts: list[str], stopwords: set[str]) -> list[list[str]]:
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    wnl = WordNetLemmatizer()
    out = []
    for text in tqdm(texts, desc="Tokenizing (non-English fallback)"):
        toks = word_tokenize(text)
        toks = [
            wnl.lemmatize(t.lower())
            for t in filter_tokens(toks)
            if t.lower() not in stopwords or t.lower() in PROTECTED_TERMS
        ]
        out.append(filter_tokens(toks))
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def has_valid_abstract(value) -> bool:
    """False for NaN, blank, or Scopus '[No abstract available]' placeholders."""
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    if text.lower() in ("nan", "none", "n/a", "na"):
        return False
    if NO_ABSTRACT_PATTERN.match(text):
        return False
    return True


def filter_missing_abstracts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, excluded) rows based on raw Abstract column."""
    valid = df["Abstract"].apply(has_valid_abstract)
    kept = df.loc[valid].copy()
    excluded = df.loc[~valid].copy()
    return kept, excluded


def load_corpus(path: Path) -> pd.DataFrame:
    logger.info("Loading %s", path)
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")
    logger.info("Loaded %d records, %d columns", len(df), len(df.columns))
    missing = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def run_pipeline(cfg: PreprocessConfig) -> pd.DataFrame:
    ensure_nltk_data()
    stopwords = build_stopword_set()
    logger.info("Stopword set size: %d (NLTK + manual − protected)", len(stopwords))

    df = load_corpus(cfg.input_path)
    n_source = len(df)

    df, excluded_no_abstract = filter_missing_abstracts(df)
    n_excluded = len(excluded_no_abstract)
    if n_excluded:
        logger.info(
            "Excluded %d records with no abstract (%.2f%% of corpus)",
            n_excluded,
            100.0 * n_excluded / n_source,
        )
    else:
        logger.info("No records excluded for missing abstract")
    df.attrs["n_excluded_no_abstract"] = n_excluded
    df.attrs["excluded_no_abstract_df"] = excluded_no_abstract

    if cfg.limit is not None:
        logger.warning(
            "TEST MODE: --limit %d — only a subset will be exported. "
            "Omit --limit to process all %d records.",
            cfg.limit,
            n_source,
        )
        df = df.head(cfg.limit).copy()
    else:
        logger.info(
            "Processing %d records (%d excluded, no abstract)",
            len(df),
            n_excluded,
        )
    df.attrs["n_source_records"] = n_source

    # Per-field cleaned text
    for col in TEXT_COLUMNS:
        clean_col = f"{col}_clean"
        logger.info("Cleaning column: %s", col)
        df[clean_col] = df[col].apply(clean_text)

    logger.info("Building merged document text")
    df["text_merged_clean"] = df.apply(merge_document_fields, axis=1)

    # Language metadata
    logger.info("Detecting document languages")
    tqdm.pandas(desc="Language detection")
    df["detected_language"] = df["text_merged_clean"].progress_apply(detect_language)

    scopus_lang = df.get("Language of Original Document", pd.Series(dtype=str))
    df["language_scopus"] = scopus_lang

    # Normalized text for tokenization
    df["text_normalized"] = df["text_merged_clean"].apply(normalize_for_tokenization)

    # Split English vs other for lemmatization strategy
    scopus_en = (
        df["language_scopus"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("english")
    )
    is_english = df["detected_language"].eq("en") | scopus_en
    eng_idx = df.index[is_english].tolist()
    other_idx = df.index[~is_english].tolist()
    logger.info("English pipeline: %d docs | Fallback: %d docs", len(eng_idx), len(other_idx))

    df["tokens"] = None
    df["tokens"] = df["tokens"].astype(object)
    df["token_count"] = 0
    df["preprocess_method"] = ""

    nlp = None
    if eng_idx and cfg.english_only_lemma:
        nlp = load_spacy_model(cfg.spacy_model)

    if eng_idx:
        eng_texts = df.loc[eng_idx, "text_normalized"].tolist()
        eng_tokens = lemmatize_spacy(eng_texts, nlp, stopwords)
        for i, toks in zip(eng_idx, eng_tokens):
            df.at[i, "tokens"] = toks
            df.at[i, "token_count"] = len(toks)
            df.at[i, "preprocess_method"] = f"spacy_{cfg.spacy_model}"

    if other_idx:
        other_texts = df.loc[other_idx, "text_normalized"].tolist()
        other_tokens = process_non_english(other_texts, stopwords)
        for i, toks in zip(other_idx, other_tokens):
            df.at[i, "tokens"] = toks
            df.at[i, "token_count"] = len(toks)
            df.at[i, "preprocess_method"] = "nltk_fallback"

    # Space-joined string for vectorizers / LDA
    df["text_processed"] = df["tokens"].apply(
        lambda ts: " ".join(ts) if isinstance(ts, list) else ""
    )

    # Drop very short documents after preprocessing
    short_mask = df["token_count"] < cfg.min_doc_tokens
    if short_mask.any():
        logger.warning(
            "Flagging %d documents with <%d tokens (kept in export, column is_short_doc)",
            short_mask.sum(),
            cfg.min_doc_tokens,
        )
    df["is_short_doc"] = short_mask

    return df


def write_preprocess_report(df: pd.DataFrame, cfg: PreprocessConfig, summary: dict) -> None:
    """Human-readable preprocessing report (Markdown)."""
    n_source = summary["n_source_records"]
    n_excl = summary["n_excluded_no_abstract"]
    n_out = summary["n_documents_exported"]
    excluded_df = df.attrs.get("excluded_no_abstract_df")
    lines = [
        "# Preprocessing Report — UX × Generative AI Corpus",
        "",
        f"**Input:** `{cfg.input_path}`  ",
        f"**Output directory:** `{cfg.output_dir}`  ",
        "",
        "## Record counts",
        "",
        "| Stage | Count |",
        "|-------|------:|",
        f"| Records in Scopus export | {n_source} |",
        f"| Excluded (no abstract) | {n_excl} |",
        f"| **Final corpus** | **{n_out}** |",
        "",
        "## Preprocessing steps",
        "",
        "1. Load CSV (UTF-8)",
        "2. **Drop** rows with missing/placeholder abstract (`[No abstract available]`)",
        "3. Per-field text cleaning (encoding, boilerplate, URLs/DOI)",
        "4. Merge Title + Abstract + Author Keywords + Index Keywords",
        "5. Language detection + English spaCy lemmatization (non-English: NLTK fallback)",
        "6. NLTK + manual stopword removal (domain terms protected)",
        "",
        "## Token statistics (final corpus)",
        "",
        f"- Mean tokens per document: **{summary['mean_tokens']:.1f}**",
        f"- Median tokens per document: **{summary['median_tokens']:.0f}**",
        f"- Documents flagged as short (<{cfg.min_doc_tokens} tokens): **{summary['n_short_docs']}**",
        "",
        "## Language / lemmatization",
        "",
        f"- spaCy (`{cfg.spacy_model}`): **{summary['n_english_spacy']}** documents",
        f"- NLTK fallback: **{summary['n_fallback']}** documents",
        "",
    ]
    if n_excl and excluded_df is not None and len(excluded_df):
        lines.extend(["## Excluded titles (no abstract)", ""])
        for _, row in excluded_df.iterrows():
            title = str(row.get("Title", ""))[:120]
            year = row.get("Year", "")
            lines.append(f"- ({year}) {title}")
        lines.append("")
    report_path = cfg.output_dir / "preprocess_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)


def export_outputs(df: pd.DataFrame, cfg: PreprocessConfig) -> None:
    n_source = df.attrs.get("n_source_records", len(df))
    n_excluded = df.attrs.get("n_excluded_no_abstract", 0)
    expected = n_source - n_excluded
    if cfg.limit is not None:
        expected = min(expected, cfg.limit)
    if len(df) != expected:
        raise RuntimeError(
            f"Record count mismatch: expected {expected} after exclusions, got {len(df)}."
        )

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    excluded_df = df.attrs.get("excluded_no_abstract_df")
    if excluded_df is not None and len(excluded_df):
        excl_path = out / "excluded_no_abstract.csv"
        excluded_df.to_csv(excl_path, index=False, encoding="utf-8")
        logger.info("Wrote %d excluded records to %s", len(excluded_df), excl_path)

    # Columns safe for CSV (tokens as string)
    export_df = df.copy()
    export_df["tokens"] = export_df["tokens"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]"
    )

    csv_path = out / "corpus_preprocessed.csv"
    export_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Wrote %s", csv_path)

    if cfg.save_parquet:
        try:
            pq_path = out / "corpus_preprocessed.parquet"
            # Parquet: keep list column natively
            df_parquet = df.copy()
            df_parquet.to_parquet(pq_path, index=False)
            logger.info("Wrote %s", pq_path)
        except Exception as e:
            logger.warning("Parquet export skipped: %s", e)

    if cfg.save_tokens_json:
        tokens_path = out / "corpus_tokens.jsonl"
        with tokens_path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                rec = {
                    "doi": row.get("DOI"),
                    "title": row.get("Title"),
                    "year": row.get("Year"),
                    "tokens": row.get("tokens") or [],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %s", tokens_path)

    # Summary stats
    summary = {
        "n_source_records": int(n_source),
        "n_excluded_no_abstract": int(n_excluded),
        "n_documents_exported": int(len(df)),
        "limit_applied": cfg.limit,
        "n_documents": int(len(df)),
        "n_english_spacy": int((df["preprocess_method"].str.startswith("spacy")).sum()),
        "n_fallback": int((df["preprocess_method"] == "nltk_fallback").sum()),
        "mean_tokens": float(df["token_count"].mean()),
        "median_tokens": float(df["token_count"].median()),
        "n_short_docs": int(df["is_short_doc"].sum()),
        "language_counts": df["detected_language"].value_counts().to_dict(),
    }
    summary_path = out / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    write_preprocess_report(df, cfg, summary)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


def parse_args() -> PreprocessConfig:
    p = argparse.ArgumentParser(description="Preprocess Scopus UX×GenAI corpus")
    p.add_argument("--input", type=Path, default=Path("data.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--min-doc-tokens", type=int, default=10)
    p.add_argument("--no-parquet", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    args = p.parse_args()
    return PreprocessConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        spacy_model=args.spacy_model,
        min_doc_tokens=args.min_doc_tokens,
        save_parquet=not args.no_parquet,
        limit=args.limit,
    )


def main() -> None:
    cfg = parse_args()
    if not cfg.input_path.is_file():
        logger.error("Input file not found: %s", cfg.input_path)
        sys.exit(1)

    df = run_pipeline(cfg)
    export_outputs(df, cfg)
    logger.info("Done. Outputs in %s", cfg.output_dir.resolve())


if __name__ == "__main__":
    main()
