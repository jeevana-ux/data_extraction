#!/usr/bin/env python3
"""
Extract full text and tables from one or more PDFs, while removing
email caution / disclaimer content in both text and tables.

Key constraints:
1. No caution/disclaimer content in text output.
2. No caution/disclaimer content in extracted tables (CSVs).
3. Email Subject line must be preserved (important info).

Folder structure per file:

output/
  <pdf_basename>/
    <YYYYMMDD_HHMMSS>/
      <pdf_basename>_full_text.txt
      <pdf_basename>_pageX_table_Y.csv
      <pdf_basename>_summary.json
"""

import io
import json
import argparse
from pathlib import Path
from datetime import datetime
import re

# --- third-party libs ---
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image
import pytesseract
import pandas as pd

# try to import camelot (optional)
try:
    import camelot
    HAS_CAMELOT = True
except Exception:
    HAS_CAMELOT = False

# -------------------------
INPUT_DIR_DEFAULT = "input"
OUT_DIR = "output"
# -------------------------


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " ._-()" else "_" for c in s)


# ==============================
#  DISCLAIMER / CAUTION FILTERING
# ==============================

# Phrases that strongly indicate disclaimer/caution text
DISCLAIMER_KEY_PHRASES = [
    "this email and any files transmitted with it are confidential",
    "intended solely for the use of the",
    "if you have received this email in error, please notify",
    "if you are not the intended recipient",
    "you are notified that disclosing, copying, distributing or taking any action in reliance",
    "strictly prohibited",
    "any views or opinions presented in this email are solely those of the author",
    "our organization accepts no liability for the content of this email",
    "email, and any attachment herein, is confidential and intended solely for the addressee",
    "email transmission cannot be guaranteed to be secure, error-free, or virus-free",
    "pidilite industries ltd.",
    "this email may contain confidential, proprietary or legally privileged information",
]

# Line-level noise / Gmail print artefacts
EMAIL_NOISE_LINE_PATTERNS = [
    r"\[Quoted text hidden\]",
    r"https://mail\.google\.com/mail/u/",
]

# Email headers we *do* want to drop (note: Subject is intentionally NOT here)
EMAIL_HEADER_PREFIXES = (
    "From:",
    "To:",
    "Cc:",
    "Bcc:",
    "Sent:",
    "Date:",
)

SEPARATOR_LINE_RE = re.compile(r"^\s*-{5,}\s*$")


def looks_like_disclaimer(text: str) -> bool:
    """
    Heuristic: if text looks like a disclaimer / caution block.
    """
    if not text:
        return False

    low = " ".join(text.lower().split())
    # Paragraphs starting with "disclaimer:" or "caution:" are almost always disclaimers
    if low.startswith("disclaimer:") or low.startswith("caution:"):
        return True

    # If any strong key phrase appears, treat as disclaimer
    for phrase in DISCLAIMER_KEY_PHRASES:
        if phrase in low:
            return True

    return False


def clean_email_text(text: str) -> str:
    """
    Remove email headers, caution/disclaimer blocks, gmail noise, etc.
    while PRESERVING Subject: lines.
    """
    if not text:
        return text

    # 1) Paragraph-level filtering (for big disclaimer blocks)
    paragraphs = re.split(r"\n\s*\n", text)
    kept_paragraphs = []
    for p in paragraphs:
        if not p.strip():
            kept_paragraphs.append(p)
            continue
        if looks_like_disclaimer(p):
            # drop the entire disclaimer/caution paragraph
            continue
        kept_paragraphs.append(p)

    cleaned = "\n\n".join(kept_paragraphs)

    # 2) Line-level filtering
    result_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()

        # keep blank lines (we'll compress later)
        if not stripped:
            result_lines.append(line)
            continue

        # Drop non-subject headers
        if any(stripped.startswith(prefix) for prefix in EMAIL_HEADER_PREFIXES):
            continue

        # IMPORTANT: we DO NOT drop "Subject:" lines because they may contain relevant info
        # if stripped.startswith("Subject:"):
        #     -> KEEP

        # Drop separator lines
        if SEPARATOR_LINE_RE.match(stripped):
            continue

        # Drop gmail noise like [Quoted text hidden], Gmail URLs, etc.
        if any(re.search(pat, stripped) for pat in EMAIL_NOISE_LINE_PATTERNS):
            continue

        # Drop pure "1/2", "2/2", etc.
        if re.fullmatch(r"\d+/\d+", stripped):
            continue

        # Drop individual lines that look like disclaimer text
        if looks_like_disclaimer(stripped):
            continue

        result_lines.append(line)

    cleaned = "\n".join(result_lines)

    # 3) Compress multiple blank lines
    final_lines = []
    blank_count = 0
    for line in cleaned.splitlines():
        if line.strip() == "":
            blank_count += 1
            if blank_count > 2:
                continue
        else:
            blank_count = 0
        final_lines.append(line)

    return "\n".join(final_lines).strip()


def _is_empty_value(v) -> bool:
    """
    Helper to check if a table cell is effectively empty.
    """
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _is_empty_value(v) -> bool:
    """
    Helper to check if a table cell is effectively empty.
    """
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def clean_email_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Remove disclaimer/caution content from a table.
    - Any cell that looks like disclaimer text or Gmail noise is blanked.
    - Rows that become entirely empty are removed.
    If the whole table becomes empty, returns None.
    """
    if df is None or df.empty:
        return None

    new_df = df.copy()

    def clean_cell(val):
        # Only operate on strings
        if isinstance(val, str):
            txt = val

            # Remove Gmail noise like URLs, [Quoted text hidden], etc.
            if any(re.search(pat, txt) for pat in EMAIL_NOISE_LINE_PATTERNS):
                return ""

            # Remove disclaimer / caution text
            if looks_like_disclaimer(txt):
                return ""

            return txt
        return val

    # Clean every cell in the table
    new_df = new_df.applymap(clean_cell)

    # Drop rows that are completely empty after cleaning
    mask_all_empty = new_df.apply(
        lambda row: all(_is_empty_value(v) for v in row), axis=1
    )
    new_df = new_df.loc[~mask_all_empty]

    if new_df.empty:
        return None

    return new_df



# ==============================
#  EXTRACTION
# ==============================

def extract_text_with_pdfplumber(pdf_path: str) -> str:
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    txt = page.extract_text()
                except Exception:
                    txt = None
                if not txt:
                    txt = ""
                page_header = f"\n\n--- PAGE {i} ---\n\n"
                full_text.append(page_header + txt)
        return "\n".join(full_text).strip()
    except Exception as e:
        print(f"[pdfplumber] failed for {pdf_path}: {e}")
        return ""


def ocr_pdf_pages_with_pypdfium2(pdf_path: str, dpi: int = 200, lang: str = "eng") -> str:
    texts = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"[pypdfium2] failed to open PDF for OCR: {pdf_path}: {e}")
        return ""

    for i, page in enumerate(pdf, start=1):
        try:
            # Render the page to a PIL image
            bitmap = page.render(scale=dpi / 72)
            pil_image = bitmap.to_pil()
            
            # Run OCR
            txt = pytesseract.image_to_string(pil_image, lang=lang)
            header = f"\n\n--- OCR PAGE {i} ---\n\n"
            texts.append(header + txt)
        except Exception as e:
            print(f"[OCR] page {i} failed for {pdf_path}: {e}")
            texts.append(f"\n\n--- OCR PAGE {i} ERROR ---\n\n")
    return "\n".join(texts).strip()


def extract_tables_with_pdfplumber(pdf_path: str) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    page_tables = page.extract_tables()
                except Exception:
                    page_tables = []
                for t in page_tables:
                    try:
                        if not t or len(t) < 1:
                            continue
                        df = pd.DataFrame(t[1:], columns=t[0])
                    except Exception:
                        df = pd.DataFrame(t)
                    df.attrs["page"] = i
                    tables.append(df)
    except Exception as e:
        print(f"[pdfplumber tables] failed for {pdf_path}: {e}")
    return tables


def extract_tables_with_camelot(pdf_path: str) -> list[pd.DataFrame]:
    if not HAS_CAMELOT:
        return []
    dfs: list[pd.DataFrame] = []
    try:
        for flavor in ["lattice", "stream"]:
            try:
                tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
                for t in tables:
                    try:
                        dfs.append(t.df)
                    except Exception:
                        continue
            except Exception:
                pass
    except Exception as e:
        print(f"[camelot] failed for {pdf_path}: {e}")
    return dfs


# ==============================
#  OUTPUT DIRECTORY
# ==============================

def prepare_run_output_dir(pdf_path: str):
    basename = Path(pdf_path).stem
    safe_base = safe_filename(basename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(OUT_DIR) / safe_base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, safe_base, timestamp


def save_outputs(pdf_path: str, text: str, tables: list[pd.DataFrame],
                 run_dir: Path, safe_base: str, timestamp: str):
    # text file
    text_file = run_dir / f"{safe_base}_full_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text or "")

    # tables
    table_files: list[str] = []
    for i, df in enumerate(tables, start=1):
        csv_path = run_dir / f"{safe_base}_table_{i}.csv"
        try:
            page = df.attrs.get("page", None) if hasattr(df, "attrs") else None
            if page:
                csv_path = run_dir / f"{safe_base}_page{page}_table_{i}.csv"
        except Exception:
            pass
        df.to_csv(csv_path, index=False)
        table_files.append(str(csv_path))

    # summary
    summary = {
        "pdf_path": str(pdf_path),
        "output_root": str(Path(OUT_DIR).resolve()),
        "input_basename": safe_base,
        "run_timestamp": timestamp,
        "run_folder": str(run_dir),
        "text_file": str(text_file),
        "num_tables": len(table_files),
        "table_files": table_files,
    }
    summary_file = run_dir / f"{safe_base}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary_file, text_file, table_files


# ==============================
#  MAIN PER-FILE RUN
# ==============================

def run_for_pdf(pdf_path: str, ocr_if_empty: bool = True, try_camelot: bool = True):
    print(f"\n[run] extracting from: {pdf_path}")

    run_dir, safe_base, timestamp = prepare_run_output_dir(pdf_path)
    print(f"[run] output directory: {run_dir}")

    # --- TEXT ---
    text = extract_text_with_pdfplumber(pdf_path)
    print(f"[run] pdfplumber extracted {len(text)} chars")

    # --- TABLES ---
    tables = extract_tables_with_pdfplumber(pdf_path)
    print(f"[run] pdfplumber found {len(tables)} tables")

    if try_camelot and HAS_CAMELOT:
        camelot_tables = extract_tables_with_camelot(pdf_path)
        print(f"[run] camelot found {len(camelot_tables)} tables")
        tables.extend(camelot_tables)

    # OCR fallback for text
    if ocr_if_empty and (not text or len(text.strip()) < 50):
        print("[run] text is short/empty, running OCR fallback...")
        ocr_text = ocr_pdf_pages_with_pypdfium2(pdf_path)
        if ocr_text and (not text or len(text.strip()) < 50):
            text = ocr_text
            print(f"[run] OCR extracted {len(text)} chars")

    # --- CLEANING PHASE ---
    # 1) clean text (remove disclaimers, keep Subject)
    text = clean_email_text(text)
    print(f"[run] cleaned text length: {len(text)} chars")

    # 2) clean tables (remove disclaimer rows/cells)
    cleaned_tables: list[pd.DataFrame] = []
    for df in tables:
        cleaned_df = clean_email_table(df)
        if cleaned_df is not None and not cleaned_df.empty:
            cleaned_tables.append(cleaned_df)
    print(f"[run] tables after disclaimer cleaning: {len(cleaned_tables)}")

    # --- SAVE ---
    summary_file, text_file, table_files = save_outputs(
        pdf_path, text, cleaned_tables, run_dir, safe_base, timestamp
    )
    print(f"[run] saved outputs. summary: {summary_file}")

    return {
        "run_folder": str(run_dir),
        "summary_file": str(summary_file),
        "text_file": str(text_file),
        "table_files": table_files,
    }


# ==============================
#  CLI ENTRYPOINT
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract full text and tables from PDFs, removing email disclaimers/cautions."
    )
    parser.add_argument(
        "--pdf",
        help="Path to a single PDF. If omitted, all PDFs in --input-dir are processed.",
    )
    parser.add_argument(
        "--input-dir",
        default=INPUT_DIR_DEFAULT,
        help="Directory containing PDF files (default: input/).",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    parser.add_argument("--no-camelot", action="store_true", help="Disable camelot attempts")
    args = parser.parse_args()

    if args.pdf:
        pdf_paths = [Path(args.pdf)]
    else:
        input_dir = Path(args.input_dir)
        pdf_paths = sorted(input_dir.glob("*.pdf"))
        if not pdf_paths:
            print(f"No PDFs found in {input_dir.resolve()}")
            raise SystemExit(1)

    all_results = {}
    for pdf_path in pdf_paths:
        res = run_for_pdf(str(pdf_path),
                          ocr_if_empty=(not args.no_ocr),
                          try_camelot=(not args.no_camelot))
        all_results[str(pdf_path)] = res

    print("\n[all runs completed]")
    print(json.dumps(all_results, indent=2))
