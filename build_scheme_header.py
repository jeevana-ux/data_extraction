"""
build_scheme_header_from_extraction.py

Reads extracted text+tables from output/<pdf_id>/<timestamp>/,
creates a combined mail_body, calls LLM to extract all Retailer Hub
scheme header fields, and outputs scheme_header.csv.

Usage:
1. pip install openai python-dotenv pandas
2. Create .env file with: OPENAI_API_KEY=your_key_here
3. Run: python build_scheme_header_from_extraction.py
"""

import os
import json
import hashlib
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# =============================
# Load .env
# =============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# Folder Configuration
# =============================
OUTPUT_ROOT = "output"
FINAL_OUT_DIR = "out"
os.makedirs(FINAL_OUT_DIR, exist_ok=True)

SCHEME_HEADER_CSV = os.path.join(FINAL_OUT_DIR, "scheme_header.csv")


# =============================
# LLM Prompt
# =============================
SCHEME_SYSTEM_PROMPT = """
You help prepare Flipkart Retailer Hub scheme headers from brand emails.

INPUT:
- email subject
- email body text (includes pasted table CSVs)

OUTPUT:
Return ONLY a valid JSON object. No extra text.
Format:
{
  "schemes": [
    {
      "scheme_type": "...",
      "scheme_sub_type": "...",
      "scheme_name": "...",
      "scheme_description": "...",
      "description": "...",
      "scheme_period": "EVENT | DURATION",
      "duration_start_date": "YYYY-MM-DD or null",
      "duration_end_date": "YYYY-MM-DD or null",
      "discount_type": "...",
      "global_cap_amount": null or number,
      "min_actual_or_agreed": "Yes | No",
      "remove_gst_from_final_claim": "Yes | No",
      "over_and_above": "Yes | No",
      "discount_slab_type": "Flat | Quantity_Slab | Value_Slab | Other",
      "best_bet": "Yes | No",
      "brand_support_absolute": null or number,
      "gst_rate": null or number,
      "vendors": [
        {
          "vendor_name": "string",
          "location": "string or null",
          "amount": number or null
        }
      ]
    }
  ]
}
"""


# =============================
# Step 1: Collect Emails from output/
# =============================
def load_emails_from_output(root_dir: str) -> pd.DataFrame:
    email_records = []

    for dirpath, _, filenames in os.walk(root_dir):
        full_txt_files = [f for f in filenames if f.endswith("_full_text.txt")]
        if not full_txt_files:
            continue

        for full_txt in full_txt_files:
            base = full_txt.replace("_full_text.txt", "")
            full_txt_path = os.path.join(dirpath, full_txt)

            with open(full_txt_path, "r", encoding="utf-8", errors="ignore") as f:
                txt_content = f.read()

            # extract subject
            subject = None
            for line in txt_content.splitlines():
                if "Mail - " in line:
                    subject = line.split("Mail - ", 1)[-1].strip()
                    break

            if not subject:
                lines = [l.strip() for l in txt_content.splitlines() if l.strip()]
                if len(lines) >= 3:
                    subject = lines[2]
                elif lines:
                    subject = lines[0]
                else:
                    subject = base

            # collect csv tables
            table_files = [f for f in filenames if f.startswith(base) and f.endswith(".csv")]
            tables_text = ""
            for csv_name in sorted(table_files):
                csv_path = os.path.join(dirpath, csv_name)
                try:
                    df = pd.read_csv(csv_path)
                    tables_text += f"\n\nTABLE FROM {csv_name}\n" + df.to_csv(index=False)
                except:
                    pass

            # read summary.json if exists
            summary_path = os.path.join(dirpath, base + "_summary.json")
            summary_text = ""
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r") as sf:
                        summary_data = json.load(sf)
                    summary_text = "\n\nSUMMARY_JSON:\n" + json.dumps(summary_data, indent=2)
                except:
                    pass

            full_body = txt_content + tables_text + summary_text
            source_file = base + ".pdf"

            email_records.append({
                "mail_subject": subject,
                "mail_body": full_body,
                "sourceFile": source_file
            })

    return pd.DataFrame(email_records)


# =============================
# Step 2: LLM call
# =============================
def call_llm(email_subject: str, email_body: str) -> Dict[str, Any]:
    payload = {
        "mail_subject": email_subject,
        "mail_body": email_body[:12000]
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SCHEME_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)}
        ],
        response_format={"type": "json_object"}
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except:
        data = {"schemes": []}

    if "schemes" not in data:
        data["schemes"] = []

    return data


# =============================
# Step 3: Build scheme_header DF
# =============================
def build_scheme_header(result: Dict[str, Any], source_file: str) -> pd.DataFrame:
    rows = []

    for scheme in result["schemes"]:
        raw_id = f"{scheme.get('scheme_name')}|{scheme.get('duration_start_date')}|{scheme.get('duration_end_date')}"
        scheme_id = hashlib.md5(raw_id.encode()).hexdigest()[:10]

        rows.append({
            "scheme_id": scheme_id,
            "Scheme Type": scheme.get("scheme_type"),
            "Sub Type": scheme.get("scheme_sub_type"),
            "Scheme Name": scheme.get("scheme_name"),
            "Scheme description": scheme.get("scheme_description"),
            "Description": scheme.get("description"),
            "Scheme Period": scheme.get("scheme_period"),
            "Duration_start_date": scheme.get("duration_start_date"),
            "Duration_end_date": scheme.get("duration_end_date"),
            "DISCOUNT_TYPE": scheme.get("discount_type"),
            "GLOBAL_CAP_AMOUNT": scheme.get("global_cap_amount"),
            "Minimum of actual discount OR agreed claim amount": scheme.get("min_actual_or_agreed"),
            "Remove GST from final claim amount": scheme.get("remove_gst_from_final_claim"),
            "Over & Above": scheme.get("over_and_above"),
            "DISCOUNT_SLAB_TYPE": scheme.get("discount_slab_type"),
            "BEST_BET": scheme.get("best_bet"),
            "BRAND_SUPPORT_ABSOLUTE": scheme.get("brand_support_absolute"),
            "GST Rate": scheme.get("gst_rate"),
            "Scheme Document": source_file,
            "FSN File/Config File": None
        })

    return pd.DataFrame(rows)


# =============================
# Main
# =============================
def main():
    print("\n🔍 Scanning extracted output folders…")
    emails_df = load_emails_from_output(OUTPUT_ROOT)

    if emails_df.empty:
        print("⚠ No extracted emails found.")
        return

    all_rows = []

    for idx, row in emails_df.iterrows():
        print(f"\n➡ Processing email {idx+1}/{len(emails_df)}: {row['mail_subject'][:80]}…")
        data = call_llm(row["mail_subject"], row["mail_body"])
        df = build_scheme_header(data, row["sourceFile"])

        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("\n⚠ No scheme headers extracted.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(SCHEME_HEADER_CSV, index=False)

    print(f"\n✅ Done! scheme_header.csv created at: {SCHEME_HEADER_CSV}")


if __name__ == "__main__":
    main()
