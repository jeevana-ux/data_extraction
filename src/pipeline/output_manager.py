"""Output management for saving extraction results and scheme headers."""

import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import pandas as pd

from src.config import ExtractionConfig
from src.models import ExtractionResult, SchemeHeader, ProcessingMetadata

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Manages all output operations for the extraction pipeline.
    
    Handles saving extraction results, scheme headers, and metadata.
    """
    
    def __init__(self, config: ExtractionConfig):
        """
        Initialize output manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def save_extraction_result(
        self,
        result: ExtractionResult,
        metadata: ProcessingMetadata
    ) -> None:
        """
        Save PDF extraction results to output directory.
        
        Args:
            result: Extraction result to save
            metadata: Processing metadata
        """
        output_dir = metadata.output_directory
        pdf_id = metadata.pdf_id
        
        logger.info(f"Saving extraction results to: {output_dir}")
        
        # Save full text
        text_file = output_dir / f"{pdf_id}_full_text.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            # Include subject if available
            if result.email_subject:
                f.write(f"Subject: {result.email_subject}\n\n")
            f.write(result.full_text)
        logger.debug(f"Saved text to: {text_file}")
        
        # Save tables as CSV
        for idx, table_dict in enumerate(result.tables, 1):
            page = table_dict.get('page', '')
            table_idx = table_dict.get('table_index', idx)
            
            if page:
                csv_file = output_dir / f"{pdf_id}_page{page}_table_{table_idx}.csv"
            else:
                csv_file = output_dir / f"{pdf_id}_table_{idx}.csv"
            
            # Write CSV content
            with open(csv_file, "w", encoding="utf-8") as f:
                f.write(table_dict.get('csv_content', ''))
            
            logger.debug(f"Saved table to: {csv_file}")
        
        # Save summary JSON
        summary_file = output_dir / f"{pdf_id}_summary.json"
        summary_data = {
            "pdf_filename": result.pdf_path.name,
            "extraction_timestamp": result.extraction_timestamp.isoformat(),
            "page_count": result.page_count,
            "table_count": result.table_count,
            "used_ocr": result.used_ocr,
            "email_subject": result.email_subject,
            "text_length": len(result.full_text)
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        logger.debug(f"Saved summary to: {summary_file}")
        
        logger.info(f"Extraction results saved: {result.table_count} tables, {len(result.full_text)} chars")
    
    def save_schemes(
        self,
        schemes: List[SchemeHeader],
        output_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Save scheme headers to JSON file.
        
        Args:
            schemes: List of scheme headers
            output_file: Output file path (defaults to config path)
            
        Returns:
            DataFrame of schemes
        """
        if output_file is None:
            output_file = self.config.scheme_header_path
        
        logger.info(f"Saving {len(schemes)} schemes to: {output_file}")
        
        # Convert schemes to list of dictionaries
        scheme_data = []
        for scheme in schemes:
            scheme_dict = {
                "scheme_type": scheme.scheme_type,
                "scheme_sub_type": scheme.scheme_sub_type,
                "scheme_name": scheme.scheme_name,
                "duration_start_date": scheme.duration_start_date.isoformat() if scheme.duration_start_date else None,
                "duration_end_date": scheme.duration_end_date.isoformat() if scheme.duration_end_date else None,
                "starting_at": scheme.starting_at.isoformat() if scheme.starting_at else None,
                "ending_at": scheme.ending_at.isoformat() if scheme.ending_at else None,
                "price_drop_date": scheme.price_drop_date.isoformat() if scheme.price_drop_date else None,
                "discount_type": scheme.discount_type,
                "discount_value": scheme.discount_value,
                "min_order_value": scheme.min_order_value,
                "max_discount_cap": scheme.max_discount_cap,
                "vendor_name": scheme.vendor_name,
                "category": scheme.category,
                "remarks": scheme.remarks,
                "confidence": scheme.confidence,
                "needs_escalation": scheme.needs_escalation,
                "source_file": scheme.source_file,
                "extracted_at": scheme.extracted_at.isoformat(),
            }
            scheme_data.append(scheme_dict)
        
        # Save to JSON with proper formatting
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "schemes": scheme_data,
                "total_count": len(scheme_data),
                "generated_at": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Schemes saved to: {output_file}")
        
        # Also return DataFrame for compatibility
        rows = []
        for scheme in schemes:
            row = {
                "scheme_type": scheme.scheme_type,
                "scheme_sub_type": scheme.scheme_sub_type,
                "scheme_name": scheme.scheme_name,
                "duration_start_date": scheme.duration_start_date,
                "duration_end_date": scheme.duration_end_date,
                "starting_at": scheme.starting_at,
                "ending_at": scheme.ending_at,
                "price_drop_date": scheme.price_drop_date,
                "discount_type": scheme.discount_type,
                "discount_value": scheme.discount_value,
                "min_order_value": scheme.min_order_value,
                "max_discount_cap": scheme.max_discount_cap,
                "vendor_name": scheme.vendor_name,
                "category": scheme.category,
                "remarks": scheme.remarks,
                "confidence": scheme.confidence,
                "needs_escalation": scheme.needs_escalation,
                "source_file": scheme.source_file,
                "extracted_at": scheme.extracted_at,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def load_extracted_emails(self) -> pd.DataFrame:
        """
        Load previously extracted emails from output directory.
        
        Returns:
            DataFrame with columns: mail_subject, mail_body, sourceFile
        """
        logger.info(f"Loading extracted emails from: {self.config.output_dir}")
        
        email_records = []
        
        # Walk through output directory
        for dirpath in Path(self.config.output_dir).rglob("*"):
            if not dirpath.is_dir():
                continue
            
            # Look for full_text files
            text_files = list(dirpath.glob("*_full_text.txt"))
            
            for text_file in text_files:
                # Extract base name
                base = text_file.stem.replace("_full_text", "")
                
                # Read text content
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    txt_content = f.read()
                
                # Extract subject
                subject = self._extract_subject(txt_content, base)
                
                # Collect tables
                table_files = sorted(dirpath.glob(f"{base}*.csv"))
                tables_text = ""
                for csv_file in table_files:
                    try:
                        df = pd.read_csv(csv_file)
                        tables_text += f"\n\nTABLE FROM {csv_file.name}\n{df.to_csv(index=False)}"
                    except Exception as e:
                        logger.warning(f"Failed to read table {csv_file}: {e}")
                
                # Read summary if exists
                summary_file = dirpath / f"{base}_summary.json"
                summary_text = ""
                if summary_file.exists():
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            summary_data = json.load(f)
                        summary_text = f"\n\nSUMMARY:\n{json.dumps(summary_data, indent=2)}"
                    except Exception as e:
                        logger.warning(f"Failed to read summary {summary_file}: {e}")
                
                # Combine all content
                full_body = txt_content + tables_text + summary_text
                source_file = f"{base}.pdf"
                
                email_records.append({
                    "mail_subject": subject,
                    "mail_body": full_body,
                    "sourceFile": source_file
                })
        
        logger.info(f"Loaded {len(email_records)} extracted emails")
        return pd.DataFrame(email_records)
    
    def _extract_subject(self, text: str, fallback: str) -> str:
        """Extract subject from text or use fallback."""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Subject:"):
                return line.replace("Subject:", "").strip()
            if "Mail - " in line:
                return line.split("Mail - ", 1)[-1].strip()
        
        # Fallback: use first non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 3:
            return lines[2]
        elif lines:
            return lines[0]
        else:
            return fallback
