"""Main PDF processor orchestrator that coordinates extraction and cleaning."""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging
import hashlib

import pandas as pd

from src.config import ExtractionConfig
from src.models import ExtractionResult, ProcessingMetadata
from src.extractors.text_extractors import (
    PDFPlumberTextExtractor,
    OCRTextExtractor,
    is_text_extractable
)
from src.extractors.table_extractors import (
    PDFPlumberTableExtractor,
    CamelotTableExtractor,
    OCRTableExtractor,
    merge_table_results
)
from src.cleaners import ContentCleaner, TableCleaner

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Main orchestrator for PDF extraction and cleaning.
    
    Coordinates multiple extraction strategies with fallback logic.
    """
    
    def __init__(self, config: ExtractionConfig):
        """
        Initialize PDF processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Initialize extractors
        self.text_extractor = PDFPlumberTextExtractor()
        self.ocr_text_extractor = OCRTextExtractor(
            dpi=config.ocr_dpi,
            lang=config.ocr_language
        )
        
        self.table_extractors = [PDFPlumberTableExtractor()]
        if config.camelot_enabled:
            self.table_extractors.append(CamelotTableExtractor())
        
        self.ocr_table_extractor = OCRTableExtractor(lang=config.ocr_language)
        
        # Initialize cleaners
        self.content_cleaner = ContentCleaner()
        self.table_cleaner = TableCleaner()
    
    def _generate_pdf_id(self, pdf_path: Path) -> str:
        """
        Generate a unique ID for a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Unique identifier string
        """
        # Use filename without extension
        base_name = pdf_path.stem
        
        # Create a short hash for uniqueness
        hash_obj = hashlib.md5(str(pdf_path).encode())
        short_hash = hash_obj.hexdigest()[:8]
        
        return f"{base_name}-{short_hash}"
    
    def _extract_subject(self, text: str) -> Optional[str]:
        """
        Extract email subject from text.
        
        Args:
            text: Full text content
            
        Returns:
            Subject line if found, None otherwise
        """
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Subject:"):
                return line.replace("Subject:", "").strip()
            if "Mail - " in line:
                return line.split("Mail - ", 1)[-1].strip()
        
        # Fallback: use first non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[0] if lines else None
    
    def process(self, pdf_path: Path) -> ExtractionResult:
        """
        Process a PDF file: extract text and tables, then clean.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractionResult with all extracted content
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Extract text
        text = self._extract_text(pdf_path)
        
        # Extract tables
        tables = self._extract_tables(pdf_path)
        
        # Clean content
        cleaned_text = self.content_cleaner.clean_text(text)
        cleaned_tables = self._clean_tables(tables)
        
        # Extract subject
        subject = self._extract_subject(cleaned_text)
        
        # Count pages
        page_count = self._count_pages(pdf_path)
        
        # Create result
        result = ExtractionResult(
            pdf_path=pdf_path,
            full_text=cleaned_text,
            email_subject=subject,
            tables=[self._table_to_dict(t) for t in cleaned_tables],
            page_count=page_count,
            table_count=len(cleaned_tables),
            used_ocr=not is_text_extractable(pdf_path)
        )
        
        logger.info(
            f"Extraction complete: {len(cleaned_text)} chars, "
            f"{len(cleaned_tables)} tables, {page_count} pages"
        )
        
        return result
    
    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text with fallback to OCR if needed."""
        # Try regular extraction first
        text = self.text_extractor.extract(pdf_path)
        
        # If insufficient text and OCR enabled, use OCR
        if len(text.strip()) < 100 and self.config.ocr_enabled:
            logger.info("Text extraction yielded little content, trying OCR")
            text = self.ocr_text_extractor.extract(pdf_path)
        
        return text
    
    def _extract_tables(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extract tables using multiple strategies."""
        all_tables = []
        
        # Try each table extractor
        for extractor in self.table_extractors:
            try:
                tables = extractor.extract(pdf_path)
                all_tables.extend(tables)
            except Exception as e:
                logger.warning(f"{extractor.name} failed: {e}")
        
        # If no tables found and OCR enabled, try OCR
        if not all_tables and self.config.ocr_enabled:
            logger.info("No tables found, trying OCR table extraction")
            try:
                ocr_tables = self.ocr_table_extractor.extract(pdf_path)
                all_tables.extend(ocr_tables)
            except Exception as e:
                logger.warning(f"OCR table extraction failed: {e}")
        
        # Merge and deduplicate
        return merge_table_results(all_tables)
    
    def _clean_tables(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Clean all tables."""
        cleaned = []
        for table in tables:
            cleaned_table = self.table_cleaner.clean(table)
            if cleaned_table is not None:
                cleaned.append(cleaned_table)
        return cleaned
    
    def _table_to_dict(self, df: pd.DataFrame) -> dict:
        """Convert DataFrame to dictionary with metadata."""
        return {
            'data': df.to_dict(orient='records'),
            'columns': df.columns.tolist(),
            'csv_content': df.to_csv(index=False),
            'page': df.attrs.get('page', None),
            'table_index': df.attrs.get('table_index', None),
            'extractor': df.attrs.get('extractor', None),
        }
    
    def _count_pages(self, pdf_path: Path) -> int:
        """Count pages in PDF."""
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                return len(pdf.pages)
        except Exception:
            return 0
    
    def create_metadata(self, pdf_path: Path) -> ProcessingMetadata:
        """
        Create processing metadata for a PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ProcessingMetadata instance
        """
        pdf_id = self._generate_pdf_id(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = self.config.output_dir / pdf_id / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return ProcessingMetadata(
            pdf_id=pdf_id,
            pdf_filename=pdf_path.name,
            output_directory=output_dir
        )
