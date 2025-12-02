"""Table extraction implementations using various strategies."""

from pathlib import Path
from typing import List
import logging

import pandas as pd

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False

try:
    from img2table.document import PDF
    from img2table.ocr import TesseractOCR
    HAS_IMG2TABLE = True
except ImportError:
    HAS_IMG2TABLE = False

from src.extractors.base import BaseTableExtractor

logger = logging.getLogger(__name__)


class PDFPlumberTableExtractor(BaseTableExtractor):
    """Extract tables using pdfplumber library."""
    
    @property
    def name(self) -> str:
        return "PDFPlumber Table Extractor"
    
    def extract(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Extract tables from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DataFrames, one per table
        """
        if not HAS_PDFPLUMBER:
            logger.error("pdfplumber not available")
            return []
        
        try:
            all_tables = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            # Add metadata
                            df.attrs['page'] = page_num
                            df.attrs['table_index'] = table_idx
                            df.attrs['extractor'] = self.name
                            
                            all_tables.append(df)
            
            logger.info(f"PDFPlumber extracted {len(all_tables)} tables")
            return all_tables
            
        except Exception as e:
            logger.error(f"PDFPlumber table extraction failed: {e}")
            return []


class CamelotTableExtractor(BaseTableExtractor):
    """Extract tables using Camelot library (better for complex tables)."""
    
    def __init__(self, flavor: str = "lattice"):
        """
        Initialize Camelot extractor.
        
        Args:
            flavor: 'lattice' for bordered tables, 'stream' for borderless
        """
        self.flavor = flavor
    
    @property
    def name(self) -> str:
        return f"Camelot Table Extractor ({self.flavor})"
    
    def extract(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Extract tables from PDF using Camelot.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DataFrames, one per table
        """
        if not HAS_CAMELOT:
            logger.error("Camelot not available")
            return []
        
        try:
            # Try lattice first (for bordered tables)
            tables = camelot.read_pdf(
                str(pdf_path),
                pages='all',
                flavor=self.flavor,
                suppress_stdout=True
            )
            
            all_tables = []
            for idx, table in enumerate(tables):
                df = table.df
                
                # Add metadata
                df.attrs['page'] = table.page
                df.attrs['table_index'] = idx
                df.attrs['extractor'] = self.name
                df.attrs['accuracy'] = table.accuracy
                
                all_tables.append(df)
            
            logger.info(f"Camelot extracted {len(all_tables)} tables")
            return all_tables
            
        except Exception as e:
            logger.error(f"Camelot table extraction failed: {e}")
            return []


class OCRTableExtractor(BaseTableExtractor):
    """Extract tables using OCR (for image-based PDFs)."""
    
    def __init__(self, lang: str = "eng"):
        """
        Initialize OCR table extractor.
        
        Args:
            lang: Tesseract language code
        """
        self.lang = lang
    
    @property
    def name(self) -> str:
        return f"OCR Table Extractor (Lang={self.lang})"
    
    def extract(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Extract tables using OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DataFrames, one per table
        """
        if not HAS_IMG2TABLE:
            logger.error("img2table not available")
            return []
        
        try:
            # Initialize OCR
            ocr = TesseractOCR(lang=self.lang)
            
            # Extract tables
            pdf_doc = PDF(str(pdf_path))
            extracted_tables = pdf_doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=True
            )
            
            all_tables = []
            for page_num, page_tables in enumerate(extracted_tables, 1):
                for table_idx, table in enumerate(page_tables):
                    df = table.df
                    
                    # Add metadata
                    df.attrs['page'] = page_num
                    df.attrs['table_index'] = table_idx
                    df.attrs['extractor'] = self.name
                    
                    all_tables.append(df)
            
            logger.info(f"OCR extracted {len(all_tables)} tables")
            return all_tables
            
        except Exception as e:
            logger.error(f"OCR table extraction failed: {e}")
            return []


def merge_table_results(
    *table_lists: List[pd.DataFrame]
) -> List[pd.DataFrame]:
    """
    Merge results from multiple table extractors, removing duplicates.
    
    Args:
        *table_lists: Variable number of table lists from different extractors
        
    Returns:
        Deduplicated list of tables
    """
    all_tables = []
    seen_shapes = set()
    
    for table_list in table_lists:
        for table in table_list:
            # Use shape and first few cells as fingerprint
            fingerprint = (
                table.shape,
                str(table.iloc[0, 0] if not table.empty else ""),
                str(table.iloc[-1, -1] if not table.empty else "")
            )
            
            if fingerprint not in seen_shapes:
                seen_shapes.add(fingerprint)
                all_tables.append(table)
    
    return all_tables
