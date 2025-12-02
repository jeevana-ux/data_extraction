"""Text extraction implementations using various strategies."""

from pathlib import Path
from typing import Optional
import logging

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdfium2 import PdfDocument
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

from src.extractors.base import BaseTextExtractor

logger = logging.getLogger(__name__)


class PDFPlumberTextExtractor(BaseTextExtractor):
    """Extract text using pdfplumber library."""
    
    @property
    def name(self) -> str:
        return "PDFPlumber Text Extractor"
    
    def extract(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if not HAS_PDFPLUMBER:
            logger.error("pdfplumber not available")
            return ""
        
        try:
            text_parts = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters using pdfplumber")
            return full_text
            
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
            return ""


class OCRTextExtractor(BaseTextExtractor):
    """Extract text using OCR (for image-based PDFs)."""
    
    def __init__(self, dpi: int = 200, lang: str = "eng"):
        """
        Initialize OCR extractor.
        
        Args:
            dpi: Resolution for rendering PDF pages
            lang: Tesseract language code
        """
        self.dpi = dpi
        self.lang = lang
    
    @property
    def name(self) -> str:
        return f"OCR Text Extractor (DPI={self.dpi}, Lang={self.lang})"
    
    def extract(self, pdf_path: Path) -> str:
        """
        Extract text using OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            OCR-extracted text content
        """
        if not HAS_OCR:
            logger.error("OCR dependencies not available")
            return ""
        
        try:
            text_parts = []
            pdf = PdfDocument(str(pdf_path))
            
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                # Render page to image
                bitmap = page.render(scale=self.dpi/72)
                pil_image = bitmap.to_pil()
                
                # Perform OCR
                page_text = pytesseract.image_to_string(pil_image, lang=self.lang)
                
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} (OCR) ---\n{page_text}")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"OCR extracted {len(full_text)} characters from {len(pdf)} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""


def is_text_extractable(pdf_path: Path, min_chars: int = 100) -> bool:
    """
    Check if PDF has extractable text (not image-based).
    
    Args:
        pdf_path: Path to PDF file
        min_chars: Minimum characters to consider as extractable
        
    Returns:
        True if text can be extracted, False if OCR is needed
    """
    if not HAS_PDFPLUMBER:
        return False
    
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            # Check first few pages
            for page in pdf.pages[:3]:
                text = page.extract_text() or ""
                if len(text.strip()) >= min_chars:
                    return True
        return False
    except Exception:
        return False
