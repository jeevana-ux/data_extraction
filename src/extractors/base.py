"""Base classes and interfaces for PDF extraction components."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.models import ExtractionResult


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    
    @abstractmethod
    def extract(self, pdf_path: Path) -> any:
        """
        Extract content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted content (type depends on extractor)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this extractor."""
        pass


class BaseTextExtractor(BaseExtractor):
    """Base class for text extraction strategies."""
    
    @abstractmethod
    def extract(self, pdf_path: Path) -> str:
        """
        Extract text content from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        pass


class BaseTableExtractor(BaseExtractor):
    """Base class for table extraction strategies."""
    
    @abstractmethod
    def extract(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Extract tables from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of pandas DataFrames, one per table
        """
        pass


class BaseContentCleaner(ABC):
    """Base class for content cleaning strategies."""
    
    @abstractmethod
    def clean(self, content: any) -> any:
        """
        Clean content (remove disclaimers, noise, etc.).
        
        Args:
            content: Content to clean (str or DataFrame)
            
        Returns:
            Cleaned content
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this cleaner."""
        pass
