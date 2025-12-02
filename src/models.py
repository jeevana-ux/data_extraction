"""
Pydantic data models for type-safe data handling throughout the pipeline.

These models provide validation, serialization, and clear contracts between components.
"""

from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ===== Enumerations =====

class SchemeType(str, Enum):
    """Scheme type classification."""
    BUY_SIDE = "BUY_SIDE"
    SELL_SIDE = "SELL_SIDE"
    ONE_OFF = "ONE_OFF"
    OTHER = "OTHER"


class SchemeSubType(str, Enum):
    """Scheme sub-type classification."""
    PERIODIC_CLAIM = "PERIODIC_CLAIM"
    PDC = "PDC"
    PUC_FDC = "PUC_FDC"
    COUPON = "COUPON"
    SUPER_COIN = "SUPER_COIN"
    PREXO = "PREXO"
    BANK_OFFER = "BANK_OFFER"
    LIFESTYLE = "LIFESTYLE"
    ONE_OFF = "ONE_OFF"
    OTHER = "OTHER"


class DiscountType(str, Enum):
    """Discount type classification."""
    PERCENTAGE = "PERCENTAGE"
    FLAT = "FLAT"
    SLAB = "SLAB"
    OTHER = "OTHER"


# ===== Core Data Models =====

class SchemeHeader(BaseModel):
    """
    Represents a single Retailer Hub scheme header.
    
    This is the primary output model for LLM extraction.
    """
    
    model_config = ConfigDict(use_enum_values=True)
    
    # Classification
    scheme_type: SchemeType = Field(
        description="Primary scheme classification"
    )
    scheme_sub_type: SchemeSubType = Field(
        description="Detailed scheme sub-classification"
    )
    scheme_name: str = Field(
        description="Human-readable scheme name"
    )
    
    # Dates
    duration_start_date: Optional[date] = Field(
        default=None,
        description="Scheme duration start date"
    )
    duration_end_date: Optional[date] = Field(
        default=None,
        description="Scheme duration end date"
    )
    starting_at: Optional[date] = Field(
        default=None,
        description="Scheme effective start date"
    )
    ending_at: Optional[date] = Field(
        default=None,
        description="Scheme effective end date"
    )
    price_drop_date: Optional[date] = Field(
        default=None,
        description="Price drop date (PDC only)"
    )
    
    # Financial
    discount_type: Optional[DiscountType] = Field(
        default=None,
        description="Type of discount offered"
    )
    discount_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Discount value (percentage or flat amount)"
    )
    min_order_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum order value for scheme eligibility"
    )
    max_discount_cap: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum discount cap"
    )
    
    # Metadata
    vendor_name: Optional[str] = Field(
        default=None,
        description="Vendor/brand name"
    )
    category: Optional[str] = Field(
        default=None,
        description="Product category"
    )
    remarks: Optional[str] = Field(
        default=None,
        description="Additional notes or remarks"
    )
    
    # Quality metrics
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0-1)"
    )
    needs_escalation: bool = Field(
        default=False,
        description="Whether this extraction needs manual review"
    )
    
    # Source tracking
    source_file: Optional[str] = Field(
        default=None,
        description="Source PDF filename"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of extraction"
    )


class ExtractionResult(BaseModel):
    """
    Result of PDF text and table extraction.
    
    Contains all extracted content from a single PDF.
    """
    
    pdf_path: Path = Field(
        description="Path to the source PDF file"
    )
    
    # Extracted content
    full_text: str = Field(
        default="",
        description="Complete extracted text content"
    )
    
    email_subject: Optional[str] = Field(
        default=None,
        description="Extracted email subject line"
    )
    
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of extracted tables (as dict representations)"
    )
    
    # Metadata
    page_count: int = Field(
        ge=0,
        description="Number of pages in PDF"
    )
    
    table_count: int = Field(
        ge=0,
        description="Number of tables extracted"
    )
    
    used_ocr: bool = Field(
        default=False,
        description="Whether OCR was used for extraction"
    )
    
    extraction_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When extraction was performed"
    )
    
    @property
    def combined_body(self) -> str:
        """Combine text and tables into a single body for LLM processing."""
        body_parts = [self.full_text]
        
        for i, table in enumerate(self.tables, 1):
            body_parts.append(f"\n\nTABLE {i}:\n{table.get('csv_content', '')}")
        
        return "\n".join(body_parts)


class ProcessingMetadata(BaseModel):
    """
    Metadata for tracking processing status and results.
    """
    
    pdf_id: str = Field(
        description="Unique identifier for the PDF"
    )
    
    pdf_filename: str = Field(
        description="Original PDF filename"
    )
    
    output_directory: Path = Field(
        description="Directory where outputs are saved"
    )
    
    processing_started: datetime = Field(
        default_factory=datetime.now
    )
    
    processing_completed: Optional[datetime] = None
    
    success: bool = Field(
        default=False,
        description="Whether processing completed successfully"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    
    schemes_extracted: int = Field(
        default=0,
        ge=0,
        description="Number of schemes extracted"
    )


class LLMResponse(BaseModel):
    """
    Structured response from LLM scheme extraction.
    """
    
    schemes: List[SchemeHeader] = Field(
        default_factory=list,
        description="List of extracted schemes"
    )
    
    raw_response: Optional[str] = Field(
        default=None,
        description="Raw LLM response text"
    )
    
    tokens_used: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tokens consumed"
    )
    
    model_used: Optional[str] = Field(
        default=None,
        description="Model identifier used for extraction"
    )
    
    # Chain-of-Thought tracking
    reasoning: Optional[str] = Field(
        default=None,
        description="Full Chain-of-Thought reasoning trace"
    )
    
    cot_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual CoT reasoning steps with outputs"
    )
    
    @property
    def needs_escalation(self) -> bool:
        """Check if any scheme needs escalation."""
        return any(scheme.needs_escalation for scheme in self.schemes)
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all schemes."""
        if not self.schemes:
            return 0.0
        return sum(s.confidence for s in self.schemes) / len(self.schemes)
