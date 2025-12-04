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
    Comprehensive scheme header model with 21 fields for vendor-Flipkart schemes.
    Based on expert-engineered extraction prompt.
    
    Date Format: DD/MM/YYYY
    All fields mandatory - use null/None if information not available.
    """
    model_config = ConfigDict(use_enum_values=True)
    
    # Core Identification
    scheme_name: Optional[str] = Field(None, description="Scheme name/title (3-10 words)")
    scheme_description: Optional[str] = Field(None, description="1-2 sentence summary of scheme purpose")
    vendor_name: Optional[str] = Field(None, description="Official company/seller name")
    
    # Scheme Classification
    scheme_type: str = Field("OTHER", description="BUY_SIDE | SELL_SIDE | ONE_OFF")
    scheme_subtype: str = Field("OTHER", description="PERIODIC_CLAIM | PDC | PUC/FDC | COUPON | SUPER COIN | PREXO | BANK OFFER | ONE_OFF")
    
    # Temporal Information (DD/MM/YYYY format)
    scheme_period: str = Field("Duration", description="Duration | Event")
    duration: Optional[str] = Field(None, description="DD/MM/YYYY to DD/MM/YYYY")
    start_date: Optional[str] = Field(None, description="Scheme start date (DD/MM/YYYY)")
    end_date: Optional[str] = Field(None, description="Scheme end date (DD/MM/YYYY)")
    price_drop_date: Optional[str] = Field(None, description="PDC effective date (DD/MM/YYYY) or Not Applicable")
    
    # Financial Terms
    discount_type: Optional[str] = Field(None, description="Percentage of NLC | Percentage of MRP | Absolute")
    max_cap: Optional[str] = Field(None, description="Maximum cap amount or Not Specified")
    discount_slab_type: Optional[str] = Field(None, description="Slab structure for PERIODIC_CLAIM or Not Applicable")
    brand_support_absolute: Optional[str] = Field(None, description="Support amount for ONE_OFF or Not Applicable")
    gst_rate: Optional[str] = Field(None, description="GST percentage for ONE_OFF (e.g., 18%) or Not Applicable")
    
    # Conditions and Metadata
    additional_conditions: Optional[str] = Field(None, description="Caps, exclusions, special terms")
    fsn_file_config_file: str = Field("No", description="Yes if FSN/config files present, else No")
    minimum_of_actual_discount_or_agreed_claim: str = Field("No", description="Yes if lesser of actual/agreed applies")
    remove_gst_from_final_claim: Optional[str] = Field(None, description="Yes | No | Not Specified")
    over_and_above: str = Field("No", description="Yes if explicitly mentioned as additional support")
    scheme_document: str = Field("No", description="Yes if formal approval document present")
    best_bet: Optional[str] = Field(None, description="Performance-based incentive for PERIODIC_CLAIM or No")
    
    # Source tracking
    source_file: Optional[str] = Field(
        default=None,
        description="Source PDF filename"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of extraction"
    )
    
    # Legacy fields for backward compatibility (deprecated but kept for now)
    confidence: Optional[float] = Field(0.7, description="Extraction confidence score (0.0-1.0)")
    needs_escalation: Optional[bool] = Field(False, description="Whether scheme needs human review")


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
