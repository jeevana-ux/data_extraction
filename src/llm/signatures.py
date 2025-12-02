"""DSPy signatures for scheme extraction.

Defines type-safe input/output contracts for the extraction pipeline.
"""

import dspy
from typing import Optional


class SchemeExtractionSignature(dspy.Signature):
    """Extract Retailer Hub scheme headers from email content.
    
    This is the main signature for the full extraction process.
    Uses Chain-of-Thought to reason through the extraction steps.
    """
    
    # Inputs
    mail_subject: str = dspy.InputField(
        desc="Email subject line containing scheme information"
    )
    mail_body: str = dspy.InputField(
        desc="Email body with extracted text, tables, and CSV content"
    )
    
    # Outputs
    schemes_json: str = dspy.OutputField(
        desc="JSON array of extracted schemes following the Retailer Hub schema"
    )
    reasoning: str = dspy.OutputField(
        desc="Chain of thought reasoning explaining the extraction decisions"
    )


class SchemeClassificationSignature(dspy.Signature):
    """Classify scheme type and subtype based on content analysis.
    
    Determines whether a scheme is BUY_SIDE, SELL_SIDE, ONE_OFF, etc.
    and assigns the appropriate sub-classification.
    """
    
    # Inputs
    scheme_description: str = dspy.InputField(
        desc="Description or summary of the scheme"
    )
    keywords: str = dspy.InputField(
        desc="Extracted keywords and phrases from the email"
    )
    content_context: str = dspy.InputField(
        desc="Full context from email for reference"
    )
    
    # Outputs
    scheme_type: str = dspy.OutputField(
        desc="Primary classification: BUY_SIDE, SELL_SIDE, ONE_OFF, or OTHER"
    )
    scheme_sub_type: str = dspy.OutputField(
        desc="Sub-classification: PERIODIC_CLAIM, PDC, PUC_FDC, COUPON, etc."
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0 for this classification"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of why this classification was chosen"
    )


class DateExtractionSignature(dspy.Signature):
    """Extract and normalize dates from unstructured text.
    
    Identifies scheme duration dates, starting dates, ending dates,
    and price drop dates. Normalizes to YYYY-MM-DD format.
    """
    
    # Inputs
    text_content: str = dspy.InputField(
        desc="Text content containing date information"
    )
    context: str = dspy.InputField(
        desc="Additional context to resolve ambiguous dates"
    )
    
    # Outputs
    duration_start_date: Optional[str] = dspy.OutputField(
        desc="Scheme duration start date in YYYY-MM-DD format or null"
    )
    duration_end_date: Optional[str] = dspy.OutputField(
        desc="Scheme duration end date in YYYY-MM-DD format or null"
    )
    starting_at: Optional[str] = dspy.OutputField(
        desc="Scheme effective start date in YYYY-MM-DD format or null"
    )
    ending_at: Optional[str] = dspy.OutputField(
        desc="Scheme effective end date in YYYY-MM-DD format or null"
    )
    price_drop_date: Optional[str] = dspy.OutputField(
        desc="Price drop date (PDC only) in YYYY-MM-DD format or null"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of date extraction and normalization logic"
    )


class FinancialExtractionSignature(dspy.Signature):
    """Extract financial information from scheme description.
    
    Identifies discount types, amounts, caps, GST rates, and other
    financial parameters.
    """
    
    # Inputs
    text_content: str = dspy.InputField(
        desc="Text content containing financial information"
    )
    table_data: str = dspy.InputField(
        desc="CSV table data with financial details"
    )
    
    # Outputs
    discount_type: Optional[str] = dspy.OutputField(
        desc="PERCENTAGE, FLAT, SLAB, or OTHER"
    )
    discount_value: Optional[float] = dspy.OutputField(
        desc="Discount value as a number (percentage or flat amount)"
    )
    min_order_value: Optional[float] = dspy.OutputField(
        desc="Minimum order value for scheme eligibility"
    )
    max_discount_cap: Optional[float] = dspy.OutputField(
        desc="Maximum discount cap amount"
    )
    gst_rate: Optional[float] = dspy.OutputField(
        desc="GST rate as a percentage"
    )
    brand_support_absolute: Optional[float] = dspy.OutputField(
        desc="Absolute brand support amount"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of financial data extraction"
    )


class VendorExtractionSignature(dspy.Signature):
    """Extract vendor information from tables and text.
    
    Identifies vendor names, locations, and associated amounts
    from structured and unstructured data.
    """
    
    # Inputs
    table_data: str = dspy.InputField(
        desc="CSV table data containing vendor information"
    )
    text_content: str = dspy.InputField(
        desc="Text content that may mention vendors"
    )
    
    # Outputs
    vendors_json: str = dspy.OutputField(
        desc="JSON array of vendors with name, location, and amount fields"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of vendor extraction logic"
    )


class KeyFactsExtractionSignature(dspy.Signature):
    """Extract key facts and entities from email content.
    
    First step in the CoT pipeline - identifies important information
    before deeper analysis.
    """
    
    # Inputs
    mail_subject: str = dspy.InputField(desc="Email subject line")
    mail_body: str = dspy.InputField(desc="Email body content")
    
    # Outputs
    scheme_names: str = dspy.OutputField(
        desc="Comma-separated list of identified scheme names"
    )
    key_dates: str = dspy.OutputField(
        desc="All dates mentioned in the content"
    )
    key_amounts: str = dspy.OutputField(
        desc="All monetary amounts mentioned"
    )
    keywords: str = dspy.OutputField(
        desc="Important keywords for classification (JBP, PDC, sellout, etc.)"
    )
    vendor_names: str = dspy.OutputField(
        desc="Vendor or brand names mentioned"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of key facts identification"
    )


class ConfidenceAssessmentSignature(dspy.Signature):
    """Assess extraction confidence and determine if escalation is needed.
    
    Final step that evaluates the quality of extracted data and decides
    whether manual review is required.
    """
    
    # Inputs
    extracted_data: str = dspy.InputField(
        desc="JSON of the extracted scheme data"
    )
    extraction_reasoning: str = dspy.InputField(
        desc="Reasoning trace from all previous steps"
    )
    
    # Outputs
    confidence_score: float = dspy.OutputField(
        desc="Overall confidence score between 0.0 and 1.0"
    )
    needs_escalation: bool = dspy.OutputField(
        desc="Whether this extraction needs manual review (true/false)"
    )
    missing_fields: str = dspy.OutputField(
        desc="Comma-separated list of critical fields that are missing or uncertain"
    )
    quality_issues: str = dspy.OutputField(
        desc="Description of any quality concerns with the extraction"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of confidence assessment"
    )
