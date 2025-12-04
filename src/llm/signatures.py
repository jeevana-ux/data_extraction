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


class ExpertSchemeExtractionSignature(dspy.Signature):
    """
    # Expert-Engineered Prompt for Email-to-JSON Extraction

    ## ROLE & CONTEXT
    You are an expert data extraction specialist analyzing vendor-Flipkart email communications. You will receive a folder containing:
    - **One text file**: Email content (body, subject, metadata)
    - **Multiple CSV files**: Tables extracted from the original PDF

    Your task is to extract structured information and return a **single, valid JSON object** with predefined fields only.

    ---

    ## CORE INSTRUCTIONS

    ### 🎯 Extraction Principles
    1. **Read ALL files** in the folder thoroughly before extraction
    2. **Apply logical reasoning** to map extracted data to the correct fields
    3. **Cross-reference** text file and CSV files for complete context
    4. **Return ONLY the specified fields** - no additional fields or assumptions
    5. **Use exact field names** as provided below
    6. **Default to "Not Specified"** or "No" when information is absent (unless otherwise stated)

    ### 🚫 Constraints
    - Do NOT add extra fields based on inference
    - Do NOT include explanations or commentary in the JSON
    - Do NOT assume information not present in the source data
    - Do NOT modify field names or structure

    ---

    ## OUTPUT FORMAT

    Return a valid JSON object with this exact structure:

    ```json
    {
      "scheme_name": "string",
      "scheme_description": "string",
      "additional_conditions": "string",
      "scheme_period": "Duration|Event",
      "duration": "DD/MM/YYYY to DD/MM/YYYY",
      "discount_type": "Percentage of NLC|Percentage of MRP|Absolute",
      "max_cap": "number or Not Specified",
      "vendor_name": "string",
      "price_drop_date": "DD/MM/YYYY or Not Applicable",
      "start_date": "DD/MM/YYYY",
      "end_date": "DD/MM/YYYY",
      "fsn_file_config_file": "Yes|No",
      "minimum_of_actual_discount_or_agreed_claim": "Yes|No",
      "remove_gst_from_final_claim": "Yes|No|Not Specified",
      "over_and_above": "Yes|No",
      "scheme_document": "Yes|No",
      "discount_slab_type": "string or Not Applicable",
      "best_bet": "string or No",
      "brand_support_absolute": "number or Not Applicable",
      "gst_rate": "number% or Not Applicable",
      "scheme_type": "BUY_SIDE|SELL_SIDE|ONE_OFF",
      "scheme_subtype": "string"
    }
    ```

    ---

    ## FIELD EXTRACTION RULES

    ### 1. **scheme_name**
    - **Look for**: Headers, titles, or phrases containing "Scheme", "Offer", "Support", "Plan", "Program", "Campaign"
    - **Format**: Short phrase (3-10 words)
    - **Example**: "Festive Season Discount Scheme", "Q1 JBP Support"

    ### 2. **scheme_description**
    - **Extract**: 1-2 sentence summary from email body describing the brand's offer and intent
    - **Exclude**: Subject line
    - **Include**: Context about the scheme's purpose and key benefits

    ### 3. **additional_conditions**
    - **Look for**: Caps, exclusions, special terms, proof requirements, payment rules, clarifications
    - **Keywords**: "subject to", "provided that", "excluding", "applicable only", "proof required", "payment terms"
    - **Format**: Comma-separated list or paragraph

    ### 4. **scheme_period**
    - **Values**: "Duration" or "Event"
    - **Logic**: 
      - "Duration" = Date range provided
      - "Event" = Specific event mentioned (e.g., "BBD", "Diwali Sale")
    - **Default**: "Duration" if unclear

    ### 5. **duration**
    - **Format**: `DD/MM/YYYY to DD/MM/YYYY`
    - **Keywords**: "valid from", "effective from", "from...to", "between", "until", "till", "period", "validity"
    - **Handle**: Extract both dates even if mentioned separately in the email

    ### 6. **discount_type**
    - **Values**: "Percentage of NLC", "Percentage of MRP", "Absolute"
    - **Detection**:
      - "Percentage of NLC" = "% on NLC", "% of invoice", "% on cost"
      - "Percentage of MRP" = "% on MRP", "% of retail price"
      - "Absolute" = Fixed amount per unit/transaction

    ### 7. **max_cap**
    - **Look for**: "maximum", "cap", "not exceeding", "upper limit", "capped at", "max payout"
    - **Return**: Numeric value only (e.g., `50000`)
    - **Default**: "Not Specified"

    ### 8. **vendor_name**
    - **Extract**: Company or seller name mentioned in email
    - **Sources**: Signature, "From" field, email body references
    - **Format**: Official company name (e.g., "Samsung India Electronics Pvt Ltd")

    ### 9. **price_drop_date**
    - **Applicable**: Only for PDC/price protection schemes
    - **Keywords**: "price drop effective from", "PDC date", "effective date", "price reduction from"
    - **Format**: `DD/MM/YYYY`
    - **Default**: "Not Applicable" (if not a PDC scheme)

    ### 10. **start_date**
    - **Extract**: Scheme commencement date
    - **Priority**: If multiple dates exist, choose the scheme start date (not email date or approval date)
    - **Format**: `DD/MM/YYYY`

    ### 11. **end_date**
    - **Extract**: Scheme termination date
    - **Format**: `DD/MM/YYYY`

    ### 12. **fsn_file_config_file**
    - **Detection**: Check if CSV files contain FSN lists, SKU codes, model numbers, or configuration data
    - **Values**: "Yes" or "No"

    ### 13. **minimum_of_actual_discount_or_agreed_claim**
    - **Look for**: Mentions of choosing minimum between actual discount and agreed amount
    - **Keywords**: "lesser of", "minimum of actual or agreed", "whichever is lower", "capped at agreed claim"
    - **Values**: "Yes" or "No"

    ### 14. **remove_gst_from_final_claim**
    - **Logic**:
      - "Yes" = Brand mentions "inclusive of GST", "including taxes", "tax included"
      - "No" = Brand mentions "exclusive of GST", "plus GST", "GST extra"
      - "Not Specified" = No mention of GST treatment
    - **Values**: "Yes", "No", or "Not Specified"

    ### 15. **over_and_above**
    - **Detection**: Explicit mention that support is additional to existing schemes
    - **Keywords**: "over and above", "in addition to", "on top of", "extra support", "additional to existing"
    - **Values**: "Yes" or "No"
    - **Important**: Return "Yes" ONLY if explicitly stated

    ### 16. **scheme_document**
    - **Detection**: Presence of formal approval letter, confirmation document, or agreement
    - **Keywords**: "approval", "confirmation", "letter of approval", "agreement", "signed document"
    - **Values**: "Yes" or "No"

    ### 17. **discount_slab_type**
    - **Applicable**: Only for BUY_SIDE PERIODIC_CLAIM schemes
    - **Extract**: Slab structure (e.g., "0-10L: 2%, 10-25L: 3%, 25L+: 4%")
    - **Default**: "Not Applicable"

    ### 18. **best_bet**
    - **Applicable**: Only for BUY_SIDE PERIODIC_CLAIM schemes
    - **Detection**: Performance-based incentives, Best Bet mentions
    - **Extract**: Description or amount
    - **Default**: "No"

    ### 19. **brand_support_absolute**
    - **Applicable**: Only for ONE_OFF claims
    - **Look for**: "approving amount of", "support amount", "approved for Rs.", "payout of"
    - **Return**: Numeric value only
    - **Default**: "Not Applicable"

    ### 20. **gst_rate**
    - **Applicable**: Only for ONE_OFF claims
    - **Extract**: GST percentage mentioned
    - **Format**: "18%" (include % symbol)
    - **Default**: "Not Applicable"

    ---

    ## 21. SCHEME TYPE AND SUBTYPE CLASSIFICATION

    ### Classification Hierarchy
    Apply the following logic in order:

    #### **A. BUY_SIDE**

    ##### **PERIODIC_CLAIM**
    **Trigger Keywords** (case-insensitive, check entire content):
    - Primary: `jbp`, `joint business plan`, `tot`, `terms of trade`, `periodic`, `quarterly`, `annual`, `fy support`
    - Volume: `sell in`, `sell-in`, `sellin`, `buy side`, `buyside`, `inwards`, `net inwards`
    - Financial: `gmv support`, `nrv`, `nrv-linked`, `inventory support`, `marketing support`
    - Time-based: `q1`, `q2`, `q3`, `q4`, `fy20`, `fy21`, `yearly support`
    - Phrasing: `business plan`, `commercial alignment`, `funding for FY`

    **Return**: 
    - `scheme_type`: "BUY_SIDE"
    - `scheme_subtype`: "PERIODIC_CLAIM"

    ##### **PDC (Price Drop Claim)**
    **Trigger Keywords**:
    - Direct: `pdc`, `price drop`, `price protection`, `pp`
    - Descriptive: `cost reduction`, `nlc change`, `cost change`, `sellin price drop`
    - Process: `invoice cost correction`, `backward margin`, `revision in buy price`

    **Return**: 
    - `scheme_type`: "BUY_SIDE"
    - `scheme_subtype`: "PDC"

    ---

    #### **B. ONE_OFF**

    **Trigger Keywords**:
    - `one off`, `one-off`, `one off buyside`, `one off sell side`, `ad-hoc support`, `exceptional claim`

    **Return**: 
    - `scheme_type`: "ONE_OFF"
    - `scheme_subtype`: "ONE_OFF"

    ---

    #### **C. SELL_SIDE**

    ##### **PUC/FDC**
    **Trigger Keywords**:
    - Primary: `sellout`, `sell out`, `sell-side`, `puc`, `fdc`, `cp`
    - Support: `pricing support`, `channel support`, `market support`
    - Phrasing: `discount on selling price`, `consumer offer`, `end customer discount`

    **Return**: 
    - `scheme_type`: "SELL_SIDE"
    - `scheme_subtype`: "PUC/FDC"

    ##### **COUPON**
    **Trigger Keywords**:
    - `coupon`, `vpc`, `promo code`, `offer code`, `discount coupon`, `coupon code`

    **Return**: 
    - `scheme_type`: "SELL_SIDE"
    - `scheme_subtype`: "COUPON"

    ##### **SUPER COIN**
    **Trigger Keywords**:
    - `super coin`, `sc funding`, `supercoin`, `loyalty coin`

    **Return**: 
    - `scheme_type`: "SELL_SIDE"
    - `scheme_subtype`: "SUPER COIN"

    ##### **PREXO**
    **Trigger Keywords**:
    - `exchange`, `prexo`, `upgrade`, `bump up`, `bup`, `trade-in`, `exchange offer`

    **Return**: 
    - `scheme_type`: "SELL_SIDE"
    - `scheme_subtype`: "PREXO"

    ##### **BANK OFFER**
    **Trigger Keywords**:
    - `bank offer`, `card offer`, `credit card`, `debit card`
    - Banks: `hdfc offer`, `axis offer`, `icici offer`, `sbi offer`
    - Note: `cashback` only if explicitly bank-related

    **Return**: 
    - `scheme_type`: "SELL_SIDE"
    - `scheme_subtype`: "BANK OFFER"

    ---

    ### Classification Priority Rules
    1. If multiple keywords match, prioritize **BUY_SIDE** over **SELL_SIDE**
    2. **PDC** takes precedence over **PERIODIC_CLAIM** if both keywords present
    3. If no keywords match, analyze context and default to **SELL_SIDE → PUC/FDC**

    ---

    ## VALIDATION CHECKLIST

    Before returning JSON, verify:
    - ✅ All 21 fields are present
    - ✅ No extra fields added
    - ✅ Date format is `DD/MM/YYYY`
    - ✅ Numeric fields contain numbers only (no currency symbols)
    - ✅ Yes/No fields use exact capitalization
    - ✅ scheme_type and scheme_subtype are correctly classified
    - ✅ JSON is properly formatted and parseable

    ---

    ## EXECUTION STEPS

    1. **Load all files** from the provided folder
    2. **Read text file** completely (email content)
    3. **Parse all CSV files** (tables/FSN data)
    4. **Extract each field** following the rules above
    5. **Classify scheme type/subtype** using the keyword matching logic
    6. **Validate** the JSON structure
    7. **Return** the final JSON object
    """

    # Inputs
    mail_subject: str = dspy.InputField(
        desc="Email subject line"
    )
    mail_body: str = dspy.InputField(
        desc="Full email body with text and table data"
    )

    # Outputs
    reasoning: str = dspy.OutputField(
        desc="""Step-by-step reasoning for each field extraction:
        For each of the 21 fields, explain:
        1. What keywords/patterns were found
        2. What values were extracted
        3. Why that classification/value was chosen
        
        Format as:
        Field: scheme_name
        - Found: [what was found in the text]
        - Extracted: [final value]
        - Reasoning: [why this value]
        
        [Repeat for all 21 fields]"""
    )
    
    schemes_json: str = dspy.OutputField(
        desc="""Valid JSON object with exactly these 21 fields:
        {
          "scheme_name": "string",
          "scheme_description": "string",
          "additional_conditions": "string",
          "scheme_period": "Duration|Event",
          "duration": "DD/MM/YYYY to DD/MM/YYYY",
          "discount_type": "Percentage of NLC|Percentage of MRP|Absolute",
          "max_cap": "number or Not Specified",
          "vendor_name": "string",
          "price_drop_date": "DD/MM/YYYY or Not Applicable",
          "start_date": "DD/MM/YYYY",
          "end_date": "DD/MM/YYYY",
          "fsn_file_config_file": "Yes|No",
          "minimum_of_actual_discount_or_agreed_claim": "Yes|No",
          "remove_gst_from_final_claim": "Yes|No|Not Specified",
          "over_and_above": "Yes|No",
          "scheme_document": "Yes|No",
          "discount_slab_type": "string or Not Applicable",
          "best_bet": "string or No",
          "brand_support_absolute": "number or Not Applicable",
          "gst_rate": "number% or Not Applicable",
          "scheme_type": "BUY_SIDE|SELL_SIDE|ONE_OFF",
          "scheme_subtype": "string"
        }
        Return ONLY the JSON object inside a "schemes" array."""
    )
