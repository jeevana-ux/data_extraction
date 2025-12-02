"""DSPy modules and signatures for scheme extraction."""

import json
import logging
from typing import Dict, Any, List, Optional

import dspy
from pydantic import ValidationError

from src.models import SchemeHeader, LLMResponse
from src.config import ExtractionConfig

logger = logging.getLogger(__name__)


# ===== System Prompt (from original build_scheme_header.py) =====

SCHEME_SYSTEM_PROMPT = """You are an expert Retailer Hub scheme-header extractor for Flipkart's internal Retailer Hub system. 
You convert unstructured brand emails + extracted text + CSV tables into STRICT Retailer Hub JSON.

Your priorities IN ORDER:
1. ACCURACY over creativity  
2. DO NOT GUESS — never invent data  
3. Follow all rules EXACTLY as written  
4. Output STRICT JSON ONLY  
5. Null fields instead of assumptions  
6. Flag incomplete data via confidence scoring  
7. Extract MULTIPLE schemes when applicable  
8. Maintain absolute schema integrity  

You help prepare Flipkart Retailer Hub scheme headers from brand emails.

You will be given:
- email_subject
- email_body (plain text extracted from PDF/email)
- table_data (CSV content pasted as text if available)

Your job:
1. Read all components together (subject + body + tables).
2. Identify every scheme/claim described.
3. Produce EXACT Retailer Hub JSON as per schema below.
4. IF ANY FIELD IS UNCLEAR → USE null.
5. NEVER guess numeric values, dates, or classifications.
6. STRICTLY follow the classification rules & mapping rules.
7. Provide a confidence score for each scheme (0–1).
8. If ANY critical field is missing or ambiguous, set:
     "needs_escalation": true
   This allows the pipeline to re-run with a stronger model.

========================================================
ABSOLUTELY CRITICAL OUTPUT RULES
========================================================
- Your response MUST be ONLY JSON.
- No markdown, no commentary, no backticks, no text.
- JSON MUST match the schema EXACTLY.
- If parsing fails, the entire output is rejected.
- Vendors array MUST always exist (empty list if none).
- Dates MUST be YYYY-MM-DD or null.
- Numbers MUST be numbers or null (no commas or symbols).
- Strings must never contain trailing spaces or formatting.

========================================================
OUTPUT JSON FORMAT (STRICT, ENHANCED VERSION)
========================================================
{
  "schemes": [
    {
      "scheme_type": "BUY_SIDE | SELL_SIDE | ONE_OFF | OTHER",
      "scheme_sub_type": "PERIODIC_CLAIM | PDC | PUC_FDC | COUPON | SUPER_COIN | PREXO | BANK_OFFER | LIFESTYLE | ONE_OFF | OTHER",
      "scheme_name": "string",
      "scheme_description": "string",
      "description": "string",
      "scheme_period": "EVENT | DURATION",
      "duration_start_date": "YYYY-MM-DD or null",
      "duration_end_date": "YYYY-MM-DD or null",
      "discount_type": "Percentage of MRP | Percentage of NLC | Absolute | Other",
      "global_cap_amount": number or null,
      "min_actual_or_agreed": "Yes | No",
      "remove_gst_from_final_claim": "Yes | No",
      "over_and_above": "Yes | No",
      "discount_slab_type": "Flat | Quantity_Slab | Value_Slab | Other",
      "best_bet": "Yes | No",
      "brand_support_absolute": number or null,
      "gst_rate": number or null,
      "price_drop_date": "YYYY-MM-DD or null",
      "starting_at": "YYYY-MM-DD or null",
      "ending_at": "YYYY-MM-DD or null",
      "vendors": [
        {
          "vendor_name": "string",
          "location": "string or null",
          "amount": number or null
        }
      ],
      "confidence": number between 0 and 1,
      "needs_escalation": true or false
    }
  ]
}

========================================================
STRICT CLASSIFICATION RULES (REWRITE FOR ACCURACY)
========================================================

Follow EXACT matching rules below. Never override based on intuition.

### BUY_SIDE – PERIODIC_CLAIM
Trigger when dominated by:
"jbp", "joint business plan", "tot", "sell in", "inwards", 
"periodic", "quarter", "q1/q2/q3/q4", "annual", "fy", 
"business plan", "nrv support", "inventory support"

### BUY_SIDE – PDC (Price Drop Claim)
Trigger on:
"price drop", "price protection", "pp", "pdc",
"nlc change", "cost reduction", "invoice cost correction"

### SELL_SIDE – PUC_FDC
Trigger when dominated by:
"sellout", "puc", "cp", "fdc", "pricing support",
"channel support", "market support"

### SELL_SIDE – COUPON
Trigger:
"coupon", "vpc", "promo code", "offer code"

### SUPER COIN
Trigger:
"super coin", "sc funding"

### PREXO
Trigger:
"exchange", "prexo", "upgrade", "bump up"

### BANK OFFER
Trigger:
"bank offer", "bank cashback", 
"HDFC", "SBI", "ICICI", "Axis", "Kotak" etc.

### LIFESTYLE
Trigger:
"lifestyle" AND not fitting above categories.

### ONE_OFF
Trigger:
"one-off", "one time", "lump sum", "one-off sales support"

### OTHERWISE
scheme_type = "OTHER"
scheme_sub_type = "OTHER"

========================================================
DATE RULES (STRICT)
========================================================
- If month/year only → use 1st & last day.
- If no dates → duration_start_date = duration_end_date = null.
- starting_at = duration_start_date
- ending_at = duration_end_date
- price_drop_date ONLY for PDC; otherwise null.

========================================================
FINANCIAL RULES (STRICT)
========================================================
- discount_type: EXACT logic must be followed.
- global_cap_amount: extract only explicit caps.
- min_actual_or_agreed = "Yes" if cap exists else "No".
- brand_support_absolute: ONLY when one-off or explicitly stated amount.
- remove_gst_from_final_claim: 
    default = "No"
    if "inclusive of GST" → "Yes"
- gst_rate: extract % only if explicitly mentioned.

========================================================
VENDOR RULES (STRICT)
========================================================
- Parse vendor tables from CSV text.
- If location missing → null.
- If amount missing → null.
- If vendor list absent → empty array [].

========================================================
CONFLICT & UNCERTAINTY HANDLING
========================================================
NEVER guess conflicting information.

If ANY of the following is true:
- required field missing,
- dates unclear,
- classification uncertain,
- amounts inconsistent,
- no scheme clearly detected,

THEN:
  "needs_escalation": true  
  confidence should be < 0.75

Otherwise:
  "needs_escalation": false

========================================================
FINAL INSTRUCTIONS
========================================================
READ ALL PROVIDED DATA CAREFULLY:
- email_subject
- email_body (extracted text)
- table_data (CSV extracted text)

Return the JSON EXACTLY as per schema. No text.
If no scheme is present, return:
{ "schemes": [] }
"""


class SchemeExtractor:
    """
    DSPy-based scheme extractor.
    
    Uses OpenRouter LLM to extract scheme headers from email content.
    """
    
    def __init__(self, llm: dspy.LM, config: ExtractionConfig):
        """
        Initialize scheme extractor.
        
        Args:
            llm: DSPy LM instance (OpenRouterLLM)
            config: Application configuration
        """
        self.llm = llm
        self.config = config
        
        # Set as default LM for DSPy
        dspy.settings.configure(lm=llm)
    
    def extract(
        self,
        email_subject: str,
        email_body: str
    ) -> LLMResponse:
        """
        Extract scheme headers from email content.
        
        Args:
            email_subject: Email subject line
            email_body: Full email body with tables
            
        Returns:
            LLMResponse with extracted schemes
        """
        logger.info(f"Extracting schemes from: {email_subject[:80]}")
        
        # Truncate body to avoid token limits
        truncated_body = email_body[:12000]
        
        # Prepare user message
        user_content = json.dumps({
            "mail_subject": email_subject,
            "mail_body": truncated_body
        })
        
        # Call LLM
        try:
            messages = [
                {"role": "system", "content": SCHEME_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            responses = self.llm(messages=messages)
            raw_response = responses[0] if responses else ""
            
            logger.debug(f"Raw LLM response: {raw_response[:200]}...")
            
            # Parse JSON response
            schemes = self._parse_response(raw_response)
            
            # Get token usage
            usage_stats = self.llm.get_usage_stats()
            
            return LLMResponse(
                schemes=schemes,
                raw_response=raw_response,
                tokens_used=usage_stats.get("total_tokens"),
                model_used=self.llm.model_name
            )
            
        except Exception as e:
            logger.error(f"Scheme extraction failed: {e}")
            return LLMResponse(
                schemes=[],
                raw_response=str(e),
                model_used=self.llm.model_name
            )
    
    def _parse_response(self, response: str) -> List[SchemeHeader]:
        """
        Parse LLM JSON response into SchemeHeader objects.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            List of SchemeHeader instances
        """
        try:
            # Clean response (remove markdown if present)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Parse JSON
            data = json.loads(cleaned)
            
            if "schemes" not in data:
                logger.warning("Response missing 'schemes' key")
                return []
            
            schemes = []
            for scheme_data in data["schemes"]:
                try:
                    # Map fields to our Pydantic model
                    scheme = self._map_to_scheme_header(scheme_data)
                    schemes.append(scheme)
                except ValidationError as e:
                    logger.warning(f"Scheme validation failed: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(schemes)} schemes")
            return schemes
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return []
    
    def _map_to_scheme_header(self, data: Dict[str, Any]) -> SchemeHeader:
        """
        Map raw LLM data to SchemeHeader model.
        
        Args:
            data: Raw scheme data from LLM
            
        Returns:
            SchemeHeader instance
        """
        # Map discount_type to our enum
        discount_type_map = {
            "Percentage of MRP": "PERCENTAGE",
            "Percentage of NLC": "PERCENTAGE",
            "Absolute": "FLAT",
            "Other": "OTHER"
        }
        
        discount_type = data.get("discount_type")
        if discount_type:
            discount_type = discount_type_map.get(discount_type, "OTHER")
        
        return SchemeHeader(
            scheme_type=data.get("scheme_type", "OTHER"),
            scheme_sub_type=data.get("scheme_sub_type", "OTHER"),
            scheme_name=data.get("scheme_name", ""),
            duration_start_date=data.get("duration_start_date"),
            duration_end_date=data.get("duration_end_date"),
            starting_at=data.get("starting_at"),
            ending_at=data.get("ending_at"),
            price_drop_date=data.get("price_drop_date"),
            discount_type=discount_type,
            discount_value=data.get("global_cap_amount"),  # Simplified mapping
            min_order_value=None,  # Not in original schema
            max_discount_cap=data.get("global_cap_amount"),
            vendor_name=None,  # Extract from vendors array if needed
            category=None,
            remarks=data.get("description"),
            confidence=data.get("confidence", 0.5),
            needs_escalation=data.get("needs_escalation", False)
        )
