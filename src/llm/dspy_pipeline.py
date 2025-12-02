"""DSPy Chain-of-Thought pipeline for scheme extraction.

Implements a multi-step reasoning process for extracting Retailer Hub
scheme headers from email content.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import dspy
from pydantic import ValidationError

from src.llm.signatures import (
    SchemeExtractionSignature,
    SchemeClassificationSignature,
    DateExtractionSignature,
    FinancialExtractionSignature,
    VendorExtractionSignature,
    KeyFactsExtractionSignature,
    ConfidenceAssessmentSignature
)
from src.models import SchemeHeader, LLMResponse
from src.config import ExtractionConfig

logger = logging.getLogger(__name__)


# System prompt for scheme extraction context
SCHEME_EXTRACTION_CONTEXT = """You are an expert Retailer Hub scheme-header extractor for Flipkart.
You convert unstructured brand emails into structured Retailer Hub JSON.

CRITICAL RULES:
1. ACCURACY over creativity - never guess or invent data
2. Use null for missing/unclear fields
3. Extract MULTIPLE schemes when present
4. Follow classification rules exactly
5. Dates must be YYYY-MM-DD or null
6. Set needs_escalation=true if ANY critical field is uncertain

Scheme Types:
- BUY_SIDE: JBP, TOT, sell-in, periodic claims, inventory support
- SELL_SIDE: Sellout, PUC, FDC, channel support, pricing support
- ONE_OFF: One-time lump sum payments
- OTHER: Doesn't fit above categories

Sub-types:
- PERIODIC_CLAIM: JBP, quarterly, annual business plans
- PDC: Price drop, price protection, NLC changes
- PUC_FDC: Sellout support, channel pricing
- COUPON: Coupon codes, VPC, promo codes
- SUPER_COIN: Super coin funding
- PREXO: Exchange, upgrade programs
- BANK_OFFER: Bank cashback, card offers
- LIFESTYLE: Lifestyle category schemes
- ONE_OFF: One-time support
- OTHER: Doesn't fit above
"""


class SchemeExtractionCoT(dspy.Module):
    """Chain-of-Thought module for comprehensive scheme extraction.
    
    Implements a multi-step reasoning process:
    1. Extract key facts (dates, amounts, vendors, keywords)
    2. Classify scheme types based on keywords
    3. Extract dates and normalize formats
    4. Extract financial information
    5. Extract vendor details
    6. Assemble complete JSON
    7. Assess confidence and escalation needs
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize CoT modules for each step
        self.extract_key_facts = dspy.ChainOfThought(KeyFactsExtractionSignature)
        self.classify_scheme = dspy.ChainOfThought(SchemeClassificationSignature)
        self.extract_dates = dspy.ChainOfThought(DateExtractionSignature)
        self.extract_financials = dspy.ChainOfThought(FinancialExtractionSignature)
        self.extract_vendors = dspy.ChainOfThought(VendorExtractionSignature)
        self.assess_confidence = dspy.ChainOfThought(ConfidenceAssessmentSignature)
        
        # Main extractor with full context
        self.extract_schemes = dspy.ChainOfThought(SchemeExtractionSignature)
    
    def forward(self, mail_subject: str, mail_body: str) -> Dict[str, Any]:
        """Execute the full Chain-of-Thought extraction pipeline.
        
        Args:
            mail_subject: Email subject line
            mail_body: Email body with text and tables
            
        Returns:
            Dictionary with schemes_json, reasoning, and cot_steps
        """
        cot_steps = []
        
        try:
            # Step 1: Extract key facts
            logger.debug("CoT Step 1: Extracting key facts")
            facts = self.extract_key_facts(
                mail_subject=mail_subject,
                mail_body=mail_body[:8000]  # Truncate to avoid token limits
            )
            
            cot_steps.append({
                "step": "1_key_facts",
                "output": {
                    "scheme_names": facts.scheme_names,
                    "key_dates": facts.key_dates,
                    "key_amounts": facts.key_amounts,
                    "keywords": facts.keywords,
                    "vendor_names": facts.vendor_names
                },
                "reasoning": facts.reasoning
            })
            
            # Step 2: Classify scheme type
            logger.debug("CoT Step 2: Classifying scheme")
            classification = self.classify_scheme(
                scheme_description=facts.scheme_names,
                keywords=facts.keywords,
                content_context=mail_body[:5000]
            )
            
            cot_steps.append({
                "step": "2_classification",
                "output": {
                    "scheme_type": classification.scheme_type,
                    "scheme_sub_type": classification.scheme_sub_type,
                    "confidence": classification.confidence
                },
                "reasoning": classification.reasoning
            })
            
            # Step 3: Extract dates
            logger.debug("CoT Step 3: Extracting dates")
            dates = self.extract_dates(
                text_content=mail_body[:6000],
                context=f"Keywords: {facts.keywords}, Dates mentioned: {facts.key_dates}"
            )
            
            cot_steps.append({
                "step": "3_dates",
                "output": {
                    "duration_start_date": dates.duration_start_date,
                    "duration_end_date": dates.duration_end_date,
                    "starting_at": dates.starting_at,
                    "ending_at": dates.ending_at,
                    "price_drop_date": dates.price_drop_date
                },
                "reasoning": dates.reasoning
            })
            
            # Step 4: Extract financial data
            logger.debug("CoT Step 4: Extracting financials")
            financials = self.extract_financials(
                text_content=mail_body[:6000],
                table_data=mail_body[6000:12000]  # Assume tables are later in body
            )
            
            cot_steps.append({
                "step": "4_financials",
                "output": {
                    "discount_type": financials.discount_type,
                    "discount_value": financials.discount_value,
                    "min_order_value": financials.min_order_value,
                    "max_discount_cap": financials.max_discount_cap,
                    "gst_rate": financials.gst_rate,
                    "brand_support_absolute": financials.brand_support_absolute
                },
                "reasoning": financials.reasoning
            })
            
            # Step 5: Extract vendors
            logger.debug("CoT Step 5: Extracting vendors")
            vendors = self.extract_vendors(
                table_data=mail_body[6000:12000],
                text_content=facts.vendor_names
            )
            
            cot_steps.append({
                "step": "5_vendors",
                "output": {
                    "vendors_json": vendors.vendors_json
                },
                "reasoning": vendors.reasoning
            })
            
            # Step 6: Full extraction with context from previous steps
            logger.debug("CoT Step 6: Assembling final scheme JSON")
            
            # Build context from previous steps
            context_summary = f"""
Subject: {mail_subject}

Classification: {classification.scheme_type} - {classification.scheme_sub_type}
Dates: {dates.duration_start_date} to {dates.duration_end_date}
Discount: {financials.discount_type} - {financials.discount_value}
Vendors: {vendors.vendors_json}
"""
            
            extraction = self.extract_schemes(
                mail_subject=mail_subject,
                mail_body=context_summary + "\\n\\n" + mail_body[:10000]
            )
            
            cot_steps.append({
                "step": "6_full_extraction",
                "output": {
                    "schemes_json": extraction.schemes_json
                },
                "reasoning": extraction.reasoning
            })
            
            # Step 7: Assess confidence
            logger.debug("CoT Step 7: Assessing confidence")
            
            all_reasoning = "\\n\\n".join([
                step["reasoning"] for step in cot_steps
            ])
            
            confidence = self.assess_confidence(
                extracted_data=extraction.schemes_json,
                extraction_reasoning=all_reasoning
            )
            
            cot_steps.append({
                "step": "7_confidence",
                "output": {
                    "confidence_score": confidence.confidence_score,
                    "needs_escalation": confidence.needs_escalation,
                    "missing_fields": confidence.missing_fields,
                    "quality_issues": confidence.quality_issues
                },
                "reasoning": confidence.reasoning
            })
            
            # Return complete result
            return {
                "schemes_json": extraction.schemes_json,
                "reasoning": all_reasoning,
                "cot_steps": cot_steps,
                "confidence_score": confidence.confidence_score,
                "needs_escalation": confidence.needs_escalation
            }
            
        except Exception as e:
            logger.error(f"CoT pipeline failed: {e}", exc_info=True)
            return {
                "schemes_json": '{"schemes": []}',
                "reasoning": f"Extraction failed: {str(e)}",
                "cot_steps": cot_steps,
                "confidence_score": 0.0,
                "needs_escalation": True
            }


class DSPySchemeExtractor:
    """Main interface for DSPy-based scheme extraction.
    
    Provides the same interface as the original SchemeExtractor
    but uses Chain-of-Thought reasoning internally.
    """
    
    def __init__(self, llm: dspy.LM, config: ExtractionConfig):
        """Initialize DSPy scheme extractor.
        
        Args:
            llm: DSPy LM instance (e.g., OpenRouterLLM)
            config: Application configuration
        """
        self.llm = llm
        self.config = config
        
        # Configure DSPy to use this LLM
        dspy.settings.configure(lm=llm)
        
        # Initialize CoT module
        self.cot_module = SchemeExtractionCoT()
        
        logger.info(f"Initialized DSPy extractor with CoT (model: {llm.model_name})")
    
    def extract(
        self,
        email_subject: str,
        email_body: str
    ) -> LLMResponse:
        """Extract scheme headers using Chain-of-Thought reasoning.
        
        Args:
            email_subject: Email subject line
            email_body: Full email body with tables
            
        Returns:
            LLMResponse with extracted schemes and reasoning trace
        """
        logger.info(f"Extracting schemes with CoT: {email_subject[:80]}")
        
        try:
            # Run CoT pipeline
            result = self.cot_module(
                mail_subject=email_subject,
                mail_body=email_body
            )
            
            # Parse schemes JSON
            schemes = self._parse_schemes_json(
                result["schemes_json"],
                result.get("confidence_score", 0.5),
                result.get("needs_escalation", False)
            )
            
            # Get token usage
            usage_stats = self.llm.get_usage_stats()
            
            # Build response
            return LLMResponse(
                schemes=schemes,
                raw_response=result["schemes_json"],
                tokens_used=usage_stats.get("total_tokens"),
                model_used=self.llm.model_name,
                reasoning=result.get("reasoning", ""),
                cot_steps=result.get("cot_steps", [])
            )
            
        except Exception as e:
            logger.error(f"Scheme extraction failed: {e}", exc_info=True)
            return LLMResponse(
                schemes=[],
                raw_response=str(e),
                model_used=self.llm.model_name,
                reasoning=f"Extraction failed: {str(e)}",
                cot_steps=[]
            )
    
    def _parse_schemes_json(
        self,
        schemes_json: str,
        default_confidence: float,
        default_escalation: bool
    ) -> List[SchemeHeader]:
        """Parse schemes JSON into SchemeHeader objects.
        
        Args:
            schemes_json: JSON string with schemes array
            default_confidence: Default confidence if not in JSON
            default_escalation: Default escalation flag if not in JSON
            
        Returns:
            List of SchemeHeader instances
        """
        try:
            # Clean JSON (remove markdown if present)
            cleaned = schemes_json.strip()
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
                    # Apply defaults from CoT confidence assessment
                    if "confidence" not in scheme_data:
                        scheme_data["confidence"] = default_confidence
                    if "needs_escalation" not in scheme_data:
                        scheme_data["needs_escalation"] = default_escalation
                    
                    # Map to SchemeHeader
                    scheme = self._map_to_scheme_header(scheme_data)
                    schemes.append(scheme)
                    
                except ValidationError as e:
                    logger.warning(f"Scheme validation failed: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(schemes)} schemes")
            return schemes
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected parsing error: {e}")
            return []
    
    def _map_to_scheme_header(self, data: Dict[str, Any]) -> SchemeHeader:
        """Map raw scheme data to SchemeHeader model.
        
        Args:
            data: Raw scheme dictionary
            
        Returns:
            SchemeHeader instance
        """
        # Map discount types
        discount_type_map = {
            "Percentage of MRP": "PERCENTAGE",
            "Percentage of NLC": "PERCENTAGE",
            "Absolute": "FLAT",
            "PERCENTAGE": "PERCENTAGE",
            "FLAT": "FLAT",
            "SLAB": "SLAB",
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
            discount_value=data.get("discount_value") or data.get("global_cap_amount"),
            min_order_value=data.get("min_order_value"),
            max_discount_cap=data.get("max_discount_cap") or data.get("global_cap_amount"),
            vendor_name=data.get("vendor_name"),
            category=data.get("category"),
            remarks=data.get("description") or data.get("remarks"),
            confidence=data.get("confidence", 0.5),
            needs_escalation=data.get("needs_escalation", False)
        )
