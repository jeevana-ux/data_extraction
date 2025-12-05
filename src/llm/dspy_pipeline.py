"""Fixed DSPy Chain-of-Thought scheme extraction pipeline with expert-engineered prompt.

This version uses direct LLM calls with comprehensive 21-field extraction prompt
while maintaining DSPy structure and logging.
"""

import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

import dspy

from src.config import ExtractionConfig
from src.models import LLMResponse, SchemeHeader
from pydantic import ValidationError

logger = logging.getLogger(__name__)

from src.llm.signatures import ExpertSchemeExtractionSignature

class DSPySchemeExtractor:
    """DSPy-based scheme extractor with expert-engineered 21-field extraction.
    
    Uses DSPy infrastructure with comprehensive prompt for accurate extraction.
    """
    
    def __init__(self, llm: dspy.LM, config: ExtractionConfig):
        """Initialize DSPy scheme extractor.
    
        Args:
            llm: DSPy LM instance (e.g., OpenRouterLLM)
            config: Application configuration
        """
        self.llm = llm
        self.config = config
        
        # Configure DSPy
        dspy.settings.configure(lm=llm)
        
        # Initialize DSPy ChainOfThought module with Expert Signature
        self.extract_module = dspy.ChainOfThought(ExpertSchemeExtractionSignature)
        
        logger.info(f"✓ Initialized DSPy CoT extractor (model: {llm.model_name})")
    
    def extract(
        self,
        email_subject: str,
        email_body: str
    ) -> LLMResponse:
        """Extract scheme headers using expert-engineered prompt via DSPy with CoT.
        
        Args:
            email_subject: Email subject line
            email_body: Full email body with tables
            
        Returns:
            LLMResponse with extracted schemes and CoT reasoning
        """
        logger.info("="*80)
        logger.info(f"DSPy CoT Extraction Started: {email_subject[:60]}...")
        logger.info("="*80)
        
        try:
            # Execute DSPy ChainOfThought module
            logger.info("Calling DSPy ChainOfThought module...")
            logger.debug(f"Input Subject: {email_subject}")
            logger.debug(f"Input Body Length: {len(email_body)}")
            
            prediction = self.extract_module(
                mail_subject=email_subject,
                mail_body=email_body[:15000]  # Truncate to avoid context limits
            )
            
            # Extract reasoning and JSON response
            reasoning_text = prediction.reasoning
            response_text = prediction.schemes_json
            
            logger.info(f"✓ LLM response received: {len(response_text)} chars")
            logger.info("="*80)
            logger.info("CHAIN OF THOUGHT REASONING:")
            logger.info("="*80)
            logger.info(reasoning_text)
            logger.info("="*80)
            
            # Log field-level reasoning
            self._log_field_reasoning(reasoning_text)
            
            logger.info(f"Response JSON preview: {response_text[:500]}...")
            
            # Parse and validate JSON
            schemes = self._parse_schemes_json(response_text)
            
            # Get usage stats
            usage_stats = self.llm.get_usage_stats()
            
            # Save CoT reasoning if enabled
            if self.config.save_cot_reasoning and schemes:
                self._save_cot_reasoning_log(email_subject, reasoning_text, response_text, schemes)
            
            logger.info("="*80)
            logger.info(f"DSPy CoT Extraction Complete: {len(schemes)} scheme(s) extracted")
            logger.info("="*80)
            
            return LLMResponse(
                schemes=schemes,
                raw_response=response_text,
                tokens_used=usage_stats.get("total_tokens"),
                model_used=self.llm.model_name,
                reasoning=reasoning_text,
                cot_steps=[]
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return LLMResponse(
                schemes=[],
                raw_response=str(e),
                model_used=self.llm.model_name,
                reasoning=f"Extraction failed: {str(e)}",
                cot_steps=[]
            )
    
    def _log_field_reasoning(self, reasoning_text: str):
        """Log field-by-field reasoning in structured format.
        
        Args:
            reasoning_text: Full CoT reasoning from LLM
        """
        logger.info("="*80)
        logger.info("FIELD-LEVEL EXTRACTION REASONING:")
        logger.info("="*80)
        
        # Parse reasoning by field
        field_sections = reasoning_text.split("Field:")
        
        for section in field_sections[1:]:  # Skip first empty split
            lines = section.strip().split("\n")
            if lines:
                field_name = lines[0].strip()
                logger.info(f"\n📋 Field: {field_name}")
                
                for line in lines[1:]:
                    if line.strip():
                        logger.info(f"   {line.strip()}")
        
        logger.info("="*80)
    
    # _build_extraction_prompt is no longer needed as it's in the Signature docstring
    
    def _parse_schemes_json(self, json_str: str) -> List[SchemeHeader]:
        """Parse and validate schemes JSON with 21 fields.
        
        Args:
            json_str: Raw JSON string from LLM
            
        Returns:
            List of SchemeHeader objects
        """
        try:
            # Clean markdown if present
            cleaned = json_str.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            logger.debug(f"Cleaned JSON length: {len(cleaned)} chars")
            
            # Parse JSON
            data = json.loads(cleaned)
            
            # Handle list output (e.g. [{"scheme_name": ...}])
            if isinstance(data, list):
                logger.info("Received list of schemes directly")
                data = {"schemes": data}
            
            # Handle single object without "schemes" key
            elif isinstance(data, dict) and "schemes" not in data:
                # Check if it looks like a scheme object
                if "scheme_name" in data or "scheme_type" in data:
                    logger.info("Received single scheme object, wrapping in array")
                    data = {"schemes": [data]}
                else:
                    logger.warning(f"JSON missing 'schemes' key. Keys found: {list(data.keys())}")
                    return []
            
            if "schemes" not in data or not isinstance(data["schemes"], list):
                logger.error(f"Invalid JSON structure: {type(data)}")
                return []
            
            # Parse each scheme
            schemes = []
            for i, scheme_data in enumerate(data["schemes"], 1):
                try:
                    logger.debug(f"Parsing scheme {i}/{len(data['schemes'])}...")
                    scheme = self._map_to_scheme_header(scheme_data)
                    schemes.append(scheme)
                    logger.info(f"✓ Scheme {i}: {scheme.scheme_name} ({scheme.scheme_type}/{scheme.scheme_subtype})")
                    
                except ValidationError as e:
                    logger.warning(f"Scheme {i} validation failed: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Scheme {i} parsing error: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(schemes)}/{len(data['schemes'])} schemes")
            return schemes
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            logger.debug(f"Raw response: {json_str[:1000]}...")
            return []
        except Exception as e:
            logger.error(f"Parsing error: {e}", exc_info=True)
            return []
    
    def _map_to_scheme_header(self, data: Dict[str, Any]) -> SchemeHeader:
        """Map JSON data to SchemeHeader model with 21 fields.
        
        Args:
            data: Scheme dictionary from LLM JSON response
            
        Returns:
            SchemeHeader instance with all 21 fields mapped
        """
        # Normalize discount type
        discount_type = data.get("discount_type")
        if discount_type and discount_type not in ["Percentage of NLC", "Percentage of MRP", "Absolute"]:
            # Try to normalize
            if "nlc" in str(discount_type).lower():
                discount_type = "Percentage of NLC"
            elif "mrp" in str(discount_type).lower():
                discount_type = "Percentage of MRP"
            elif "absolute" in str(discount_type).lower() or "flat" in str(discount_type).lower():
                discount_type = "Absolute"
        
        # Normalize scheme_subtype
        scheme_subtype = data.get("scheme_subtype", "")
        # Map variations to standard names
        subtype_map = {
            "puc": "PUC/FDC",
            "fdc": "PUC/FDC",
            "puc/fdc": "PUC/FDC",
            "periodic claim": "PERIODIC_CLAIM",
            "periodic_claim": "PERIODIC_CLAIM",
            "super coin": "SUPER COIN",
            "supercoin": "SUPER COIN",
            "bank offer": "BANK OFFER",
            "one off": "ONE_OFF",
            "one-off": "ONE_OFF"
        }
        scheme_subtype = subtype_map.get(str(scheme_subtype).lower(), scheme_subtype)
        
        return SchemeHeader(
            # Core Identification
            scheme_name=data.get("scheme_name"),
            scheme_description=data.get("scheme_description"),
            vendor_name=data.get("vendor_name"),
            
            # Scheme Classification
            scheme_type=data.get("scheme_type", "OTHER"),
            scheme_subtype=scheme_subtype or "OTHER",
            
            # Temporal Information (DD/MM/YYYY format - keep as-is from LLM)
            scheme_period=data.get("scheme_period", "Duration"),
            duration=data.get("duration"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            price_drop_date=data.get("price_drop_date"),
            
            # Financial Terms
            discount_type=discount_type,
            max_cap=str(data.get("max_cap")) if data.get("max_cap") is not None else "Not Specified",
            discount_slab_type=data.get("discount_slab_type"),
            brand_support_absolute=str(data.get("brand_support_absolute")) if data.get("brand_support_absolute") is not None else "Not Applicable",
            gst_rate=str(data.get("gst_rate")) if data.get("gst_rate") is not None else "Not Applicable",
            
            # Conditions and Metadata
            additional_conditions=data.get("additional_conditions"),
            fsn_file_config_file=data.get("fsn_file_config_file", "No"),
            minimum_of_actual_discount_or_agreed_claim=data.get("minimum_of_actual_discount_or_agreed_claim", "No"),
            remove_gst_from_final_claim=data.get("remove_gst_from_final_claim"),
            over_and_above=data.get("over_and_above", "No"),
            scheme_document=data.get("scheme_document", "No"),
            best_bet=data.get("best_bet", "No"),
            
            # Legacy fields (optional)
            confidence=data.get("confidence", 0.7),
            needs_escalation=data.get("needs_escalation", False)
        )
    
    def _save_cot_reasoning_log(
        self, 
        subject: str, 
        reasoning: str,
        json_response: str,
        schemes: List[SchemeHeader]
    ):
        """Save CoT reasoning trace to log file.
        
        Args:
            subject: Email subject
            reasoning: CoT reasoning text
            json_response: JSON response
            schemes: Extracted schemes
        """
        if not self.config.cot_log_dir.exists():
            self.config.cot_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from subject
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_subject = "".join(c for c in subject[:50] if c.isalnum() or c in " _-").replace(" ", "_")
        filename = f"{safe_subject}_{timestamp}_cot.json"
        filepath = self.config.cot_log_dir / filename
        
        # Get last LLM call metadata from history
        llm_metadata = {}
        if hasattr(self.llm, 'history') and len(self.llm.history) > 0:
            last_call = self.llm.history[-1]
            llm_metadata = {
                "model": last_call.get("model"),
                "temperature": last_call.get("temperature"),
                "max_tokens": last_call.get("max_tokens"),
                "top_p": last_call.get("top_p"),
                "frequency_penalty": last_call.get("frequency_penalty"),
                "presence_penalty": last_call.get("presence_penalty"),
                "usage": last_call.get("usage", {}),
                "latency_seconds": last_call.get("latency_seconds"),
                "call_id": last_call.get("call_id")
            }
            
            # Calculate cost if we have usage stats
            usage = last_call.get("usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                
                if hasattr(self.llm, 'llm_logger') and self.llm.llm_logger:
                    cost_info = self.llm.llm_logger.calculate_cost(input_tokens, output_tokens)
                    llm_metadata["cost"] = cost_info
        
        # Create structured log
        log_data = {
            "timestamp": timestamp,
            "email_subject": subject,
            "llm_metadata": llm_metadata,
            "cot_reasoning": reasoning,
            "json_response": json_response,
            "extracted_schemes": [
                {
                    "scheme_name": s.scheme_name,
                    "scheme_type": s.scheme_type,
                    "scheme_subtype": s.scheme_subtype,
                    "start_date": s.start_date,
                    "end_date": s.end_date,
                    "vendor_name": s.vendor_name
                }
                for s in schemes
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ CoT reasoning saved to: {filepath}")


