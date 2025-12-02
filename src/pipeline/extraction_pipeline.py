"""Main extraction pipeline orchestrating PDF processing and scheme extraction."""

import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import pandas as pd

from src.config import ExtractionConfig, get_config
from src.models import ExtractionResult, SchemeHeader, ProcessingMetadata
from src.extractors.pdf_processor import PDFProcessor
from src.llm.llm_client import OpenRouterLLM
from src.llm.dspy_modules import SchemeExtractor
from src.pipeline.output_manager import OutputManager

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Main pipeline orchestrator for end-to-end PDF extraction and scheme generation.
    
    Coordinates:
    1. PDF text/table extraction
    2. Content cleaning
    3. LLM-based scheme extraction
    4. Output management
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize extraction pipeline.
        
        Args:
            config: Application configuration (uses default if None)
        """
        self.config = config or get_config()
        
        # Initialize components
        self.pdf_processor = PDFProcessor(self.config)
        self.output_manager = OutputManager(self.config)
        
        # Initialize LLM client
        self.llm = OpenRouterLLM(
            api_key=self.config.openrouter_api_key,
            model=self.config.openrouter_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            timeout=self.config.llm_timeout
        )
        
        # Initialize scheme extractor
        self.scheme_extractor = SchemeExtractor(self.llm, self.config)
        
        logger.info("Extraction pipeline initialized")
    
    def process_pdf(self, pdf_path: Path, save_output: bool = True) -> ExtractionResult:
        """
        Process a single PDF: extract text and tables.
        
        Args:
            pdf_path: Path to PDF file
            save_output: Whether to save extraction results
            
        Returns:
            ExtractionResult
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Create metadata
        metadata = self.pdf_processor.create_metadata(pdf_path)
        metadata.processing_started = datetime.now()
        
        try:
            # Extract content
            result = self.pdf_processor.process(pdf_path)
            
            # Save if requested
            if save_output:
                self.output_manager.save_extraction_result(result, metadata)
            
            # Update metadata
            metadata.processing_completed = datetime.now()
            metadata.success = True
            
            logger.info(f"PDF processing complete: {pdf_path.name}")
            return result
            
        except Exception as e:
            logger.exception(f"PDF processing failed for {pdf_path.name}: {e}")
            metadata.processing_completed = datetime.now()
            metadata.success = False
            metadata.error_message = str(e)
            raise
    
    def process_multiple_pdfs(
        self,
        pdf_paths: List[Path],
        save_output: bool = True
    ) -> List[ExtractionResult]:
        """
        Process multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            save_output: Whether to save extraction results
            
        Returns:
            List of ExtractionResults
        """
        results = []
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"Processing PDF {i}/{len(pdf_paths)}: {pdf_path.name}")
            
            try:
                result = self.process_pdf(pdf_path, save_output=save_output)
                results.append(result)
            except Exception as e:
                logger.error(f"Skipping {pdf_path.name} due to error: {e}")
                continue
        
        logger.info(f"Processed {len(results)}/{len(pdf_paths)} PDFs successfully")
        return results
    
    def extract_schemes_from_result(
        self,
        result: ExtractionResult
    ) -> List[SchemeHeader]:
        """
        Extract scheme headers from an extraction result.
        
        Args:
            result: ExtractionResult from PDF processing
            
        Returns:
            List of SchemeHeaders
        """
        subject = result.email_subject or "No Subject"
        body = result.combined_body
        
        logger.info(f"Extracting schemes from: {subject[:80]}")
        
        # Call LLM
        llm_response = self.scheme_extractor.extract(subject, body)
        
        # Add source file to schemes
        for scheme in llm_response.schemes:
            scheme.source_file = result.pdf_path.name
        
        logger.info(
            f"Extracted {len(llm_response.schemes)} schemes "
            f"(avg confidence: {llm_response.average_confidence:.2f})"
        )
        
        return llm_response.schemes
    
    def build_scheme_headers_from_output(self) -> pd.DataFrame:
        """
        Build scheme headers from previously extracted PDFs in output directory.
        
        This is the equivalent of the original build_scheme_header.py script.
        
        Returns:
            DataFrame of scheme headers
        """
        logger.info("Building scheme headers from extracted output")
        
        # Load extracted emails
        emails_df = self.output_manager.load_extracted_emails()
        
        if emails_df.empty:
            logger.warning("No extracted emails found in output directory")
            return pd.DataFrame()
        
        logger.info(f"Found {len(emails_df)} extracted emails")
        
        # Extract schemes from each email
        all_schemes = []
        
        for idx, row in emails_df.iterrows():
            subject = row['mail_subject']
            body = row['mail_body']
            source_file = row['sourceFile']
            
            logger.info(f"Processing email {idx+1}/{len(emails_df)}: {subject[:80]}")
            
            try:
                llm_response = self.scheme_extractor.extract(subject, body)
                
                # Add source file
                for scheme in llm_response.schemes:
                    scheme.source_file = source_file
                
                all_schemes.extend(llm_response.schemes)
                
                logger.info(f"Extracted {len(llm_response.schemes)} schemes")
                
            except Exception as e:
                logger.error(f"Scheme extraction failed for {subject[:80]}: {e}")
                continue
        
        # Save schemes
        if all_schemes:
            df = self.output_manager.save_schemes(all_schemes)
            logger.info(f"Saved {len(all_schemes)} schemes to {self.config.scheme_header_path}")
            return df
        else:
            logger.warning("No schemes extracted")
            return pd.DataFrame()
    
    def run_full_pipeline(self, pdf_paths: List[Path]) -> pd.DataFrame:
        """
        Run the complete pipeline: extract PDFs and build scheme headers.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            DataFrame of scheme headers
        """
        logger.info(f"Starting full pipeline for {len(pdf_paths)} PDFs")
        
        # Step 1: Process PDFs
        logger.info("Step 1: Processing PDFs...")
        extraction_results = self.process_multiple_pdfs(pdf_paths, save_output=True)
        
        # Step 2: Extract schemes
        logger.info("Step 2: Extracting schemes...")
        all_schemes = []
        
        for result in extraction_results:
            try:
                schemes = self.extract_schemes_from_result(result)
                all_schemes.extend(schemes)
            except Exception as e:
                logger.error(f"Scheme extraction failed for {result.pdf_path.name}: {e}")
                continue
        
        # Step 3: Save schemes
        logger.info("Step 3: Saving scheme headers...")
        if all_schemes:
            df = self.output_manager.save_schemes(all_schemes)
            logger.info(f"Pipeline complete: {len(all_schemes)} schemes extracted")
            return df
        else:
            logger.warning("Pipeline complete: No schemes extracted")
            return pd.DataFrame()
    
    def get_usage_stats(self) -> dict:
        """
        Get LLM usage statistics.
        
        Returns:
            Dictionary with token usage stats
        """
        return self.llm.get_usage_stats()
