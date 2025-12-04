"""
Modern CLI interface for PDF extraction and scheme header generation.

Provides commands for:
- Extracting PDFs
- Building scheme headers
- Running full pipeline
"""

import logging
from pathlib import Path
from typing import List, Optional

import click

from src.config import get_config
from src.pipeline import ExtractionPipeline

# Configure logging to both console and file
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create log filename with timestamp
log_filename = log_dir / f"extraction_{datetime.now().strftime('%Y%m%d')}.log"

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Console handler (for terminal output)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler (for log file output with rotation)
file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)  # Capture everything in file
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers to root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filename}")


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """PDF Extraction and Scheme Header Generation Pipeline"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)


@cli.command()
@click.argument('pdf_files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--no-save', is_flag=True, help='Do not save extraction results')
def extract(pdf_files: tuple, no_save: bool):
    """
    Extract text and tables from PDF files.
    
    Example:
        python -m src.main extract input/*.pdf
    """
    click.echo(f"\n📄 Extracting {len(pdf_files)} PDF(s)...\n")
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Convert to Path objects
    pdf_paths = [Path(p) for p in pdf_files]
    
    # Process PDFs
    results = pipeline.process_multiple_pdfs(
        pdf_paths,
        save_output=not no_save
    )
    
    # Summary
    click.echo(f"\n✅ Extraction complete!")
    click.echo(f"   Processed: {len(results)}/{len(pdf_paths)} PDFs")
    
    for result in results:
        click.echo(f"   - {result.pdf_path.name}: {result.table_count} tables, {len(result.full_text)} chars")


@cli.command()
def build_headers():
    """
    Build scheme headers from previously extracted PDFs.
    
    Reads from the output directory and generates scheme_header.csv.
    
    Example:
        python -m src.main build-headers
    """
    click.echo("\n🔍 Building scheme headers from extracted output...\n")
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Build headers
    df = pipeline.build_scheme_headers_from_output()
    
    if not df.empty:
        config = get_config()
        click.echo(f"\n✅ Done! scheme_header.csv created at: {config.scheme_header_path}")
        click.echo(f"   Total schemes: {len(df)}")
        
        # Show usage stats
        stats = pipeline.get_usage_stats()
        click.echo(f"\n📊 LLM Usage:")
        click.echo(f"   API calls: {stats.get('num_calls', 0)}")
        click.echo(f"   Total tokens: {stats.get('total_tokens', 0)}")
    else:
        click.echo("\n⚠️  No schemes extracted")


@cli.command()
@click.argument('pdf_files', nargs=-1, type=click.Path(exists=True), required=True)
def run_full(pdf_files: tuple):
    """
    Run full pipeline: extract PDFs and build scheme headers.
    
    Example:
        python -m src.main run-full input/*.pdf
    """
    click.echo(f"\n🚀 Running full pipeline for {len(pdf_files)} PDF(s)...\n")
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Convert to Path objects
    pdf_paths = [Path(p) for p in pdf_files]
    
    # Run full pipeline
    df = pipeline.run_full_pipeline(pdf_paths)
    
    if not df.empty:
        config = get_config()
        click.echo(f"\n✅ Pipeline complete!")
        click.echo(f"   Schemes extracted: {len(df)}")
        click.echo(f"   Output: {config.scheme_header_path}")
        
        # Show usage stats
        stats = pipeline.get_usage_stats()
        click.echo(f"\n📊 LLM Usage:")
        click.echo(f"   API calls: {stats.get('num_calls', 0)}")
        click.echo(f"   Total tokens: {stats.get('total_tokens', 0)}")
    else:
        click.echo("\n⚠️  No schemes extracted")


@cli.command()
def info():
    """Show configuration and system information."""
    config = get_config()
    
    click.echo("\n📋 Configuration:")
    click.echo(f"   Model: {config.openrouter_model}")
    click.echo(f"   Input dir: {config.input_dir}")
    click.echo(f"   Output dir: {config.output_dir}")
    click.echo(f"   Final output: {config.final_output_dir}")
    click.echo(f"   OCR enabled: {config.ocr_enabled}")
    click.echo(f"   Camelot enabled: {config.camelot_enabled}")


if __name__ == '__main__':
    cli()
