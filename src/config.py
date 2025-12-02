"""
Configuration management for PDF extraction and scheme header generation.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class ExtractionConfig(BaseSettings):
    """Main configuration for the PDF extraction pipeline."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # ===== API Configuration =====
    openrouter_api_key: str = Field(
        ...,
        description="OpenRouter API key for LLM access"
    )
    
    openrouter_model: str = Field(
        default="qwen/qwen3-next-80b-a3b-instruct",
        description="OpenRouter model identifier"
    )
    
    openrouter_url: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter API endpoint"
    )
    
    # ===== Directory Configuration =====
    input_dir: Path = Field(
        default=Path("input"),
        description="Directory containing input PDF files"
    )
    
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for extraction outputs (text, tables, metadata)"
    )
    
    final_output_dir: Path = Field(
        default=Path("out"),
        description="Directory for final scheme header CSV"
    )
    
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )
    
    # ===== Processing Configuration =====
    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR fallback for image-based PDFs"
    )
    
    camelot_enabled: bool = Field(
        default=False,
        description="Enable Camelot for table extraction"
    )
    
    ocr_dpi: int = Field(
        default=200,
        description="DPI for OCR rendering"
    )
    
    ocr_language: str = Field(
        default="eng",
        description="Tesseract OCR language"
    )
    
    # =====  LLM Configuration (OpenRouter) =====
    llm_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="OpenRouter model to use for scheme extraction"
    )
    
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM sampling"
    )
    
    llm_max_tokens: int = Field(
        default=4000,
        ge=100,
        description="Maximum tokens for LLM response"
    )
    
    llm_timeout: int = Field(
        default=120,
        gt=0,
        description="LLM API call timeout in seconds"
    )
    
    # ===== DSPy Configuration =====
    enable_chain_of_thought: bool = Field(
        default=True,
        description="Enable Chain-of-Thought reasoning with DSPy"
    )
    
    save_cot_reasoning: bool = Field(
        default=True,
        description="Save CoT reasoning traces to log files"
    )
    
    optimizer_type: str = Field(
        default="BootstrapFewShot",
        description="DSPy optimizer for prompt optimization"
    )
    
    max_bootstrapped_demos: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Maximum number of few-shot examples"
    )
    
    cot_log_dir: Path = Field(
        default=Path("logs/cot_reasoning"),
        description="Directory for CoT reasoning logs"
    )
    
    # ===== Retry Configuration =====
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for failed operations"
    )
    
    retry_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Initial delay between retries in seconds"
    )
    
    # ===== Output Configuration =====
    scheme_header_filename: str = Field(
        default="scheme_header.json",
        description="Output filename for scheme headers"
    )
    
    @field_validator('input_dir', 'output_dir', 'final_output_dir', 'logs_dir')
    @classmethod
    def ensure_directory_exists(cls, v: Path) -> Path:
        """Ensure directory exists, create if not."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('openrouter_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("OpenRouter API key cannot be empty")
        return v.strip()
    
    @property
    def scheme_header_path(self) -> Path:
        """Full path to the scheme header CSV file."""
        return self.final_output_dir / self.scheme_header_filename


# Global config instance
_config: Optional[ExtractionConfig] = None


def get_config() -> ExtractionConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = ExtractionConfig()
    return _config


def reload_config() -> ExtractionConfig:
    """Reload configuration from environment."""
    global _config
    _config = ExtractionConfig()
    return _config
