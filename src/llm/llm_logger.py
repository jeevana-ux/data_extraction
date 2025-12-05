"""Dedicated logging module for LLM interactions with comprehensive tracking.

This module provides structured logging for all LLM API calls, including:
- Input/output token tracking
- Cost calculation based on token usage
- Model parameters (temperature, top_p, etc.)
- Request/response latency
- Detailed JSON logs for analysis
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call."""
    
    # Identification
    call_id: str
    timestamp: str
    model_name: str
    
    # Input parameters
    temperature: float
    max_tokens: int
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost tracking
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    
    # Performance
    latency_seconds: float = 0.0
    
    # Content previews
    input_preview: str = ""
    output_preview: str = ""
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LLMLogger:
    """Comprehensive logger for LLM API interactions."""
    
    def __init__(
        self,
        log_dir: Path = Path("logs/llm_calls"),
        enable_file_logging: bool = True,
        input_cost_per_1m_tokens: float = 0.50,
        output_cost_per_1m_tokens: float = 1.50
    ):
        """
        Initialize LLM logger.
        
        Args:
            log_dir: Directory for detailed JSON logs
            enable_file_logging: Whether to write detailed logs to files
            input_cost_per_1m_tokens: Cost per 1M input tokens in USD
            output_cost_per_1m_tokens: Cost per 1M output tokens in USD
        """
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging
        self.input_cost_per_1m = input_cost_per_1m_tokens
        self.output_cost_per_1m = output_cost_per_1m_tokens
        
        # Create log directory if enabled
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Call counter for unique IDs
        self.call_counter = 0
        
        logger.info(f"LLM Logger initialized (file logging: {enable_file_logging})")
    
    def log_request(
        self,
        model_name: str,
        messages: list,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """
        Log LLM request details.
        
        Args:
            model_name: Name of the model being called
            messages: Input messages
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters (top_p, frequency_penalty, etc.)
            
        Returns:
            Call ID for tracking this request
        """
        self.call_counter += 1
        call_id = f"llm_call_{self.call_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create input preview
        if messages:
            input_text = str(messages)
            input_preview = input_text[:200] + "..." if len(input_text) > 200 else input_text
        else:
            input_preview = ""
        
        # Log to console
        logger.info("="*80)
        logger.info(f"LLM REQUEST [{call_id}]")
        logger.info("="*80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max Tokens: {max_tokens}")
        
        if kwargs.get('top_p') is not None:
            logger.info(f"Top P: {kwargs['top_p']}")
        if kwargs.get('frequency_penalty') is not None:
            logger.info(f"Frequency Penalty: {kwargs['frequency_penalty']}")
        if kwargs.get('presence_penalty') is not None:
            logger.info(f"Presence Penalty: {kwargs['presence_penalty']}")
        
        logger.info(f"Input Preview: {input_preview}")
        logger.info("-"*80)
        
        return call_id
    
    def log_response(
        self,
        call_id: str,
        model_name: str,
        response_text: str,
        usage: Dict[str, int],
        latency_seconds: float,
        temperature: float,
        max_tokens: int,
        input_messages: list,
        error: Optional[str] = None,
        **kwargs
    ):
        """
        Log LLM response details with comprehensive metrics.
        
        Args:
            call_id: Call ID from log_request
            model_name: Model name
            response_text: Response from LLM
            usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens)
            latency_seconds: Time taken for the call
            temperature: Temperature used
            max_tokens: Max tokens used
            input_messages: Input messages for context
            error: Error message if call failed
            **kwargs: Additional parameters
        """
        # Extract token counts
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        total_cost = input_cost + output_cost
        
        # Create output preview
        output_preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        
        # Create input preview
        input_text = str(input_messages)
        input_preview = input_text[:200] + "..." if len(input_text) > 200 else input_text
        
        # Log to console
        logger.info("="*80)
        logger.info(f"LLM RESPONSE [{call_id}]")
        logger.info("="*80)
        
        if error:
            logger.error(f"Status: FAILED - {error}")
        else:
            logger.info("Status: SUCCESS")
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Latency: {latency_seconds:.2f}s")
        logger.info("-"*80)
        logger.info("TOKEN USAGE:")
        logger.info(f"  Input Tokens:  {input_tokens:,}")
        logger.info(f"  Output Tokens: {output_tokens:,}")
        logger.info(f"  Total Tokens:  {total_tokens:,}")
        logger.info("-"*80)
        logger.info("COST BREAKDOWN:")
        logger.info(f"  Input Cost:  ${input_cost:.6f}")
        logger.info(f"  Output Cost: ${output_cost:.6f}")
        logger.info(f"  Total Cost:  ${total_cost:.6f}")
        logger.info("-"*80)
        logger.info(f"Output Preview: {output_preview}")
        logger.info("="*80)
        
        # Create metrics object
        metrics = LLMCallMetrics(
            call_id=call_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get('top_p'),
            frequency_penalty=kwargs.get('frequency_penalty'),
            presence_penalty=kwargs.get('presence_penalty'),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            latency_seconds=latency_seconds,
            input_preview=input_preview,
            output_preview=output_preview,
            success=error is None,
            error_message=error
        )
        
        # Save detailed log to file
        if self.enable_file_logging:
            self._save_detailed_log(metrics, input_messages, response_text)
    
    def _save_detailed_log(
        self,
        metrics: LLMCallMetrics,
        input_messages: list,
        output_text: str
    ):
        """
        Save detailed log entry to JSON file.
        
        Args:
            metrics: Call metrics
            input_messages: Full input messages
            output_text: Full output text
        """
        try:
            log_file = self.log_dir / f"{metrics.call_id}.json"
            
            log_data = {
                "metrics": metrics.to_dict(),
                "full_input": input_messages,
                "full_output": output_text
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Detailed log saved: {log_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save detailed log: {e}")
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with cost breakdown
        """
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def get_summary_stats(self, log_files: Optional[list] = None) -> Dict[str, Any]:
        """
        Get summary statistics from log files.
        
        Args:
            log_files: Optional list of log files to analyze (uses all if None)
            
        Returns:
            Dictionary with summary statistics
        """
        if not self.enable_file_logging:
            return {"error": "File logging not enabled"}
        
        if log_files is None:
            log_files = list(self.log_dir.glob("llm_call_*.json"))
        
        if not log_files:
            return {"total_calls": 0}
        
        total_calls = len(log_files)
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        failed_calls = 0
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    
                    total_input_tokens += metrics.get('input_tokens', 0)
                    total_output_tokens += metrics.get('output_tokens', 0)
                    total_cost += metrics.get('total_cost', 0.0)
                    total_latency += metrics.get('latency_seconds', 0.0)
                    
                    if not metrics.get('success', True):
                        failed_calls += 1
                        
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
                continue
        
        return {
            "total_calls": total_calls,
            "successful_calls": total_calls - failed_calls,
            "failed_calls": failed_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost": total_cost,
            "average_latency": total_latency / total_calls if total_calls > 0 else 0.0,
            "average_cost_per_call": total_cost / total_calls if total_calls > 0 else 0.0
        }
