"""Custom LLM client for OpenRouter API compatible with DSPy."""

import json
import logging
import time
from typing import Optional, List, Dict, Any
import requests

import dspy

from src.llm.llm_logger import LLMLogger

logger = logging.getLogger(__name__)


class OpenRouterLLM(dspy.LM):
    """
    Custom DSPy LM adapter for OpenRouter API.
    
    This allows DSPy to work with OpenRouter's model marketplace.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-next-80b-a3b-instruct",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        timeout: int = 120,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        enable_logging: bool = True,
        llm_logger: Optional[LLMLogger] = None,
        input_cost_per_1m: float = 0.50,
        output_cost_per_1m: float = 1.50
    ):
        """
        Initialize OpenRouter LLM client.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier
            base_url: API base URL
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty for repetition
            presence_penalty: Presence penalty for topic repetition
            enable_logging: Enable detailed LLM logging
            llm_logger: Optional LLMLogger instance (creates new if None)
            input_cost_per_1m: Cost per 1M input tokens
            output_cost_per_1m: Cost per 1M output tokens
        """
        super().__init__(model=model)
        
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
        # Initialize LLM logger
        self.enable_logging = enable_logging
        if enable_logging:
            self.llm_logger = llm_logger or LLMLogger(
                input_cost_per_1m_tokens=input_cost_per_1m,
                output_cost_per_1m_tokens=output_cost_per_1m
            )
        else:
            self.llm_logger = None
        
        self.history: List[Dict[str, Any]] = []
    
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Call the LLM with a prompt or messages.
        
        Args:
            prompt: Single prompt string (converted to messages)
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            List of response strings (typically single item)
        """
        # Convert prompt to messages if needed
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        
        # Get parameters (allow kwargs to override)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        frequency_penalty = kwargs.get("frequency_penalty", self.frequency_penalty)
        presence_penalty = kwargs.get("presence_penalty", self.presence_penalty)
        
        # Log request if enabled
        call_id = None
        if self.enable_logging and self.llm_logger:
            call_id = self.llm_logger.log_request(
                model_name=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add optional parameters if provided
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        
        # Start timing
        start_time = time.time()
        
        try:
            logger.debug(f"Calling OpenRouter API with model: {self.model_name}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                
                # Store in history with complete metadata
                history_entry = {
                    "prompt": messages,
                    "response": response_text,
                    "model": self.model_name,
                    "usage": usage,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "latency_seconds": latency,
                    "call_id": call_id
                }
                self.history.append(history_entry)
                
                # Log response if enabled
                if self.enable_logging and self.llm_logger and call_id:
                    self.llm_logger.log_response(
                        call_id=call_id,
                        model_name=self.model_name,
                        response_text=response_text,
                        usage=usage,
                        latency_seconds=latency,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        input_messages=messages,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty
                    )
                
                logger.info(
                    f"LLM response received: {len(response_text)} chars, "
                    f"tokens: {usage.get('total_tokens', 'unknown')}"
                )
                
                return [response_text]
            else:
                logger.error(f"Unexpected API response format: {result}")
                return [""]
        
        except requests.exceptions.Timeout:
            latency = time.time() - start_time
            error_msg = f"OpenRouter API timeout after {self.timeout}s"
            logger.error(error_msg)
            
            # Log error if enabled
            if self.enable_logging and self.llm_logger and call_id:
                self.llm_logger.log_response(
                    call_id=call_id,
                    model_name=self.model_name,
                    response_text="",
                    usage={},
                    latency_seconds=latency,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    input_messages=messages,
                    error=error_msg,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
            raise
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            error_msg = f"OpenRouter API request failed: {e}"
            logger.error(error_msg)
            
            # Log error if enabled
            if self.enable_logging and self.llm_logger and call_id:
                self.llm_logger.log_response(
                    call_id=call_id,
                    model_name=self.model_name,
                    response_text="",
                    usage={},
                    latency_seconds=latency,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    input_messages=messages,
                    error=error_msg,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
            raise
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Unexpected error calling OpenRouter: {e}"
            logger.error(error_msg)
            
            # Log error if enabled
            if self.enable_logging and self.llm_logger and call_id:
                self.llm_logger.log_response(
                    call_id=call_id,
                    model_name=self.model_name,
                    response_text="",
                    usage={},
                    latency_seconds=latency,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    input_messages=messages,
                    error=error_msg,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
            raise
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get cumulative token usage statistics.
        
        Returns:
            Dictionary with token usage stats
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        for entry in self.history:
            usage = entry.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
        
        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "num_calls": len(self.history)
        }
