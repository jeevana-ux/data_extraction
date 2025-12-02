"""Custom LLM client for OpenRouter API compatible with DSPy."""

import json
import logging
from typing import Optional, List, Dict, Any
import requests

import dspy

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
        timeout: int = 120
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
        """
        super().__init__(model=model)
        
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
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
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        try:
            logger.debug(f"Calling OpenRouter API with model: {self.model_name}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response text
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                
                # Store in history
                self.history.append({
                    "prompt": messages,
                    "response": response_text,
                    "model": self.model_name,
                    "usage": result.get("usage", {})
                })
                
                logger.info(
                    f"LLM response received: {len(response_text)} chars, "
                    f"tokens: {result.get('usage', {}).get('total_tokens', 'unknown')}"
                )
                
                return [response_text]
            else:
                logger.error(f"Unexpected API response format: {result}")
                return [""]
        
        except requests.exceptions.Timeout:
            logger.error(f"OpenRouter API timeout after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenRouter: {e}")
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
