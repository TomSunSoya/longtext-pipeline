"""LLM base abstraction layer.

This module defines the abstract interface for LLM provider integration.
All concrete LLM implementations must inherit from LLMClient and implement
the required methods following the documented contract.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMClient(ABC):
    """Abstract base class for LLM provider integrations.
    
    This class defines the minimal interface that all LLM providers must implement.
    Concrete implementations handle provider-specific details while maintaining
    a consistent interface for the pipeline.
    
    Async Support:
        All sync methods have corresponding async versions that must be implemented
        for async/await patterns. The sync methods are preserved for backward
        compatibility with existing synchronous code.
    
    Example:
        >>> class OpenAICompatibleClient(LLMClient):
        ...     def __init__(self, api_key: str, base_url: str, model: str):
        ...         self.api_key = api_key
        ...         self.base_url = base_url
        ...         self.model = model
        ...
        ...     def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        ...         # Implementation specific to OpenAI-compatible APIs
        ...         pass
        ...
        ...     def complete_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        ...         # Implementation that parses response as JSON
        ...         pass
        ...
        ...     async def acomplete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        ...         # Async version of complete
        ...         pass
        ...
        ...     async def acomplete_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        ...         # Async version of complete_json
        ...         pass
    """
    
    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a completion response from the LLM.
        
        This is the core method for text generation. Implementations must handle
        all provider-specific communication, including authentication, request
        formatting, response parsing, and error handling.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt to set context/behavior
            
        Returns:
            The generated text response as a string
            
        Raises:
            LLMTimeoutError: If the request exceeds timeout threshold
            LLMRateLimitError: If rate limiting is encountered
            LLMAuthenticationError: If authentication fails
            LLMContentFilterError: If content violates provider policies
            LLMCommunicationError: For network/communication failures
            
        Contract Requirements:
            - Must implement timeout handling (configurable via config)
            - Must implement retry logic for transient failures
            - Must raise appropriate exceptions from errors.py
            - Must not return None or empty string on success
        """
        pass
    
    @abstractmethod
    def complete_json(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        """Generate a structured JSON response from the LLM.
        
        This method is optimized for structured output. Implementations should
        request JSON format from the LLM when supported, and parse/validate
        the response before returning.
        
        Args:
            prompt: The user prompt requesting structured output
            system_prompt: Optional system prompt (can include JSON format instructions)
            
        Returns:
            Parsed JSON response as a Python dictionary
            
        Raises:
            LLMTimeoutError: If the request exceeds timeout threshold
            LLMRateLimitError: If rate limiting is encountered
            LLMAuthenticationError: If authentication fails
            LLMContentFilterError: If content violates provider policies
            LLMCommunicationError: For network/communication failures
            LLMResponseError: If response cannot be parsed as valid JSON
            
        Contract Requirements:
            - Must validate that response is valid JSON
            - Must raise LLMResponseError on JSON parsing failures
            - Should request JSON mode/format from provider when available
            # Must not return None; return empty dict {} if no content
        """
        pass
    
    @abstractmethod
    async def acomplete(
        self, prompt: str, system_prompt: str | None = None
    ) -> str:
        """Generate an async completion response from the LLM.
        
        This is the async version of `complete()`. Implementations must handle
        all provider-specific communication, including authentication, request
        formatting, response parsing, and error handling in an async context.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt to set context/behavior
            
        Returns:
            The generated text response as a string
            
        Raises:
            LLMTimeoutError: If the request exceeds timeout threshold
            LLMRateLimitError: If rate limiting is encountered
            LLMAuthenticationError: If authentication fails
            LLMContentFilterError: If content violates provider policies
            LLMCommunicationError: For network/communication failures
            
        Contract Requirements:
            - Must implement timeout handling (configurable via config)
            - Must implement retry logic for transient failures
            - Must raise appropriate exceptions from errors.py
            - Must not return None or empty string on success
        """
        pass
    
    @abstractmethod
    async def acomplete_json(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        """Generate a structured async JSON response from the LLM.
        
        This is the async version of `complete_json()`. Implementations should
        request JSON format from the LLM when supported, and parse/validate
        the response before returning in an async context.
        
        Args:
            prompt: The user prompt requesting structured output
            system_prompt: Optional system prompt (can include JSON format instructions)
            
        Returns:
            Parsed JSON response as a Python dictionary
            
        Raises:
            LLMTimeoutError: If the request exceeds timeout threshold
            LLMRateLimitError: If rate limiting is encountered
            LLMAuthenticationError: If authentication fails
            LLMContentFilterError: If content violates provider policies
            LLMCommunicationError: For network/communication failures
            LLMResponseError: If response cannot be parsed as valid JSON
            
        Contract Requirements:
            - Must validate that response is valid JSON
            - Must raise LLMResponseError on JSON parsing failures
            - Should request JSON mode/format from provider when available
            - Must not return None; return empty dict {} if no content
        """
        pass
