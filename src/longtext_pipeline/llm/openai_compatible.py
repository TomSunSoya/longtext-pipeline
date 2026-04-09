"""OpenAI-compatible LLM client implementation.

This module provides an httpx-based client for OpenAI-compatible APIs.
Supports custom endpoints via OPENAI_BASE_URL for local models, proxies, etc.
"""

import json
import os
from typing import Optional

import httpx

from ..errors import (
    LLMError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMCommunicationError,
    LLMResponseError,
)
from ..utils.retry import retry_llm_call, retry_llm_call_async
from .base import LLMClient


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible LLM client using httpx.
    
    This client communicates with OpenAI-compatible APIs using raw HTTP requests.
    It supports custom endpoints via OPENAI_BASE_URL environment variable,
    enabling use with local models (Ollama, vLLM), proxies, or alternative providers.
    
    Environment Variables:
        OPENAI_API_KEY: API key for authentication (required)
        OPENAI_BASE_URL: Custom base URL (default: https://api.openai.com/v1)
    
    Example:
        >>> client = OpenAICompatibleClient(
        ...     model="gpt-4o-mini",
        ...     timeout=30.0
        ... )
        >>> response = client.complete("Hello, world!")
        >>> print(response)
        "Hello! How can I help you today?"
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the OpenAI-compatible client.
        
        Args:
            model: Model name to use (default: gpt-4o-mini)
            base_url: Base URL for the API (default: from OPENAI_BASE_URL or OpenAI default)
            api_key: API key for authentication (default: from OPENAI_API_KEY)
            timeout: Request timeout in seconds (default: 30.0)
            
        Raises:
            LLMAuthenticationError: If no API key is provided
        """
        self.model = model or os.getenv("LONGTEXT_MODEL_NAME") or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL
        self.timeout = timeout
        
        if not self.api_key:
            raise LLMAuthenticationError(
                "API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Ensure base_url ends with proper path
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        
        self._endpoint = f"{self.base_url}/chat/completions"
        
        # Initialize async client params (same config as sync)
        # Store timeout separately for httpx.AsyncClient
        self._async_timeout = httpx.Timeout(timeout)
    
    def _build_headers(self) -> dict:
        """Build HTTP headers for requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> dict:
        """Build the request payload for the API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: Optional response format hint (e.g., "json")
            
        Returns:
            Dictionary with request payload
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        
        # Request JSON format if needed
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        
        return payload
    
    def _handle_error(self, status_code: int, response_text: str) -> None:
        """Handle HTTP error responses.
        
        Args:
            status_code: HTTP status code
            response_text: Response body text
            
        Raises:
            LLMAuthenticationError: On 401 Unauthorized
            LLMRateLimitError: On 429 Too Many Requests
            LLMContentFilterError: On 400 with content filter error
            LLMCommunicationError: On 5xx server errors
            LLMError: On other errors
        """
        if status_code == 401:
            raise LLMAuthenticationError(
                f"Authentication failed (401). Check your API key. Response: {response_text}"
            )
        elif status_code == 429:
            raise LLMRateLimitError(
                f"Rate limit exceeded (429). Try again later. Response: {response_text}"
            )
        elif status_code == 400:
            # Check if it's a content filter error
            if "content_filter" in response_text.lower():
                raise LLMContentFilterError(
                    f"Content filtered by provider (400). Response: {response_text}"
                )
            raise LLMError(f"Invalid request (400). Response: {response_text}")
        elif status_code >= 500:
            raise LLMCommunicationError(
                f"Server error ({status_code}). Try again later. Response: {response_text}"
            )
        else:
            raise LLMError(
                f"Unexpected error ({status_code}). Response: {response_text}"
            )
    
    def _make_request(
        self,
        payload: dict,
    ) -> dict:
        """Make HTTP request to the API.
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            Parsed JSON response
            
        Raises:
            LLMTimeoutError: On request timeout
            LLMCommunicationError: On network errors
            LLMError: On API errors
        """
        try:
            # Ignore ambient proxy environment variables unless the caller
            # explicitly bakes proxying into the endpoint they provide.
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.post(
                    self._endpoint,
                    headers=self._build_headers(),
                    json=payload,
                )
                
                if response.status_code != 200:
                    self._handle_error(response.status_code, response.text)
                
                return response.json()
                
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Request timed out after {self.timeout} seconds"
            ) from e
        except httpx.NetworkError as e:
            raise LLMCommunicationError(
                f"Network error: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise LLMCommunicationError(
                f"HTTP error: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Invalid JSON response from API: {e}"
            ) from e
    
    @retry_llm_call(max_retries=3, backoff_factor=2.0)
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a completion response from the LLM.
        
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
        """
        payload = self._build_payload(prompt, system_prompt)
        response_data = self._make_request(payload)
        
        # Extract content from response
        try:
            choices = response_data.get("choices", [])
            if not choices:
                raise LLMResponseError("No choices in response")
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise LLMResponseError("Empty content in response")
            
            return content
            
        except (KeyError, TypeError) as e:
            raise LLMResponseError(
                f"Failed to parse response structure: {e}"
            ) from e
    
    @retry_llm_call(max_retries=3, backoff_factor=2.0)
    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Generate a structured JSON response from the LLM.
        
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
        """
        # Add JSON format instructions to system prompt
        json_instruction = (
            "Respond ONLY with valid JSON. Do not include any explanatory text, "
            "markdown formatting, or code blocks. Your entire response must be "
            "a valid JSON object that can be parsed directly."
        )
        
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{json_instruction}"
        else:
            system_prompt = json_instruction
        
        payload = self._build_payload(prompt, system_prompt, response_format="json")
        response_data = self._make_request(payload)
        
        # Extract and parse content
        try:
            choices = response_data.get("choices", [])
            if not choices:
                raise LLMResponseError("No choices in response")
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return {}
            
            # Parse JSON from response
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMResponseError(
                    f"Invalid JSON in response: {e}. Content: {content[:200]}"
                ) from e
                
        except (KeyError, TypeError) as e:
            raise LLMResponseError(
                f"Failed to parse response structure: {e}"
            ) from e

    async def _async_make_request(
        self,
        payload: dict,
    ) -> dict:
        """Make async HTTP request to the API.
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            Parsed JSON response
            
        Raises:
            LLMTimeoutError: On request timeout
            LLMCommunicationError: On network errors
            LLMError: On API errors
        """
        try:
            async with httpx.AsyncClient(timeout=self._async_timeout, trust_env=False) as client:
                response = await client.post(
                    self._endpoint,
                    headers=self._build_headers(),
                    json=payload,
                )
                
                if response.status_code != 200:
                    self._handle_error(response.status_code, response.text)
                
                return response.json()
                
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Request timed out after {self.timeout} seconds"
            ) from e
        except httpx.NetworkError as e:
            raise LLMCommunicationError(
                f"Network error: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise LLMCommunicationError(
                f"HTTP error: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Invalid JSON response from API: {e}"
            ) from e

    @retry_llm_call_async(max_retries=3, backoff_factor=2.0)
    async def _async_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Internal async completion method.
        
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
        """
        payload = self._build_payload(prompt, system_prompt)
        response_data = await self._async_make_request(payload)
        
        # Extract content from response
        try:
            choices = response_data.get("choices", [])
            if not choices:
                raise LLMResponseError("No choices in response")
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise LLMResponseError("Empty content in response")
            
            return content
            
        except (KeyError, TypeError) as e:
            raise LLMResponseError(
                f"Failed to parse response structure: {e}"
            ) from e

    async def acomplete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Async generate a completion response from the LLM.
        
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
        """
        return await self._async_complete(prompt, system_prompt)

    @retry_llm_call_async(max_retries=3, backoff_factor=2.0)
    async def _async_complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Internal async JSON completion method.
        
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
        """
        # Add JSON format instructions to system prompt
        json_instruction = (
            "Respond ONLY with valid JSON. Do not include any explanatory text, "
            "markdown formatting, or code blocks. Your entire response must be "
            "a valid JSON object that can be parsed directly."
        )
        
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{json_instruction}"
        else:
            system_prompt = json_instruction
        
        payload = self._build_payload(prompt, system_prompt, response_format="json")
        response_data = await self._async_make_request(payload)
        
        # Extract and parse content
        try:
            choices = response_data.get("choices", [])
            if not choices:
                raise LLMResponseError("No choices in response")
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return {}
            
            # Parse JSON from response
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMResponseError(
                    f"Invalid JSON in response: {e}. Content: {content[:200]}"
                ) from e
                
        except (KeyError, TypeError) as e:
            raise LLMResponseError(
                f"Failed to parse response structure: {e}"
            ) from e

    async def acomplete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Async generate a structured JSON response from the LLM.
        
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
        """
        return await self._async_complete_json(prompt, system_prompt)
