"""
API client utilities for making calls to LLM providers.
Includes retry logic and response handling.
"""

import os
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from tenacity import (
    retry,
    stop_after_delay,
    wait_random,
    retry_if_exception_type,
    RetryError,
)

# Model configuration
MODEL_CONFIG = {
    "provider": "openai",
    "model_id": "Qwen/Qwen3-32B",
    "tokenizer_id": "Qwen/Qwen3-32B",  # Using gpt2 tokenizer as approximation for OpenAI models
    "max_context": 128000,
    "max_output": 16384,  # gpt-4o supports up to 16K output tokens
    "default_max_tokens": 4096,
    "base_url": "https://e3c1-169-62-23-49.ngrok-free.app",
}

# Global tokenizer instance for efficiency
tokenizer = None


def get_tokenizer():
    """Get tokenizer instance (singleton pattern)"""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["tokenizer_id"])
    return tokenizer


@retry(
    stop=stop_after_delay(120),  # Shorter timeout
    wait=wait_random(min=2, max=10),  # Shorter wait times
    retry=retry_if_exception_type(
        (ConnectionError, TimeoutError)
    ),  # Don't retry auth errors
    reraise=True,
)
async def make_api_call(
    client: AsyncOpenAI,
    messages: List[Dict],
    temperature: float,
    max_tokens: int = None,
    tools: Optional[List[Dict]] = None,
):
    """Unified API call function with optional tool support and retry logic"""
    if max_tokens is None:
        max_tokens = calculate_max_tokens(messages)

    # Create request parameters
    params = {
        "model": MODEL_CONFIG["model_id"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
    }

    # Add tools if provided
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    response = await client.chat.completions.create(**params)

    # Always return the full response object
    return response


def calculate_max_tokens(messages: List[Dict], max_output: int = None) -> int:
    """Calculate maximum tokens available for generation"""
    if max_output is None:
        max_output = MODEL_CONFIG["max_output"]

    try:
        tokenizer_instance = get_tokenizer()
        prompt = tokenizer_instance.apply_chat_template(messages, tokenize=False)
        num_prompt_tokens = len(tokenizer_instance.encode(prompt))
        return min(max_output, MODEL_CONFIG["max_context"] - num_prompt_tokens)
    except Exception:
        return max_output  # Fallback


def create_openai_client() -> AsyncOpenAI:
    """Create AsyncOpenAI client with proper configuration"""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=30.0, base_url=MODEL_CONFIG["base_url"])
