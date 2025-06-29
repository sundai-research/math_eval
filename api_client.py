"""
API client utilities for making calls to LLM providers.
Includes retry logic and response handling.
"""
import os
from typing import Dict, List, Optional
from groq import AsyncGroq
from transformers import AutoTokenizer
from tenacity import retry, stop_after_delay, wait_random, retry_if_exception_type, RetryError

# Model configuration
MODEL_CONFIG = {
    "provider": "groq",
    "model_id": "qwen/qwen3-32b",
    "tokenizer_id": "qwen/qwen3-32b",
    "max_context": 128000,
    "max_output": 40960,
    "default_max_tokens": 40960
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
    stop=stop_after_delay(600),
    wait=wait_random(min=10, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def make_api_call(
    client: AsyncGroq, 
    messages: List[Dict], 
    temperature: float, 
    max_tokens: int = None, 
    tools: Optional[List[Dict]] = None
):
    """Unified API call function with optional tool support and retry logic"""
    if max_tokens is None:
        max_tokens = calculate_max_tokens(messages)
    
    # Create request parameters
    params = {
        "model": MODEL_CONFIG["model_id"],
        "messages": messages,
        "temperature": temperature,
        "parallel_tool_calls": False,
        "max_tokens": max_tokens,
        "top_p": 0.95
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


def create_groq_client() -> AsyncGroq:
    """Create AsyncGroq client with proper configuration"""
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    return AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"), timeout=600)