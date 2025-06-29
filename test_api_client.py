#!/usr/bin/env python3
"""
Test script for the OpenAI API client.
Demonstrates various use cases including basic chat, function calling, and error handling.
"""

import asyncio
import json
import os
from api_client import (
    create_openai_client,
    make_api_call,
    calculate_max_tokens,
    MODEL_CONFIG,
)


async def test_basic_chat():
    """Test basic chat functionality"""
    print("🔄 Testing basic chat...")

    try:
        client = create_openai_client()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you tell me a short joke?"},
        ]

        response = await make_api_call(
            client=client, messages=messages, temperature=0.7, max_tokens=100
        )

        print("✅ Basic chat successful!")
        print(f"Model: {response.model}")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        print()

    except Exception as e:
        print(f"❌ Basic chat failed: {e}")
        return False

    return True


async def test_function_calling():
    """Test function calling capabilities"""
    print("🔄 Testing function calling...")

    try:
        client = create_openai_client()

        # Define a simple function for the model to call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can check the weather.",
            },
            {"role": "user", "content": "What's the weather like in New York?"},
        ]

        response = await make_api_call(
            client=client, messages=messages, temperature=0.3, tools=tools
        )

        print("✅ Function calling successful!")

        # Check if the model wants to call a function
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            print(f"Function called: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
        else:
            print(f"Response: {response.choices[0].message.content}")

        print(f"Tokens used: {response.usage.total_tokens}")
        print()

    except Exception as e:
        print(f"❌ Function calling failed: {e}")
        return False

    return True


async def test_token_calculation():
    """Test token calculation functionality"""
    print("🔄 Testing token calculation...")

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "This is a test message for token calculation.",
            },
        ]

        max_tokens = calculate_max_tokens(messages)
        print(f"✅ Token calculation successful!")
        print(f"Calculated max tokens: {max_tokens}")
        print(f"Model max context: {MODEL_CONFIG['max_context']}")
        print(f"Model max output: {MODEL_CONFIG['max_output']}")
        print()

    except Exception as e:
        print(f"❌ Token calculation failed: {e}")
        return False

    return True


async def test_error_handling():
    """Test error handling with invalid input"""
    print("🔄 Testing error handling...")

    try:
        client = create_openai_client()

        # Test with empty messages (should fail)
        messages = []

        response = await make_api_call(
            client=client, messages=messages, temperature=0.7
        )

        print("❌ Error handling test failed - should have thrown an error!")
        return False

    except Exception as e:
        print(
            f"✅ Error handling successful - caught expected error: {type(e).__name__}"
        )
        print()
        return True


async def test_long_conversation():
    """Test handling of longer conversations"""
    print("🔄 Testing long conversation...")

    try:
        client = create_openai_client()

        # Simulate a multi-turn conversation
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that gives concise answers.",
            },
            {"role": "user", "content": "What is Python?"},
            {
                "role": "assistant",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
            },
            {"role": "user", "content": "What are its main uses?"},
            {
                "role": "assistant",
                "content": "Python is commonly used for web development, data science, AI/ML, automation, and scripting.",
            },
            {"role": "user", "content": "Can you give me a simple example?"},
        ]

        response = await make_api_call(
            client=client, messages=messages, temperature=0.5, max_tokens=150
        )

        print("✅ Long conversation successful!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        print()

    except Exception as e:
        print(f"❌ Long conversation failed: {e}")
        return False

    return True


def check_environment():
    """Check if required environment variables are set"""
    print("🔍 Checking environment...")

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False

    print("✅ Environment check passed!")
    print(f"Using model: {MODEL_CONFIG['model_id']}")
    print()
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting OpenAI API Client Tests\n")

    # Check environment first
    if not check_environment():
        return

    # Run all tests
    tests = [
        ("Basic Chat", test_basic_chat),
        ("Function Calling", test_function_calling),
        ("Token Calculation", test_token_calculation),
        ("Error Handling", test_error_handling),
        ("Long Conversation", test_long_conversation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
