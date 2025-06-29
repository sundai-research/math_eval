#!/usr/bin/env python3
"""
Quick script to validate OpenAI API key
"""

import os
import asyncio
from api_client import create_openai_client, make_api_call


async def validate_api_key():
    """Test if the API key is valid with a minimal request"""
    print("🔍 Validating OpenAI API key...")

    # Check if key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False

    print(f"✅ API key found: {api_key[:20]}...{api_key[-4:]}")

    try:
        # Create client
        client = create_openai_client()
        print("✅ Client created successfully")

        # Test with minimal request
        messages = [{"role": "user", "content": "Hi"}]
        response = await make_api_call(
            client=client, messages=messages, temperature=0.1, max_tokens=5
        )

        print(f"✅ API key is VALID!")
        print(f"✅ Response: {response.choices[0].message.content}")
        print(f"✅ Model: {response.model}")
        return True

    except Exception as e:
        print(f"❌ API key validation failed: {e}")

        if "401" in str(e) or "invalid_api_key" in str(e):
            print("\n💡 This looks like an invalid API key.")
            print(
                "   Please check your key at: https://platform.openai.com/account/api-keys"
            )
        elif "insufficient_quota" in str(e):
            print("\n💡 API key is valid but you're out of credits.")
            print("   Please add credits to your OpenAI account.")
        else:
            print(f"\n💡 Unexpected error: {type(e).__name__}")

        return False


if __name__ == "__main__":
    success = asyncio.run(validate_api_key())
    if success:
        print("\n🎉 Ready to use the API client!")
    else:
        print("\n🔧 Please fix the API key issue above.")
