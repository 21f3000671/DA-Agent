#!/usr/bin/env python3
"""Test script to verify LLM API connection and functionality."""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_llm_connection():
    """Test the LLM API connection with a simple prompt."""
    
    # Get configuration from environment
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: No API key found!")
        print("Please set LLM_API_KEY or OPENAI_API_KEY in your .env file")
        return False
    
    print(f"🔧 Testing LLM Configuration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    if base_url:
        print(f"  Base URL: {base_url}")
    print()
    
    try:
        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = OpenAI(**client_kwargs)
        
        # Test with a simple prompt
        print("📤 Sending test prompt to LLM...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please respond with 'LLM connection successful!' and nothing else."}
            ],
            max_tokens=50,
            temperature=0
        )
        
        # Get the response
        result = response.choices[0].message.content.strip()
        
        print(f"📥 Response received: {result}")
        print()
        
        if "successful" in result.lower():
            print("✅ LLM API connection test PASSED!")
            return True
        else:
            print("⚠️  Unexpected response, but connection works")
            return True
            
    except Exception as e:
        print(f"❌ Error connecting to LLM API: {str(e)}")
        print()
        print("Troubleshooting tips:")
        print("1. Check your API key is valid")
        print("2. Verify the model name is correct")
        print("3. If using a custom provider, check the base URL")
        print("4. Ensure you have internet connectivity")
        return False

def test_llm_with_tools():
    """Test the LLM with function calling capability."""
    
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o")
    base_url = os.getenv("LLM_BASE_URL")
    
    print("\n🔧 Testing LLM with Function Calling:")
    
    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = OpenAI(**client_kwargs)
        
        # Define a simple test tool
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"}
                    },
                    "required": ["location"]
                }
            }
        }]
        
        # Test function calling
        print("📤 Testing function calling capability...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather in New York?"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            print(f"📥 Tool call received: {tool_call.function.name}")
            print(f"   Arguments: {tool_call.function.arguments}")
            print("✅ Function calling test PASSED!")
            return True
        else:
            print("⚠️  No tool calls in response")
            return False
            
    except Exception as e:
        print(f"❌ Error testing function calling: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("DA-Agent LLM API Test")
    print("="*50)
    print()
    
    # Run basic connection test
    basic_test = test_llm_connection()
    
    # Run function calling test if basic test passes
    if basic_test:
        tools_test = test_llm_with_tools()
        
        print("\n" + "="*50)
        if tools_test:
            print("🎉 All tests PASSED! LLM is working correctly.")
        else:
            print("⚠️  Basic connection works but function calling failed.")
            print("This may affect the agent's ability to use tools.")
    else:
        print("\n" + "="*50)
        print("❌ LLM connection failed. Please check your configuration.")
        sys.exit(1)