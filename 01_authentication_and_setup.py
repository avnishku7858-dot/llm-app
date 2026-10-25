"""
Objective 1: Understand LLM APIs, Authentication, Environment Variables, and Connect LLMs Using Python

Key Concepts Covered:
1. What is an LLM API? (Stateless HTTP/REST service with tokenized input/output)
2. Safe Credential Management (Environment variables via .env, never hardcoding secrets)
3. Initializing LLM Clients (Official Google GenAI SDK & fallback OpenAI client support)
4. Connectivity Verification ("Ping" test and model availability check)
"""

import os
import sys
from dotenv import load_dotenv

# 1. Secure Environment Variable Loading
# Loads key-value pairs from .env into os.environ
load_dotenv()


def check_environment_setup() -> dict:
    """Verifies presence of API keys in environment variables."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    status = {
        "gemini_configured": bool(gemini_key and gemini_key != "your_gemini_api_key_here"),
        "openai_configured": bool(openai_key and openai_key != "your_openai_api_key_here"),
        "gemini_key_masked": f"{gemini_key[:6]}...{gemini_key[-4:]}" if gemini_key and len(gemini_key) > 10 else "Not Set",
        "openai_key_masked": f"{openai_key[:6]}...{openai_key[-4:]}" if openai_key and len(openai_key) > 10 else "Not Set",
    }
    return status


def connect_gemini(model_name: str = "gemini-3.8-flash"):
    """
    Connects to the Gemini API using the modern google-genai SDK.
    Uses genai.Client() with explicit or environment-loaded credentials.
    """
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file or environment."
        )

    # Initialize client (it automatically picks up GEMINI_API_KEY from environment)
    client = genai.Client(api_key=api_key)

    print(f"[*] Sending connectivity ping to model: {model_name}...")
    # Generate content using modern interactions API or models API
    try:
        interaction = client.interactions.create(
            model=model_name,
            input="Hello! Confirm connection in 1 short sentence."
        )
        response_text = interaction.output_text
    except Exception:
        # Fallback to generate_content if interactions endpoint differs in environment
        response = client.models.generate_content(
            model=model_name,
            contents="Hello! Confirm connection in 1 short sentence."
        )
        response_text = response.text

    return response_text


def connect_openai(model_name: str = "gpt-4o-mini"):
    """
    Connects to OpenAI API using the official openai SDK.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in your .env file or environment."
        )

    client = OpenAI(api_key=api_key)
    print(f"[*] Sending connectivity ping to OpenAI model: {model_name}...")
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello! Confirm connection in 1 short sentence."}],
        max_tokens=60
    )
    return completion.choices[0].message.content


def main():
    print("=" * 65)
    print("  Objective 1: LLM APIs, Authentication & Environment Variables")
    print("=" * 65)

    env_status = check_environment_setup()
    print(f"Gemini API Key Detected : {env_status['gemini_configured']} ({env_status['gemini_key_masked']})")
    print(f"OpenAI API Key Detected : {env_status['openai_configured']} ({env_status['openai_key_masked']})")
    print("-" * 65)

    if not env_status["gemini_configured"] and not env_status["openai_configured"]:
        print("[!] No active API key found in .env file.")
        print("[!] Please copy .env.example to .env and add your GEMINI_API_KEY or OPENAI_API_KEY.")
        print("[*] Demonstration Mode: Simulation of successful client initialization:")
        print("    1. load_dotenv() reads .env")
        print("    2. os.getenv('GEMINI_API_KEY') retrieves token")
        print("    3. genai.Client(api_key=...) authenticates session")
        print("    4. client.interactions.create(...) sends payload via HTTPS")
        print("=" * 65)
        return

    # Attempt connection with configured provider
    if env_status["gemini_configured"]:
        try:
            gemini_response = connect_gemini()
            print("[+] Gemini API Connection Successful!")
            print(f"    Model Response: {gemini_response.strip()}")
        except Exception as e:
            print(f"[-] Gemini Connection Failed: {e}")

    if env_status["openai_configured"]:
        try:
            openai_response = connect_openai()
            print("[+] OpenAI API Connection Successful!")
            print(f"    Model Response: {openai_response.strip()}")
        except Exception as e:
            print(f"[-] OpenAI Connection Failed: {e}")

    print("=" * 65)


if __name__ == "__main__":
    main()
