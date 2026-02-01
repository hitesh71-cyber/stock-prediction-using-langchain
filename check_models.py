#!/usr/bin/env python3
"""
Test script to check which Gemini models are available with your API key
Run this to see what models you can actually use
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    print("Create a .env file with: GOOGLE_API_KEY=your_key_here")
    exit(1)

print("=" * 60)
print("🔍 Checking available Gemini models...")
print("=" * 60)

# Configure Gemini
genai.configure(api_key=api_key)

# List all models
print("\n📋 All models that support generateContent:\n")
available_models = []

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
            print(f"✓ {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description}")
            print()
except Exception as e:
    print(f"❌ Error listing models: {e}")
    exit(1)

if not available_models:
    print("❌ No models found! Check your API key.")
    exit(1)

print("=" * 60)
print(f"✅ Found {len(available_models)} available models")
print("=" * 60)

# Test each model
print("\n🧪 Testing models...\n")

working_models = []

test_models = [
    'gemini-2.0-flash-exp',
    'gemini-1.5-flash-002',
    'gemini-1.5-flash-001',
    'gemini-1.5-flash',
    'gemini-1.5-pro-002',
    'gemini-1.5-pro-001',
    'gemini-1.5-pro',
    'gemini-pro',
]

for model_name in test_models:
    try:
        print(f"Testing: {model_name}...", end=" ")
        llm = genai.GenerativeModel(model_name)
        response = llm.generate_content("Say hello in one word")
        print(f"✅ WORKS - Response: {response.text.strip()}")
        working_models.append(model_name)
    except Exception as e:
        error_str = str(e)
        if "404" in error_str:
            print(f"❌ NOT FOUND")
        else:
            print(f"❌ ERROR: {error_str[:50]}")

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

if working_models:
    print(f"\n✅ {len(working_models)} working models found:\n")
    for i, model in enumerate(working_models, 1):
        print(f"   {i}. {model}")
    
    print(f"\n💡 RECOMMENDED: Use '{working_models[0]}' in your main.py")
    print(f"\n📝 Update your main.py with this model:")
    print(f"   llm = genai.GenerativeModel('{working_models[0]}')")
else:
    print("\n❌ No working models found!")
    print("\nPossible issues:")
    print("1. API key doesn't have access to Gemini models")
    print("2. API key is invalid or expired")
    print("3. Need to enable Gemini API in Google Cloud Console")
    print("\n🔗 Get API key: https://makersuite.google.com/app/apikey")

print("\n" + "=" * 60)
