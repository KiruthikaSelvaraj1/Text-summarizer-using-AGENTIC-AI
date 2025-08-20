import requests
import json

# Test the Gemini API directly
api_key = "AIzaSyC_savsQuk3rWoLoy3FWyVJCX2UgKw2PsQ"

# Test data
data = {
    "contents": [
        {
            "parts": [
                {
                    "text": " Machine learning (ML) allows computers to learn from data without explicit programming.  It uses algorithms to identify patterns, make predictions, and improve its performance over time.  This involves feeding the algorithm large datasets, allowing it to build a model representing underlying relationships.  The model then uses this learned knowledge to analyze new data and produce outputs like classifications, predictions, or decisions.  Different ML techniques like supervised, unsupervised, and reinforcement learning cater to various data types and tasks summarize the text in 10 words"
                }
            ]
        }
    ]
}

headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": api_key
}

print("Testing Gemini API...")
print("=" * 50)

# Test gemini-2.0-flash
try:
    print("Testing gemini-2.0-flash...")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        print("✅ SUCCESS with gemini-2.0-flash!")
        print(f"Response: {text}")
    else:
        print(f"❌ ERROR with gemini-2.0-flash: {response.status_code}")
        print(f"Error details: {response.text}")
        
except Exception as e:
    print(f"❌ EXCEPTION with gemini-2.0-flash: {e}")

print("\n" + "=" * 50)

# Test gemini-1.5-flash as fallback
try:
    print("Testing gemini-1.5-flash...")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        print("✅ SUCCESS with gemini-1.5-flash!")
        print(f"Response: {text}")
    else:
        print(f"❌ ERROR with gemini-1.5-flash: {response.status_code}")
        print(f"Error details: {response.text}")
        
except Exception as e:
    print(f"❌ EXCEPTION with gemini-1.5-flash: {e}")
