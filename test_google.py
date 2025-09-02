import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
print(f"Using API key: {api_key[:5]}...")  # Print first 5 chars for verification

# Configure the API
print("Configuring Google Generative AI...")
genai.configure(api_key=api_key)

# List available models
print("\nAvailable models:")
try:
    models = genai.list_models()
    for m in models:
        print(f"\nModel: {m.name}")
        print(f"Description: {m.description}")
        print(f"Supported methods: {', '.join(m.supported_generation_methods)}")
        
    # Try to create a model instance with the first model that supports generateContent
    print("\nTesting model creation...")
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"\nTrying to create model: {m.name}")
            try:
                model = genai.GenerativeModel(m.name.split('/')[-1])
                print(f"Successfully created model: {m.name}")
                break
            except Exception as e:
                print(f"Error with {m.name}: {str(e)}")
        
except Exception as e:
    print(f"Error: {str(e)}")
