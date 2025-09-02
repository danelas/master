import requests
import json

def test_server():
    url = "http://localhost:8000/query"
    headers = {"Content-Type": "application/json"}
    data = {
        "question": "What is the capital of France?",
        "models": ["openai", "anthropic", "google"]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_server()
