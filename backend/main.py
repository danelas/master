from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any
import os
from pathlib import Path
import json
import re
from collections import Counter
from difflib import SequenceMatcher
import openai
import anthropic
import google.generativeai as genai
import groq
from dotenv import load_dotenv

# Initialize Groq client
grok_client = None
try:
    grok_api_key = os.getenv("GROK_API_KEY")
    if grok_api_key:
        grok_client = groq.Client(api_key=grok_api_key)
        print("Successfully initialized Groq client")
    else:
        print("Warning: GROK_API_KEY not found in environment variables")
except Exception as e:
    print(f"Warning: Failed to initialize Groq client: {e}")
    grok_client = None

# Load environment variables
load_dotenv()

# Models
class AIResponse(BaseModel):
    model: str
    response: str
    status: str

class CombinedResponse(BaseModel):
    summary: str
    key_points: List[str]
    contradictions: List[Dict[str, str]]
    consensus_percentage: float

def analyze_responses(question: str, responses: Dict[str, AIResponse]) -> Dict:
    """Analyze and combine multiple AI responses"""
    # Extract successful responses
    successful_responses = {k: v.response for k, v in responses.items() if v.status == "success"}
    
    if not successful_responses:
        return {
            "summary": "No successful responses from any model.",
            "key_points": [],
            "contradictions": [],
            "consensus_percentage": 0.0
        }
    
    # Format the responses for the prompt
    responses_text = "\n\n".join([f"{model.upper()}:\n{response}" for model, response in successful_responses.items()])
    
    # Groq client is now initialized at the module level

    # Use OpenAI to analyze and combine responses
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a prompt to analyze the responses
    analysis_prompt = f"""
    You are an expert analyst comparing responses from multiple AI models to the following question:
    
    QUESTION: {question}
    
    Here are the responses from different AI models:
    {responses_text}
    
    Your task is to:
    1. Identify the key points that are consistent across most responses
    2. Note any significant contradictions or differences between the responses
    3. Create a comprehensive, well-structured combined response that incorporates the best elements from each
    4. Calculate the percentage of consensus between the models
    
    Return your analysis as a JSON object with the following structure:
    {{
        "summary": "A brief summary of the overall findings",
        "key_points": ["List of key points that most models agreed on"],
        "contradictions": [
            {{
                "issue": "Description of the contradiction",
                "models": ["model1", "model2"],
                "responses": ["response1", "response2"]
            }}
        ],
        "consensus_percentage": 0.0  # 0-100% how much the models agree
    }}
    """
    
    # Get the analysis from OpenAI
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert analyst that compares and combines responses from multiple AI models."},
            {"role": "user", "content": analysis_prompt.format(question=question, responses_text=responses_text)}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    try:
        # Parse the JSON response
        analysis = json.loads(response.choices[0].message.content)
        return analysis
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "summary": "Analysis completed, but could not parse the detailed results.",
            "key_points": [],
            "contradictions": [],
            "consensus_percentage": 0.0
        }

def generate_combined_response(question: str, responses: Dict[str, AIResponse]) -> Dict[str, Any]:
    """Generate a combined response by analyzing multiple AI responses."""
    # Filter out failed responses
    successful_responses = {}
    for model, response in responses.items():
        if response.status == "success":
            successful_responses[model] = response.response
    # First, analyze the responses
    analysis = analyze_responses(question, responses)
    
    # Get successful responses
    successful_responses = {k: v.response for k, v in responses.items() if v.status == "success"}
    
    # If we have at least one successful response, use it to generate a combined answer
    if successful_responses:
        # Use the first successful response as a base
        base_response = next(iter(successful_responses.values()))
        
        # Format the responses and analysis for the prompt
        responses_text = "\n\n".join([f"{model.upper()}:\n{response}" for model, response in successful_responses.items()])
        analysis_text = json.dumps(analysis, indent=2)
        
        # Create a prompt to generate a combined response
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""
        You are an expert at synthesizing information from multiple AI model responses.
        
        QUESTION: {question}
        
        Here are the original responses from different AI models:
        {responses_text}
        
        Here's the analysis of these responses:
        {analysis_text}
        
        Your task is to create a SINGLE, WELL-STRUCTURED response that:
        1. Incorporates the best elements from each response
        2. Resolves any contradictions in favor of the most accurate information
        3. Is clear, concise, and comprehensive
        4. Acknowledges any remaining uncertainties or areas of disagreement
        
        Make sure to:
        - Start with a brief introduction
        - Organize the content with clear headings
        - Use bullet points for key information
        - Highlight areas of consensus and disagreement
        - End with a concise summary
        
        FINAL COMBINED RESPONSE:
        """
        
        # Get the combined response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at synthesizing information from multiple AI model responses."},
                {"role": "user", "content": prompt.format(
                    question=question,
                    responses_text=responses_text,
                    analysis_text=analysis_text
                )}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Add the combined response to the analysis
        analysis["combined_answer"] = response.choices[0].message.content
    
    return analysis

# Initialize the FastAPI app
app = FastAPI(title="AI Response Aggregator API", docs_url=None, redoc_url=None)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Serve static files from the frontend directory
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Serve the frontend files
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Handle root path
    if not full_path or full_path == "index.html":
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    
    # Try to serve the requested file
    file_path = FRONTEND_DIR / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    
    # For SPA routing, serve index.html and let the frontend handle routing
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add middleware to handle preflight requests
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

class QueryRequest(BaseModel):
    question: str
    models: List[str] = ["openai", "anthropic", "google", "grok"]  # Default to all models

class QueryResponse(BaseModel):
    query_id: str
    question: str
    responses: Dict[str, AIResponse]
    summary: Optional[str] = None
    combined_response: Optional[Dict] = None

def query_ai_model(model: str, question: str) -> AIResponse:
    """Query the specified AI model with the given question"""
    try:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        print(f"Querying {model} with question: {question[:50]}...")
        if model == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            print(f"Using OpenAI API with key: {openai_api_key[:5]}...")  # Log first 5 chars of key
            
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate and concise answers."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content
            print(f"Successfully got response from {model}, length: {len(response_text)} chars")
            return AIResponse(
                model=model,
                response=response_text,
                status="success"
            )
        
        elif model == "anthropic":
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7,
                system="You are a helpful assistant that provides accurate and concise answers.",
                messages=[{"role": "user", "content": question}]
            )
            return AIResponse(
                model=model,
                response=response.content[0].text,
                status="success"
            )
            
        elif model == "google":
            try:
                import requests
                import json
                
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment variables")
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                
                headers = {
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": question
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topP": 0.9,
                        "topK": 40,
                        "maxOutputTokens": 1000
                    }
                }
                
                print(f"Sending request to Google Gemini API...")
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()  # Raise exception for HTTP errors
                
                response_data = response.json()
                print(f"Google API response: {json.dumps(response_data, indent=2)}")
                
                # Extract the response text
                if 'candidates' in response_data and response_data['candidates']:
                    response_text = response_data['candidates'][0]['content']['parts'][0]['text']
                else:
                    raise ValueError("Unexpected response format from Google API")
                
                return AIResponse(
                    model="google",
                    response=response_text,
                    status="success"
                )
                
            except Exception as e:
                error_msg = f"Error querying Google Gemini API: {str(e)}"
                print(error_msg)
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"API Response: {response.text}")
                return AIResponse(
                    model="google",
                    response=f"Error: {str(e)}",
                    status="error"
                )
            
        elif model == "grok":
            try:
                if not grok_client:
                    raise ValueError("Grok API key not properly configured. Please check your GROK_API_KEY in the .env file.")
                
                print("Sending request to Grok API...")
                response = grok_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are a helpful and concise assistant."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Handle the response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    response_text = response.choices[0].message.content
                else:
                    raise ValueError("Unexpected response format from Grok API")
                
                return AIResponse(
                    model="grok",
                    response=response_text,
                    status="success"
                )
                
            except Exception as e:
                error_msg = f"Error querying grok: {str(e)}"
                print(error_msg)
                return AIResponse(
                    model="grok",
                    response=f"Error: {str(e)}",
                    status="error"
                )
            
        return AIResponse(
            model=model,
            response=f"Unsupported model: {model}",
            status="error"
        )
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Error in query_ai_model for {model}: {error_details}")
        return AIResponse(
            model=model,
            response=f"Error querying {model}: {str(e)}",
            status="error"
        )

# Routes
@app.get("/api")
async def root():
    return {"message": "AI Response Aggregator API is running"}

@app.post("/api/query")
async def query_models(request: QueryRequest):
    """Query multiple AI models with the same question."""
    # Generate a unique ID for this query
    query_str = f"{'|'.join(request.models)}|{request.question}"
    query_id = f"query_{hash(query_str)}"
    
    # Validate models
    valid_models = ["openai", "anthropic", "google", "grok"]
    models_to_query = [model for model in request.models if model in valid_models]
    
    if not models_to_query:
        raise HTTPException(status_code=400, detail="No valid models specified")
    
    print(f"\n{'='*50}")
    print(f"Processing query: {request.question}")
    print(f"Querying models: {models_to_query}")
    
    # Query each model asynchronously
    responses = {}
    for model in models_to_query:
        try:
            print(f"\nQuerying {model}...")
            response = query_ai_model(model, request.question)
            responses[model] = response
            print(f"{model.upper()} response: {response.response[:100]}..." if len(response.response) > 100 else f"{model.upper()} response: {response.response}")
        except Exception as e:
            error_msg = f"Error querying {model}: {str(e)}"
            print(error_msg)
            responses[model] = AIResponse(
                model=model,
                response=f"Error: {str(e)}",
                status="error"
            )
    
    # Generate a combined response if we have at least one successful response
    combined_response = None
    successful_responses = {k: v for k, v in responses.items() if v.status == "success"}
    
    if len(successful_responses) == 1:
        # If only one model responded successfully, use its response directly
        model, response = next(iter(successful_responses.items()))
        combined_response = {
            "summary": response.response,
            "key_points": [response.response],
            "contradictions": [],
            "consensus_percentage": 100.0
        }
        print("\nUsing single model response as combined response")
    elif len(successful_responses) > 1:
        # If multiple models responded, generate a combined response
        try:
            print("\nGenerating combined response...")
            combined_response = generate_combined_response(request.question, responses)
            print("Successfully generated combined response")
        except Exception as e:
            error_msg = f"Error generating combined response: {str(e)}"
            print(error_msg)
            combined_response = {
                "summary": f"Error generating combined response: {str(e)[:200]}",
                "key_points": [],
                "contradictions": [str(e)[:500]],
                "consensus_percentage": 0.0
            }
    else:
        print("\nNo successful responses to combine")
    
    # Prepare the response
    response_data = {
        "query_id": query_id,
        "question": request.question,
        "responses": {k: v.dict() for k, v in responses.items()}
    }
    
    if combined_response:
        response_data["combined_response"] = combined_response
    
    print(f"\nQuery completed. Successful responses: {len(successful_responses)}/{len(models_to_query)}")
    print(f"Combined response generated: {'Yes' if combined_response else 'No'}")
    print(f"{'='*50}\n")
    
    try:
        return response_data
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Unexpected error in query_models: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
