
# model-router/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx
import asyncio
import os
import logging
import textstat
from typing import Optional, Dict, Any
import time
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI(title="Model Router", version="1.0.0")

# Configuration
LIGHT_MODEL_ENDPOINT = os.getenv("LIGHT_MODEL_ENDPOINT", "http://ollama:11434")
HEAVY_MODEL_ENDPOINT = os.getenv("HEAVY_MODEL_ENDPOINT", "http://ollama:11434") 
COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.7"))

# Metrics
REQUEST_COUNT = Counter('model_router_requests_total', 'Total requests', ['model_type'])
REQUEST_DURATION = Histogram('model_router_request_duration_seconds', 'Request duration')

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}

class ModelResponse(BaseModel):
    response: str
    model_used: str
    complexity_score: float
    processing_time: float

def calculate_complexity(prompt: str, context: str = "") -> float:
    """Calculate complexity score based on various factors"""
    text = f"{prompt} {context or ''}"
    
    # Basic metrics
    word_count = len(text.split())
    sentence_count = textstat.sentence_count(text)
    
    # Readability metrics
    flesch_score = textstat.flesch_reading_ease(text)
    
    # Keyword-based complexity
    complex_keywords = [
        'analyze', 'synthesize', 'compare', 'evaluate', 'critique',
        'reasoning', 'logic', 'philosophy', 'mathematics', 'calculation',
        'code', 'programming', 'algorithm', 'complex', 'detailed'
    ]
    
    keyword_score = sum(1 for keyword in complex_keywords if keyword.lower() in text.lower())
    
    # Normalize scores
    word_score = min(word_count / 100, 1.0)  # Normalize to 0-1
    sentence_score = min(sentence_count / 10, 1.0)
    flesch_normalized = max(0, (100 - flesch_score) / 100)  # Lower flesch = higher complexity
    keyword_normalized = min(keyword_score / 5, 1.0)
    
    # Weighted average
    complexity = (word_score * 0.2 + sentence_score * 0.2 + 
                 flesch_normalized * 0.3 + keyword_normalized * 0.3)
    
    return complexity

async def call_model(endpoint: str, prompt: str, parameters: Dict[str, Any]) -> str:
    """Call the appropriate model endpoint"""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": parameters.get("model", "llama3.2"),
            "prompt": prompt,
            "stream": False,
            **parameters
        }
        
        response = await client.post(f"{endpoint}/api/generate", json=payload, timeout=120.0)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")

@app.post("/route", response_model=ModelResponse)
async def route_query(request: QueryRequest):
    """Route query to appropriate model based on complexity"""
    start_time = time.time()
    
    try:
        # Calculate complexity
        complexity_score = calculate_complexity(request.prompt, request.context)
        
        # Choose model based on complexity
        if complexity_score >= COMPLEXITY_THRESHOLD:
            endpoint = HEAVY_MODEL_ENDPOINT
            model_type = "heavy"
            request.parameters["model"] = request.parameters.get("model", "llama3.1:8b")
        else:
            endpoint = LIGHT_MODEL_ENDPOINT
            model_type = "light"
            request.parameters["model"] = request.parameters.get("model", "llama3.2:3b")
        
        # Make the request
        response = await call_model(endpoint, request.prompt, request.parameters)
        
        processing_time = time.time() - start_time
        
        # Update metrics
        REQUEST_COUNT.labels(model_type=model_type).inc()
        REQUEST_DURATION.observe(processing_time)
        
        return ModelResponse(
            response=response,
            model_used=model_type,
            complexity_score=complexity_score,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error routing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)