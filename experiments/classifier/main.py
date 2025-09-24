from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import re
import logging
from typing import Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Request Classifier", version="1.0.0")

class ClassificationRequest(BaseModel):
    message: str
    context: Dict[str, Any] = {}

class ClassificationResponse(BaseModel):
    complexity: str  # simple, medium, complex
    confidence: float
    route: str
    reasoning: str

class RequestClassifier:
    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        self.light_model = os.getenv("LIGHT_MODEL", "llama3.2:1b")
        
    def classify_complexity(self, message: str, context: Dict[str, Any] = {}) -> ClassificationResponse:
        """Classify request complexity using multiple heuristics"""
        
        # Simple heuristic classification
        simple_patterns = [
            r'^(hi|hello|hey)\b',
            r'^what is \w+\?$',
            r'^(yes|no|ok|thanks?)\b',
            r'^tell me about \w+$'
        ]
        
        complex_patterns = [
            r'analyze.*and.*compare',
            r'create.*application',
            r'implement.*algorithm',
            r'design.*system',
            r'write.*code.*for',
            r'solve.*problem.*with.*steps'
        ]
        
        message_lower = message.lower()
        word_count = len(message.split())
        
        # Check simple patterns
        for pattern in simple_patterns:
            if re.search(pattern, message_lower):
                return ClassificationResponse(
                    complexity="simple",
                    confidence=0.8,
                    route=f"http://{self.ollama_host}/api/generate",
                    reasoning="Matched simple conversational pattern"
                )
        
        # Check complex patterns
        for pattern in complex_patterns:
            if re.search(pattern, message_lower):
                return ClassificationResponse(
                    complexity="complex",
                    confidence=0.9,
                    route="http://flowise:3000/api/v1/prediction",
                    reasoning="Matched complex task pattern"
                )
        
        # Word count and question complexity heuristics
        if word_count < 5:
            complexity = "simple"
            confidence = 0.7
            route = f"http://{self.ollama_host}/api/generate"
        elif word_count > 20 or '?' in message and ('how' in message_lower or 'why' in message_lower):
            complexity = "medium"
            confidence = 0.6
            route = "http://flowise:3000/api/v1/prediction"
        else:
            complexity = "medium"
            confidence = 0.5
            route = "http://flowise:3000/api/v1/prediction"
            
        return ClassificationResponse(
            complexity=complexity,
            confidence=confidence,
            route=route,
            reasoning=f"Based on word count ({word_count}) and content analysis"
        )

classifier = RequestClassifier()

@app.post("/classify", response_model=ClassificationResponse)
async def classify_request(request: ClassificationRequest):
    """Classify the complexity of a user request"""
    try:
        result = classifier.classify_complexity(request.message, request.context)
        logger.info(f"Classified: '{request.message[:50]}...' as {result.complexity}")
        return result
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "request-classifier"}

@app.post("/chat")
async def chat_endpoint(request: ClassificationRequest):
    """Direct chat endpoint that classifies and routes"""
    classification = classifier.classify_complexity(request.message, request.context)
    
    # Route to appropriate service based on classification
    if classification.complexity == "simple":
        # Route to Ollama directly
        ollama_request = {
            "model": classifier.light_model,
            "prompt": request.message,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"http://{classifier.ollama_host}/api/generate",
                json=ollama_request,
                timeout=30
            )
            if response.status_code == 200:
                return {
                    "response": response.json().get("response", ""),
                    "classification": classification.dict(),
                    "routed_to": "ollama"
                }
        except Exception as e:
            logger.error(f"Ollama routing error: {str(e)}")
    
    return {
        "response": f"Request classified as {classification.complexity}. Route to: {classification.route}",
        "classification": classification.dict(),
        "routed_to": "classifier_only"
    }
