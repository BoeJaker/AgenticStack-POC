"""
Knowledge Graph Ollama Shim - Main FastAPI Application
Intercepts Ollama API calls and builds knowledge graphs with tiered classification
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import aioredis
import httpx
import spacy
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
SERVICE_NAME = os.getenv("SERVICE_NAME", "kg-ollama-shim")
VERSION = os.getenv("VERSION", "1.0.0")

# Pydantic models for API requests
class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = False
    raw: Optional[bool] = False
    format: Optional[str] = None
    keep_alive: Optional[Union[int, str]] = None
    options: Optional[Dict[str, Any]] = None

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False
    format: Optional[str] = None
    keep_alive: Optional[Union[int, str]] = None
    options: Optional[Dict[str, Any]] = None

# Initialize services
app = FastAPI(title="Knowledge Graph Ollama Shim", version=VERSION)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Global variables for services
redis_client = None
nlp = None

@app.on_event("startup")
async def startup_event():
    global redis_client, nlp
    
    # Initialize Redis
    redis_client = await aioredis.from_url(REDIS_URL)
    
    # Initialize spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model successfully")
    except OSError:
        logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        nlp = None

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()
    neo4j_driver.close()

class KnowledgeGraphExtractor:
    """Handles tiered extraction and classification"""
    
    @staticmethod
    def create_content_hash(text: str) -> str:
        """Create SHA-256 hash of content for deduplication"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def extract_metadata(request_data: Dict, source_info: Dict) -> Dict:
        """Extract comprehensive metadata from request"""
        now = datetime.now(timezone.utc)
        return {
            "session_id": str(uuid.uuid4()),
            "content_hash": KnowledgeGraphExtractor.create_content_hash(
                json.dumps(request_data, sort_keys=True)
            ),
            "timestamp": now.isoformat(),
            "epoch_timestamp": int(now.timestamp()),
            "service_name": SERVICE_NAME,
            "service_version": VERSION,
            "model_used": request_data.get("model", "unknown"),
            "source_ip": source_info.get("client_ip"),
            "user_agent": source_info.get("user_agent"),
            "request_type": source_info.get("endpoint"),
            "stream_mode": request_data.get("stream", False),
            "processing_stage": "tier_0_light",
            "confidence": 0.0,
            "provenance": {
                "created_at": now.isoformat(),
                "created_by": SERVICE_NAME,
                "version": VERSION,
                "processing_job_id": str(uuid.uuid4()),
                "data_source": "ollama_shim_intercept"
            }
        }
    
    @staticmethod
    def tier_0_light_extraction(text: str, nlp_model=None) -> Dict:
        """Fast regex + spaCy extraction (Tier 0)"""
        extraction = {
            "entities": [],
            "code_elements": [],
            "urls": [],
            "temporal_markers": [],
            "classifications": [],
            "confidence": 0.3,
            "processing_tier": "tier_0_light"
        }
        
        # URL extraction
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        for url in urls:
            extraction["urls"].append({
                "url": url,
                "domain": re.search(r'https?://([^/]+)', url).group(1) if re.search(r'https?://([^/]+)', url) else None,
                "confidence": 0.9
            })
        
        # Code detection
        code_patterns = [
            (r'```[\s\S]*?```', 'code_block'),
            (r'`[^`]+`', 'inline_code'),
            (r'\bdef\s+\w+\s*\(', 'python_function'),
            (r'\bclass\s+\w+\s*:', 'python_class'),
            (r'\bfunction\s+\w+\s*\(', 'js_function'),
            (r'[a-zA-Z_]\w*\s*=\s*[^=]', 'assignment')
        ]
        
        for pattern, code_type in code_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                extraction["code_elements"].append({
                    "text": match.group(0),
                    "type": code_type,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.7
                })
        
        # Temporal markers
        temporal_patterns = [
            (r'\b(?:yesterday|today|tomorrow|now|currently|recently|soon|later)\b', 'relative_time'),
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'date'),
            (r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b', 'time')
        ]
        
        for pattern, temporal_type in temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                extraction["temporal_markers"].append({
                    "text": match.group(0),
                    "type": temporal_type,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.6
                })
        
        # spaCy NER if available
        if nlp_model:
            doc = nlp_model(text[:10000])  # Limit text for performance
            for ent in doc.ents:
                extraction["entities"].append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.8
                })
        
        # Basic classification
        classifications = []
        if any(word in text.lower() for word in ['error', 'exception', 'bug', 'issue', 'problem']):
            classifications.append({"type": "bug_report", "confidence": 0.7})
        if any(word in text.lower() for word in ['how to', 'help', 'explain', '?']):
            classifications.append({"type": "question", "confidence": 0.6})
        if len(extraction["code_elements"]) > 0:
            classifications.append({"type": "code_related", "confidence": 0.8})
        if len(extraction["urls"]) > 0:
            classifications.append({"type": "web_reference", "confidence": 0.9})
        
        extraction["classifications"] = classifications
        return extraction

class Neo4jIngestor:
    """Handles Neo4j graph ingestion with proper batching"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def create_indexes(self):
        """Create necessary indexes for performance"""
        with self.driver.session() as session:
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (p:Prompt) ON (p.session_id)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Prompt) ON (p.content_hash)",
                "CREATE INDEX IF NOT EXISTS FOR (r:Response) ON (r.session_id)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.canonical_id)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.url)",
                "CREATE INDEX IF NOT EXISTS FOR (c:CodeElement) ON (c.name)",
                "CREATE INDEX IF NOT EXISTS FOR (j:ProcessingJob) ON (j.job_id)",
                "CREATE INDEX IF NOT EXISTS FOR (n) ON (n.timestamp)",
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
    
    def ingest_prompt_response(self, prompt_data: Dict, response_data: Dict, extraction: Dict, metadata: Dict):
        """Batch ingest prompt, response, and extracted knowledge"""
        
        with self.driver.session() as session:
            session.write_transaction(
                self._ingest_batch,
                prompt_data, response_data, extraction, metadata
            )
    
    def _ingest_batch(self, tx, prompt_data: Dict, response_data: Dict, extraction: Dict, metadata: Dict):
        """Single transaction for all related data"""
        
        # Create Prompt node
        tx.run("""
            MERGE (p:Prompt {session_id: $session_id})
            SET p += {
                content_hash: $content_hash,
                text: $prompt_text,
                model: $model,
                system_prompt: $system_prompt,
                raw_request: $raw_request,
                timestamp: $timestamp,
                epoch_timestamp: $epoch_timestamp,
                service_name: $service_name,
                service_version: $service_version,
                source_ip: $source_ip,
                user_agent: $user_agent,
                request_type: $request_type,
                stream_mode: $stream_mode,
                created_at: $created_at,
                created_by: $created_by,
                processing_job_id: $processing_job_id,
                data_source: $data_source,
                updated_at: datetime()
            }
        """, 
            session_id=metadata["session_id"],
            content_hash=metadata["content_hash"],
            prompt_text=prompt_data.get("prompt", ""),
            model=prompt_data.get("model", ""),
            system_prompt=prompt_data.get("system", ""),
            raw_request=json.dumps(prompt_data),
            timestamp=metadata["timestamp"],
            epoch_timestamp=metadata["epoch_timestamp"],
            service_name=metadata["service_name"],
            service_version=metadata["service_version"],
            source_ip=metadata["source_ip"],
            user_agent=metadata["user_agent"],
            request_type=metadata["request_type"],
            stream_mode=metadata["stream_mode"],
            created_at=metadata["provenance"]["created_at"],
            created_by=metadata["provenance"]["created_by"],
            processing_job_id=metadata["provenance"]["processing_job_id"],
            data_source=metadata["provenance"]["data_source"]
        )
        
        # Create Response node and relationship
        if response_data:
            tx.run("""
                MATCH (p:Prompt {session_id: $session_id})
                MERGE (r:Response {session_id: $session_id})
                SET r += {
                    text: $response_text,
                    model: $model,
                    raw_response: $raw_response,
                    timestamp: $timestamp,
                    created_at: $created_at,
                    processing_time_ms: $processing_time_ms,
                    token_count: $token_count,
                    updated_at: datetime()
                }
                MERGE (p)-[:ANSWERED_BY {
                    timestamp: $timestamp,
                    processing_time_ms: $processing_time_ms,
                    confidence: $confidence
                }]->(r)
            """,
                session_id=metadata["session_id"],
                response_text=response_data.get("response", ""),
                model=response_data.get("model", ""),
                raw_response=json.dumps(response_data),
                timestamp=metadata["timestamp"],
                created_at=metadata["provenance"]["created_at"],
                processing_time_ms=response_data.get("total_duration", 0) // 1000000,
                token_count=len(response_data.get("response", "").split()),
                confidence=extraction.get("confidence", 0.0)
            )
        
        # Batch create entities using UNWIND
        if extraction.get("entities"):
            entities_data = []
            for entity in extraction["entities"]:
                entities_data.append({
                    "name": entity["text"].strip(),
                    "label": entity.get("label", "UNKNOWN"),
                    "canonical_id": self._create_canonical_id(entity["text"]),
                    "text": entity["text"],
                    "start_pos": entity.get("start", 0),
                    "end_pos": entity.get("end", 0),
                    "confidence": entity.get("confidence", 0.5),
                    "extraction_method": "spacy_ner",
                    "processing_tier": extraction.get("processing_tier", "tier_0"),
                    "created_at": metadata["provenance"]["created_at"],
                    "session_id": metadata["session_id"]
                })
            
            tx.run("""
                UNWIND $entities_data AS entity_data
                MERGE (e:Entity {canonical_id: entity_data.canonical_id})
                ON CREATE SET e += {
                    name: entity_data.name,
                    label: entity_data.label,
                    first_seen: entity_data.created_at,
                    created_at: entity_data.created_at,
                    occurrence_count: 1
                }
                ON MATCH SET e.occurrence_count = coalesce(e.occurrence_count, 0) + 1
                SET e += {
                    text: entity_data.text,
                    confidence: entity_data.confidence,
                    extraction_method: entity_data.extraction_method,
                    processing_tier: entity_data.processing_tier,
                    updated_at: datetime()
                }
                WITH e, entity_data
                MATCH (p:Prompt {session_id: entity_data.session_id})
                MERGE (p)-[:MENTIONS {
                    confidence: entity_data.confidence,
                    start_pos: entity_data.start_pos,
                    end_pos: entity_data.end_pos,
                    text_excerpt: entity_data.text,
                    extraction_method: entity_data.extraction_method,
                    created_at: entity_data.created_at
                }]->(e)
            """, entities_data=entities_data)
        
        # Batch create URLs
        if extraction.get("urls"):
            urls_data = []
            for url_info in extraction["urls"]:
                urls_data.append({
                    "url": url_info["url"],
                    "domain": url_info.get("domain", ""),
                    "confidence": url_info.get("confidence", 0.9),
                    "created_at": metadata["provenance"]["created_at"],
                    "session_id": metadata["session_id"],
                    "status": "pending_crawl"
                })
            
            tx.run("""
                UNWIND $urls_data AS url_data
                MERGE (d:Document {url: url_data.url})
                ON CREATE SET d += {
                    domain: url_data.domain,
                    first_referenced: url_data.created_at,
                    status: url_data.status,
                    reference_count: 1,
                    created_at: url_data.created_at
                }
                ON MATCH SET d.reference_count = coalesce(d.reference_count, 0) + 1
                SET d.updated_at = datetime()
                WITH d, url_data
                MATCH (p:Prompt {session_id: url_data.session_id})
                MERGE (p)-[:REFERENCES {
                    confidence: url_data.confidence,
                    created_at: url_data.created_at
                }]->(d)
            """, urls_data=urls_data)
        
        # Batch create code elements
        if extraction.get("code_elements"):
            code_data = []
            for code_elem in extraction["code_elements"]:
                code_data.append({
                    "name": self._extract_code_name(code_elem["text"], code_elem["type"]),
                    "type": code_elem["type"],
                    "code_text": code_elem["text"][:1000],  # Limit length
                    "start_pos": code_elem.get("start", 0),
                    "end_pos": code_elem.get("end", 0),
                    "confidence": code_elem.get("confidence", 0.7),
                    "language": self._detect_language(code_elem["text"]),
                    "created_at": metadata["provenance"]["created_at"],
                    "session_id": metadata["session_id"]
                })
            
            tx.run("""
                UNWIND $code_data AS code
                MERGE (c:CodeElement {name: code.name, type: code.type, code_hash: apoc.util.sha1([code.code_text])})
                ON CREATE SET c += {
                    code_text: code.code_text,
                    language: code.language,
                    first_seen: code.created_at,
                    occurrence_count: 1,
                    created_at: code.created_at
                }
                ON MATCH SET c.occurrence_count = coalesce(c.occurrence_count, 0) + 1
                SET c += {
                    confidence: code.confidence,
                    updated_at: datetime()
                }
                WITH c, code
                MATCH (p:Prompt {session_id: code.session_id})
                MERGE (p)-[:CONTAINS_CODE {
                    confidence: code.confidence,
                    start_pos: code.start_pos,
                    end_pos: code.end_pos,
                    created_at: code.created_at
                }]->(c)
            """, code_data=code_data)
        
        # Create Processing Job node for tracking
        tx.run("""
            MERGE (j:ProcessingJob {job_id: $job_id})
            SET j += {
                session_id: $session_id,
                stage: $stage,
                confidence: $confidence,
                processing_tier: $processing_tier,
                entities_extracted: $entities_count,
                urls_found: $urls_count,
                code_elements_found: $code_count,
                timestamp: $timestamp,
                created_at: $created_at,
                service_version: $service_version,
                completed: true
            }
            WITH j
            MATCH (p:Prompt {session_id: $session_id})
            MERGE (p)-[:PROCESSED_BY]->(j)
        """,
            job_id=metadata["provenance"]["processing_job_id"],
            session_id=metadata["session_id"],
            stage=metadata["processing_stage"],
            confidence=extraction.get("confidence", 0.0),
            processing_tier=extraction.get("processing_tier", "tier_0"),
            entities_count=len(extraction.get("entities", [])),
            urls_count=len(extraction.get("urls", [])),
            code_count=len(extraction.get("code_elements", [])),
            timestamp=metadata["timestamp"],
            created_at=metadata["provenance"]["created_at"],
            service_version=metadata["service_version"]
        )
    
    def _create_canonical_id(self, text: str) -> str:
        """Create canonical ID for entity deduplication"""
        # Simple normalization - could be more sophisticated
        normalized = re.sub(r'\s+', '_', text.strip().lower())
        return f"entity_{hashlib.md5(normalized.encode()).hexdigest()[:12]}"
    
    def _extract_code_name(self, code_text: str, code_type: str) -> str:
        """Extract meaningful name from code"""
        if code_type == "python_function":
            match = re.search(r'def\s+(\w+)', code_text)
            return match.group(1) if match else "anonymous_function"
        elif code_type == "python_class":
            match = re.search(r'class\s+(\w+)', code_text)
            return match.group(1) if match else "anonymous_class"
        elif code_type == "assignment":
            match = re.search(r'([a-zA-Z_]\w*)\s*=', code_text)
            return match.group(1) if match else "variable"
        else:
            return f"{code_type}_{hashlib.md5(code_text.encode()).hexdigest()[:8]}"
    
    def _detect_language(self, code_text: str) -> str:
        """Simple language detection"""
        if 'def ' in code_text or 'import ' in code_text or 'class ' in code_text:
            return "python"
        elif 'function ' in code_text or 'const ' in code_text or 'let ' in code_text:
            return "javascript"
        elif '```' in code_text:
            lang_match = re.search(r'```(\w+)', code_text)
            return lang_match.group(1) if lang_match else "unknown"
        else:
            return "unknown"

# Initialize services
extractor = KnowledgeGraphExtractor()
ingestor = Neo4jIngestor(neo4j_driver)

async def enqueue_heavy_processing(session_id: str, prompt_text: str, response_text: str = ""):
    """Enqueue tier 1 and tier 2 processing jobs"""
    job_data = {
        "session_id": session_id,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": "tier_1_structured"
    }
    
    await redis_client.lpush("heavy_processing_queue", json.dumps(job_data))
    logger.info(f"Enqueued heavy processing for session {session_id}")

async def call_ollama_api(endpoint: str, payload: Dict) -> Union[Dict, httpx.Response]:
    """Call the actual Ollama API"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(f"{OLLAMA_URL}{endpoint}", json=payload)
            response.raise_for_status()
            
            if payload.get("stream", False):
                return response
            else:
                return response.json()
                
        except httpx.RequestError as e:
            logger.error(f"Error calling Ollama: {e}")
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e))

@app.post("/api/generate")
async def generate(request: OllamaGenerateRequest, req: Request):
    """Ollama generate endpoint with knowledge graph integration"""
    
    # Extract source information
    source_info = {
        "client_ip": req.client.host,
        "user_agent": req.headers.get("user-agent", ""),
        "endpoint": "/api/generate"
    }
    
    # Create metadata
    metadata = extractor.extract_metadata(request.dict(), source_info)
    
    # Tier 0 light extraction (fast)
    tier_0_extraction = extractor.tier_0_light_extraction(request.prompt, nlp)
    
    # Call Ollama API
    if request.stream:
        # Handle streaming response
        ollama_response = await call_ollama_api("/api/generate", request.dict())
        
        async def stream_with_kg():
            accumulated_response = ""
            async for chunk in ollama_response.aiter_text():
                if chunk.strip():
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data.get("response"):
                            accumulated_response += chunk_data["response"]
                        yield chunk
                        
                        # If this is the final chunk, process KG
                        if chunk_data.get("done", False):
                            response_data = chunk_data
                            response_data["response"] = accumulated_response
                            
                            # Ingest to Neo4j
                            ingestor.ingest_prompt_response(
                                request.dict(), response_data, tier_0_extraction, metadata
                            )
                            
                            # Enqueue heavy processing
                            await enqueue_heavy_processing(
                                metadata["session_id"], request.prompt, accumulated_response
                            )
                            
                    except json.JSONDecodeError:
                        yield chunk
        
        return StreamingResponse(stream_with_kg(), media_type="text/plain")
    
    else:
        # Handle non-streaming response
        response_data = await call_ollama_api("/api/generate", request.dict())
        
        # Ingest to Neo4j (async to not block response)
        asyncio.create_task(
            asyncio.to_thread(
                ingestor.ingest_prompt_response,
                request.dict(), response_data, tier_0_extraction, metadata
            )
        )
        
        # Enqueue heavy processing
        await enqueue_heavy_processing(
            metadata["session_id"], 
            request.prompt, 
            response_data.get("response", "")
        )
        
        return response_data

@app.post("/api/chat")
async def chat(request: OllamaChatRequest, req: Request):
    """Ollama chat endpoint with knowledge graph integration"""
    
    # Extract conversation text for processing
    conversation_text = "\n".join([
        f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
        for msg in request.messages
    ])
    
    # Extract source information
    source_info = {
        "client_ip": req.client.host,
        "user_agent": req.headers.get("user-agent", ""),
        "endpoint": "/api/chat"
    }
    
    # Create metadata
    request_dict = request.dict()
    request_dict["conversation_text"] = conversation_text
    metadata = extractor.extract_metadata(request_dict, source_info)
    
    # Tier 0 light extraction
    tier_0_extraction = extractor.tier_0_light_extraction(conversation_text, nlp)
    
    # Call Ollama API
    if request.stream:
        ollama_response = await call_ollama_api("/api/chat", request.dict())
        
        async def stream_with_kg():
            accumulated_response = ""
            async for chunk in ollama_response.aiter_text():
                if chunk.strip():
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data.get("message", {}).get("content"):
                            accumulated_response += chunk_data["message"]["content"]
                        yield chunk
                        
                        if chunk_data.get("done", False):
                            response_data = chunk_data
                            response_data["response"] = accumulated_response
                            
                            # Ingest to Neo4j
                            ingestor.ingest_prompt_response(
                                request_dict, response_data, tier_0_extraction, metadata
                            )
                            
                            # Enqueue heavy processing
                            await enqueue_heavy_processing(
                                metadata["session_id"], conversation_text, accumulated_response
                            )
                            
                    except json.JSONDecodeError:
                        yield chunk
        
        return StreamingResponse(stream_with_kg(), media_type="text/plain")
    
    else:
        response_data = await call_ollama_api("/api/chat", request.dict())
        
        # Ingest to Neo4j (async to not block response)
        asyncio.create_task(
            asyncio.to_thread(
                ingestor.ingest_prompt_response,
                request_dict, response_data, tier_0_extraction, metadata
            )
        )
        
        # Enqueue heavy processing
        response_text = response_data.get("message", {}).get("content", "")
        await enqueue_heavy_processing(
            metadata["session_id"], conversation_text, response_text
        )
        
        return response_data

# Proxy other Ollama endpoints
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_ollama(request: Request, path: str):
    """Proxy other Ollama API endpoints"""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Forward the request
            response = await client.request(
                method=request.method,
                url=f"{OLLAMA_URL}/api/{path}",
                content=await request.body(),
                headers={key: value for key, value in request.headers.items() 
                        if key.lower() not in ['host', 'content-length']},
                params=request.query_params
            )
            
            # Return the response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Neo4j connection
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        
        # Check Redis connection
        await redis_client.ping()
        
        # Check Ollama connection
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{OLLAMA_URL}/api/tags")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "neo4j": "connected",
                "redis": "connected",
                "ollama": "connected",
                "spacy": "loaded" if nlp else "not_loaded"
            },
            "version": VERSION
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/kg/stats")
async def knowledge_graph_stats():
    """Get knowledge graph statistics"""
    with neo4j_driver.session() as session:
        stats = session.run("""
            CALL {
                MATCH (p:Prompt) RETURN count(p) as prompts
            }
            CALL {
                MATCH (r:Response) RETURN count(r) as responses
            }
            CALL {
                MATCH (e:Entity) RETURN count(e) as entities
            }
            CALL {
                MATCH (d:Document) RETURN count(d) as documents
            }
            CALL {
                MATCH (c:CodeElement) RETURN count(c) as code_elements
            }
            CALL {
                MATCH (j:ProcessingJob) RETURN count(j) as processing_jobs
            }
            RETURN prompts, responses, entities, documents, code_elements, processing_jobs
        """).single()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": dict(stats)
        }

if __name__ == "__main__":
    import uvicorn
    
    # Create indexes on startup
    ingestor.create_indexes()
    logger.info("Created Neo4j indexes")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)