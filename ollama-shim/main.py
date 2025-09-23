#!/usr/bin/env python3
"""
Advanced Ollama API Shim with Full Stack Integration

This shim intercepts all Ollama API calls and augments them with:
- Knowledge Graph storage/retrieval (Neo4j)
- Vector embeddings and semantic search (Weaviate + Supabase pgvector)
- Conversational memory and context management
- Tool integration via MCP
- Comprehensive metrics for Prometheus
- Intelligent routing and caching
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from pydantic import BaseModel, Field
import asyncpg
import weaviate
from neo4j import AsyncGraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# import asyncio
# import uuid
# import time
# import logging
# import json
# from typing import Dict, Any, Optional, AsyncGenerator
from collections import defaultdict

def safe_parse_entities(raw: str):
    # remove ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
    return json.loads(cleaned)

# ============================================================================
# REQUEST QUEUE AND RESOURCE MANAGEMENT
# ============================================================================

# Configure how many Ollama calls can run at once
OLLAMA_CONCURRENCY_LIMIT = 1
# ollama_semaphore = asyncio.Semaphore(OLLAMA_CONCURRENCY_LIMIT)

ollama_lock = asyncio.Lock()


class RequestQueue:
    """Manages request queuing and resource-aware scheduling"""
    
    def __init__(self, max_concurrent: int = 5, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.active_requests = 0
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_stats = {
            'queued': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0,
            'queue_full_rejections': 0
        }
    
    async def enqueue_request(self, request_data: Dict[str, Any], priority: int = 5) -> str:
        """Enqueue a request with priority (1=highest, 10=lowest)"""
        request_id = str(uuid.uuid4())
        
        # Check if queue is full
        if self.queue.qsize() >= self.max_queue_size:
            self.request_stats['queue_full_rejections'] += 1
            raise HTTPException(status_code=503, detail="Request queue is full. Please try again later.")
        
        queue_item = {
            'request_id': request_id,
            'data': request_data,
            'priority': priority,
            'enqueued_at': time.time(),
            'future': asyncio.Future()
        }
        
        await self.queue.put(queue_item)
        self.request_stats['queued'] += 1
        
        # Update queue metrics
        queue_size.set(self.queue.qsize())
        
        return request_id
    
    async def process_queue(self):
        """Background task to process queued requests"""
        while True:
            try:
                # Get next request from queue
                queue_item = await self.queue.get()
                self.request_stats['queued'] -= 1
                
                # Wait for available slot
                await self.semaphore.acquire()
                
                # Process request in background
                asyncio.create_task(self._process_request(queue_item))
                
            except Exception as e:
                logging.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_request(self, queue_item: Dict[str, Any]):
        """Process a single queued request"""
        try:
            self.active_requests += 1
            self.request_stats['processing'] += 1
            active_requests_gauge.set(self.active_requests)
            
            # Record queue wait time
            wait_time = time.time() - queue_item['enqueued_at']
            queue_wait_time.observe(wait_time)
            
            # Process the actual request
            result = await self._execute_request(queue_item['data'])
            
            # Set result on future
            queue_item['future'].set_result(result)
            self.request_stats['completed'] += 1
            
        except Exception as e:
            queue_item['future'].set_exception(e)
            self.request_stats['failed'] += 1
            logging.error(f"Request processing failed: {e}")
            
        finally:
            self.active_requests -= 1
            self.request_stats['processing'] -= 1
            active_requests_gauge.set(self.active_requests)
            self.semaphore.release()
            queue_size.set(self.queue.qsize())
    
    async def _execute_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual request (to be implemented)"""
        # This will be set by the main application
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        return {
            'active_requests': self.active_requests,
            'queue_size': self.queue.qsize(),
            'max_concurrent': self.max_concurrent,
            'max_queue_size': self.max_queue_size,
            **self.request_stats
        }

class ResourceMonitor:
    """Monitors system resources and adjusts queue behavior"""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # CPU percentage
        self.memory_threshold = 85.0  # Memory percentage
        self.gpu_threshold = 90.0  # GPU percentage (if available)
        self.ollama_response_time_threshold = 30.0  # seconds
        self.last_check = 0
        self.check_interval = 5.0  # Check every 5 seconds
        self.resource_stats = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_percent': 0.0,
            'ollama_avg_response_time': 0.0,
            'system_overloaded': False
        }
    
    async def check_resources(self) -> bool:
        """Check if system has resources available"""
        now = time.time()
        if now - self.last_check < self.check_interval:
            return not self.resource_stats['system_overloaded']
        
        self.last_check = now
        
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.resource_stats['cpu_percent'] = cpu_percent
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.resource_stats['memory_percent'] = memory_percent
            
            # Check Ollama response time (from metrics)
            ollama_response_time = self._get_ollama_avg_response_time()
            self.resource_stats['ollama_avg_response_time'] = ollama_response_time
            
            # Determine if system is overloaded
            overloaded = (
                cpu_percent > self.cpu_threshold or
                memory_percent > self.memory_threshold or
                ollama_response_time > self.ollama_response_time_threshold
            )
            
            self.resource_stats['system_overloaded'] = overloaded
            
            # Update Prometheus metrics
            system_cpu_usage.set(cpu_percent)
            system_memory_usage.set(memory_percent)
            system_overloaded.set(1 if overloaded else 0)
            
            return not overloaded
            
        except ImportError:
            logging.warning("psutil not available for resource monitoring")
            return True  # Default to allowing requests
        except Exception as e:
            logging.error(f"Resource check failed: {e}")
            return True  # Default to allowing requests on error
    
    def _get_ollama_avg_response_time(self) -> float:
        """Get average Ollama response time from metrics"""
        # This would integrate with Prometheus to get recent average
        # For now, return 0 as placeholder
        return 0.0
    
    def get_priority_adjustment(self) -> int:
        """Get priority adjustment based on system load"""
        if self.resource_stats['system_overloaded']:
            return 3  # Increase priority (lower numbers) when overloaded
        elif self.resource_stats['cpu_percent'] > 60:
            return 1  # Slight priority increase under moderate load
        return 0  # No adjustment

# ============================================================================
# CONFIGURATION AND MODELS
# ============================================================================

class Config:
    """Configuration settings for the API shim"""
    # Service endpoints
    OLLAMA_BASE_URL = "http://ollama:11434"
    SUPABASE_DB_URL = "postgresql://supabase_admin:your_secure_supabase_password@supabase-db:5432/postgres"
    WEAVIATE_URL = "http://weaviate:8080"
    NEO4J_URI = "bolt://172.17.0.1:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = ""
    MCP_SERVER_URL = "http://mcp-server:8000"
    
    # AI Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_CONTEXT_TOKENS = 8192
    SIMILARITY_THRESHOLD = 0.7
    MAX_MEMORY_ENTRIES = 50
    
    MAX_CONCURRENT_REQUESTS = 1
    MAX_QUEUE_SIZE = 100
    # Feature flags
    ENABLE_KNOWLEDGE_GRAPH = True
    ENABLE_VECTOR_SEARCH = True
    ENABLE_MEMORY = True
    ENABLE_TOOLS = False
    ENABLE_CACHING = True

class ChatMessage(BaseModel):
    """Chat message structure"""
    role: str
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationContext(BaseModel):
    """Conversation context with memory and knowledge"""
    session_id: str
    messages: List[ChatMessage]
    knowledge_entities: List[Dict[str, Any]] = []
    relevant_memories: List[Dict[str, Any]] = []
    available_tools: List[str] = []
    embedding_cache: Dict[str, List[float]] = {}

class OllamaRequest(BaseModel):
    """Ollama API request structure"""
    model: str
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    raw: bool = False

# ============================================================================
# METRICS SETUP
# ============================================================================

# Create custom registry for our metrics
registry = CollectorRegistry()

# Request metrics
request_count = Counter(
    'ollama_shim_requests_total',
    'Total requests processed by the shim',
    ['method', 'endpoint', 'model', 'status'],
    registry=registry
)

request_duration = Histogram(
    'ollama_shim_request_duration_seconds',
    'Request processing time',
    ['method', 'endpoint', 'model'],
    registry=registry
)

# Knowledge graph metrics
kg_operations = Counter(
    'ollama_shim_kg_operations_total',
    'Knowledge graph operations',
    ['operation', 'status'],
    registry=registry
)

# Vector search metrics
vector_searches = Counter(
    'ollama_shim_vector_searches_total',
    'Vector similarity searches performed',
    ['store_type', 'status'],
    registry=registry
)

# Memory operations
memory_operations = Counter(
    'ollama_shim_memory_operations_total',
    'Memory storage/retrieval operations',
    ['operation', 'status'],
    registry=registry
)

# Tool usage metrics
tool_usage = Counter(
    'ollama_shim_tool_usage_total',
    'Tool invocation count',
    ['tool_name', 'status'],
    registry=registry
)

# Request queue metrics
queue_size = Gauge(
    'ollama_shim_queue_size',
    'Current size of request queue',
    registry=registry
)

queue_wait_time = Histogram(
    'ollama_shim_queue_wait_seconds',
    'Time requests spend waiting in queue',
    registry=registry
)

active_requests_gauge = Gauge(
    'ollama_shim_active_requests',
    'Number of currently processing requests',
    registry=registry
)

# Resource monitoring metrics
system_cpu_usage = Gauge(
    'ollama_shim_system_cpu_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_usage = Gauge(
    'ollama_shim_system_memory_percent', 
    'System memory usage percentage',
    registry=registry
)

system_overloaded = Gauge(
    'ollama_shim_system_overloaded',
    'System overload status (1=overloaded, 0=normal)',
    registry=registry
)

# Queue statistics
queued_requests = Counter(
    'ollama_shim_queued_requests_total',
    'Total requests queued',
    ['priority_level'],
    registry=registry
)

queue_rejections = Counter(
    'ollama_shim_queue_rejections_total',
    'Total requests rejected due to full queue',
    registry=registry
)

deferred_requests = Counter(
    'ollama_shim_deferred_requests_total',
    'Total requests deferred due to resource constraints',
    ['reason'],
    registry=registry
)

# ============================================================================
# DATABASE AND SERVICE CONNECTIONS
# ============================================================================

class ServiceConnections:
    """Manages connections to all backend services"""
    
    def __init__(self):
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.weaviate_client: Optional[weaviate.Client] = None
        self.neo4j_driver = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.ollama_client: Optional[httpx.AsyncClient] = None
        self.mcp_client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self):
        """Initialize all service connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                Config.SUPABASE_DB_URL,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            
            # Initialize database schema
            await self.setup_database_schema()
            
            # Weaviate client
            self.weaviate_client = weaviate.Client(
                url=Config.WEAVIATE_URL,
                timeout_config=(5, 15)
            )
            await self.setup_weaviate_schema()
            
            # Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
            )
            await self.setup_neo4j_constraints()
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            
            # HTTP clients
            self.ollama_client = httpx.AsyncClient(base_url=Config.OLLAMA_BASE_URL)
            self.mcp_client = httpx.AsyncClient(base_url=Config.MCP_SERVER_URL)
            
            logging.info("All service connections initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize service connections: {e}")
            raise
    
    async def setup_database_schema(self):
        """Setup PostgreSQL schema for conversation memory and caching"""
        async with self.pg_pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Conversations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)
            
            # Messages table with vector embeddings
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),  -- MiniLM-L6-v2 dimension
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)
            
            # Knowledge entities extracted from conversations
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    entity_type VARCHAR(100) NOT NULL,
                    entity_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    embedding vector(384),
                    confidence FLOAT DEFAULT 0.0,
                    source_message_id UUID REFERENCES messages(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)
            
            # Embedding cache table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content_hash VARCHAR(64) UNIQUE NOT NULL,
                    embedding vector(384) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_embedding 
                ON messages USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_entities_embedding 
                ON knowledge_entities USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
                ON messages(conversation_id);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp);
            """)
    
    async def setup_weaviate_schema(self):
        """Setup Weaviate schema for advanced vector operations"""
        try:
            # Check if class already exists
            if not self.weaviate_client.schema.exists("ConversationMemory"):
                conversation_memory_schema = {
                    "class": "ConversationMemory",
                    "description": "Conversation memories with context",
                    "vectorizer": "text2vec-ollama",
                    "moduleConfig": {
                        "text2vec-ollama": {
                            "apiEndpoint": "http://ollama:11434",
                            "model": "nomic-embed-text"
                        }
                    },
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Message content"
                        },
                        {
                            "name": "role",
                            "dataType": ["string"],
                            "description": "Message role (user/assistant/system)"
                        },
                        {
                            "name": "sessionId",
                            "dataType": ["string"],
                            "description": "Conversation session ID"
                        },
                        {
                            "name": "timestamp",
                            "dataType": ["date"],
                            "description": "Message timestamp"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Additional metadata",
                            "properties": [
                                {
                                "name": "source",
                                "dataType": ["string"],
                                "description": "Source of the message"
                                },
                                {
                                "name": "tags",
                                "dataType": ["string[]"],
                                "description": "Tags for the message"
                                }
                            ]
                            }
                    ]
                }
                self.weaviate_client.schema.create_class(conversation_memory_schema)
            
            if not self.weaviate_client.schema.exists("KnowledgeEntity"):
                knowledge_entity_schema = {
                    "class": "KnowledgeEntity",
                    "description": "Extracted knowledge entities",
                    "vectorizer": "text2vec-ollama",
                    "moduleConfig": {
                        "text2vec-ollama": {
                            "apiEndpoint": "http://ollama:11434",
                            "model": "nomic-embed-text"
                        }
                    },
                    "properties": [
                        {
                            "name": "entityType",
                            "dataType": ["string"],
                            "description": "Type of entity (person, place, concept, etc.)"
                        },
                        {
                            "name": "entityName",
                            "dataType": ["string"],
                            "description": "Name of the entity"
                        },
                        {
                            "name": "description",
                            "dataType": ["text"],
                            "description": "Entity description"
                        },
                        {
                            "name": "confidence",
                            "dataType": ["number"],
                            "description": "Extraction confidence score"
                        },
                        {
                            "name": "sourceSessionId",
                            "dataType": ["string"],
                            "description": "Source conversation session"
                        }
                    ]
                }
                self.weaviate_client.schema.create_class(knowledge_entity_schema)
                
        except Exception as e:
            logging.warning(f"Weaviate schema setup warning: {e}")
    
    async def setup_neo4j_constraints(self):
        """Setup Neo4j constraints and indexes"""
        try:
            async with self.neo4j_driver.session() as session:
                # Create constraints
                await session.run("""
                    CREATE CONSTRAINT conversation_session_id IF NOT EXISTS
                    FOR (c:Conversation) REQUIRE c.session_id IS UNIQUE
                """)
                
                await session.run("""
                    CREATE CONSTRAINT entity_name_type IF NOT EXISTS
                    FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE
                """)
                
                # Create indexes
                await session.run("""
                    CREATE INDEX message_timestamp IF NOT EXISTS
                    FOR (m:Message) ON (m.timestamp)
                """)
                
                await session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """)
                
        except Exception as e:
            logging.warning(f"Neo4j setup warning: {e}")
    
    async def cleanup(self):
        """Cleanup all connections"""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            if self.ollama_client:
                await self.ollama_client.aclose()
            if self.mcp_client:
                await self.mcp_client.aclose()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
                
    async def run_ollama(self, payload: dict):
        async with ollama_lock:
            return await self.ollama_client.post(
                    "/api/generate",
                    json=payload,
                    timeout=600.0  # 10 minutes timeout
                )
            # return {"ollama_result": f"processed {payload}"}

# Global service connections instance
services = ServiceConnections()

# ============================================================================
# CORE PROCESSING CLASSES
# ============================================================================

import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from prometheus_client import Counter, Gauge

# Prometheus metrics - define these at module level
# memory_operations = Counter('memory_operations_total', 'Memory operations', ['operation', 'status'])
# vector_searches = Counter('vector_searches_total', 'Vector searches', ['store_type', 'status'])
# cached_embeddings = Gauge('cached_embeddings_total', 'Total cached embeddings')

class MemoryManager:
    """Manages conversational memory and context"""
    
    def __init__(self, services: ServiceConnections):
        self.services = services
        self.session_cache: Dict[str, ConversationContext] = {}
        self.active_sessions = 0
    
    async def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Get existing session or create new one"""
        if session_id in self.session_cache:
            return self.session_cache[session_id]
        
        # Try to load from database
        async with self.services.pg_pool.acquire() as conn:
            # Get or create conversation
            conversation = await conn.fetchrow("""
                INSERT INTO conversations (session_id) 
                VALUES ($1) 
                ON CONFLICT (session_id) DO UPDATE SET updated_at = NOW()
                RETURNING id, session_id, created_at, metadata
            """, session_id)
            
            # Load recent messages
            messages = await conn.fetch("""
                SELECT role, content, timestamp, metadata 
                FROM messages 
                WHERE conversation_id = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            """, conversation['id'], Config.MAX_MEMORY_ENTRIES)
            
            # Create context with proper metadata parsing
            chat_messages = []
            for msg in reversed(messages):  # Reverse to get chronological order
                # Parse metadata properly - handle all possible cases
                metadata = msg['metadata']
                
                # Handle different metadata formats from PostgreSQL
                if metadata is None:
                    metadata = {}
                elif isinstance(metadata, str):
                    # Handle string representations
                    if metadata.strip() in ['', '{}']:
                        metadata = {}
                    else:
                        try:
                            metadata = json.loads(metadata)
                            # Ensure it's a dict after parsing
                            if not isinstance(metadata, dict):
                                metadata = {}
                        except (json.JSONDecodeError, TypeError):
                            logging.warning(f"Failed to parse metadata: {metadata}")
                            metadata = {}
                elif not isinstance(metadata, dict):
                    # Handle any other types that might come from the database
                    try:
                        # Try to convert to dict if it's a dict-like object
                        metadata = dict(metadata)
                    except (TypeError, ValueError):
                        metadata = {}
                
                # Ensure metadata is always a dictionary
                if not isinstance(metadata, dict):
                    metadata = {}
                
                chat_messages.append(ChatMessage(
                    role=msg['role'],
                    content=msg['content'],
                    timestamp=msg['timestamp'],
                    metadata=metadata
                ))
            
            context = ConversationContext(
                session_id=session_id,
                messages=chat_messages
            )
            
            self.session_cache[session_id] = context
            # Update active sessions gauge if it exists
            try:
                active_sessions.set(len(self.session_cache))
            except NameError:
                pass  # Metric not defined yet
            
            return context
    
    async def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation memory"""
        if Config.ENABLE_MEMORY is False:
            return
        
        context = await self.get_or_create_session(session_id)
        print(metadata)
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {"None":"none"}
        )
        
        context.messages.append(message)
        
        # Keep only recent messages in memory
        if len(context.messages) > Config.MAX_MEMORY_ENTRIES:
            context.messages = context.messages[-Config.MAX_MEMORY_ENTRIES:]
        
        # Store in database
        try:
            async with self.services.pg_pool.acquire() as conn:
                # Get conversation ID
                conversation = await conn.fetchrow(
                    "SELECT id FROM conversations WHERE session_id = $1",
                    session_id
                )
                
                # Generate embedding
                embedding_vector = await self.get_embedding(content)
                
                # Convert embedding to proper format for database storage
                if self._is_using_pgvector():
                    # For pgvector extension - convert list to vector string format
                    embedding_db_format = f"[{','.join(map(str, embedding_vector))}]"
                else:
                    # For JSON storage - convert to JSON string
                    embedding_db_format = json.dumps(embedding_vector)
                
                # Insert message
                await conn.execute("""
                    INSERT INTO messages (conversation_id, role, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """, conversation['id'], role, content, embedding_db_format, json.dumps(metadata or {}))
                
                memory_operations.labels(operation='store', status='success').inc()
                
        except Exception as e:
            logging.error(f"Failed to store message: {e}")
            memory_operations.labels(operation='store', status='error').inc()
    
    def _is_using_pgvector(self) -> bool:
        """Check if using pgvector extension based on config or database schema"""
        # You can implement this based on your configuration
        # For now, assume pgvector if using vector similarity operators
        return hasattr(Config, 'USE_PGVECTOR') and Config.USE_PGVECTOR
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching"""
        import hashlib
        
        # Create content hash
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Try cache first
        try:
            async with self.services.pg_pool.acquire() as conn:
                cached = await conn.fetchrow("""
                    SELECT embedding FROM embedding_cache 
                    WHERE content_hash = $1
                """, content_hash)
                
                if cached:
                    # Update access info
                    await conn.execute("""
                        UPDATE embedding_cache 
                        SET access_count = access_count + 1, last_accessed = NOW()
                        WHERE content_hash = $1
                    """, content_hash)
                    
                    # Parse embedding based on storage format
                    if self._is_using_pgvector():
                        # Parse pgvector format: "[1.0,2.0,3.0]" -> [1.0, 2.0, 3.0]
                        embedding_str = cached['embedding']
                        if isinstance(embedding_str, str):
                            # Remove brackets and split
                            embedding_str = embedding_str.strip('[]')
                            return [float(x.strip()) for x in embedding_str.split(',')]
                        else:
                            # Already parsed by database driver
                            return cached['embedding']
                    else:
                        # Parse JSON format
                        if isinstance(cached['embedding'], str):
                            return json.loads(cached['embedding'])
                        else:
                            return cached['embedding']
                            
        except Exception as e:
            logging.warning(f"Cache lookup failed: {e}")
        
        # Generate new embedding
        try:
            embedding = self.services.embedding_model.encode(text).tolist()
            
            # Cache it
            async with self.services.pg_pool.acquire() as conn:
                # Convert to appropriate storage format
                if self._is_using_pgvector():
                    embedding_db_format = f"[{','.join(map(str, embedding))}]"
                else:
                    embedding_db_format = json.dumps(embedding)
                
                await conn.execute("""
                    INSERT INTO embedding_cache (content_hash, embedding)
                    VALUES ($1, $2)
                    ON CONFLICT (content_hash) DO UPDATE SET 
                        access_count = embedding_cache.access_count + 1,
                        last_accessed = NOW()
                """, content_hash, embedding_db_format)
                
                # Update cache gauge
                count = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
                # cached_embeddings.set(count)
            
            return embedding
            
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return [0.0] * 384  # Return zero embedding as fallback
    
    async def search_similar_memories(self, query: str, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories using vector similarity"""
        try:
            query_embedding = await self.get_embedding(query)
            
            async with self.services.pg_pool.acquire() as conn:
                # Get conversation ID
                conversation = await conn.fetchrow(
                    "SELECT id FROM conversations WHERE session_id = $1",
                    session_id
                )
                
                if not conversation:
                    return []
                
                # Prepare query embedding for database
                if self._is_using_pgvector():
                    # For pgvector extension
                    query_embedding_db = f"[{','.join(map(str, query_embedding))}]"
                    
                    # Vector similarity search using pgvector
                    results = await conn.fetch("""
                        SELECT 
                            role, content, timestamp, metadata,
                            1 - (embedding <=> $1::vector) as similarity
                        FROM messages 
                        WHERE conversation_id = $2 
                            AND 1 - (embedding <=> $1::vector) > $3
                        ORDER BY similarity DESC
                        LIMIT $4
                    """, query_embedding_db, conversation['id'], Config.SIMILARITY_THRESHOLD, limit)
                    
                else:
                    # For JSON storage - use custom similarity function
                    # You'll need to create a custom function or use a different approach
                    # This is a simplified cosine similarity using JSON functions
                    results = await conn.fetch("""
                        SELECT 
                            role, content, timestamp, metadata,
                            -- Custom similarity calculation would go here
                            0.5 as similarity  -- Placeholder
                        FROM messages 
                        WHERE conversation_id = $1 
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """, conversation['id'], limit)
                
                vector_searches.labels(store_type='postgresql', status='success').inc()
                
                return [
                    {
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'].isoformat(),
                        'similarity': float(row['similarity']),
                        'metadata': row['metadata']
                    }
                    for row in results
                ]
                
        except Exception as e:
            logging.error(f"Memory search failed: {e}")
            vector_searches.labels(store_type='postgresql', status='error').inc()
            return []
    
    async def _cosine_similarity_json(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        import math
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(a * a for a in embedding2))
        
        # Cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
class KnowledgeGraphManager:
    """Manages knowledge graph operations with Neo4j"""
    
    def __init__(self, services: ServiceConnections):
        self.services = services
    
    async def extract_and_store_entities(self, content: str, session_id: str, message_role: str):
        """Extract entities from content and store in knowledge graph"""
        try:
            # Use Ollama to extract entities (this would be a specialized prompt)
            entities = await self.extract_entities_with_llm(content)
            print(entities)
            if not entities:
                return []
            
            # Store in Neo4j
            async with self.services.neo4j_driver.session() as session:
                for entity in entities:
                    await session.run("""
                        MERGE (c:Conversation {session_id: $session_id})
                        MERGE (e:Entity {name: $name, type: $type})
                        ON CREATE SET e.created_at = datetime(), e.description = $description
                        ON MATCH SET e.last_seen = datetime()
                        MERGE (c)-[r:MENTIONED]->(e)
                        ON CREATE SET r.first_mentioned = datetime(), r.count = 1
                        ON MATCH SET r.last_mentioned = datetime(), r.count = r.count + 1
                    """, 
                    session_id=session_id,
                    name=entity['name'],
                    type=entity['type'],
                    description=entity.get('description', '')
                    )
                
                kg_operations.labels(operation='store_entities', status='success').inc()
            
            # Also store in Weaviate for vector search
            if Config.ENABLE_VECTOR_SEARCH:
                await self.store_entities_in_weaviate(entities, session_id)
            
            return entities
            
        except Exception as e:
            logging.error(f"Entity extraction failed: {e}")
            kg_operations.labels(operation='store_entities', status='error').inc()
            return []
    
    async def extract_entities_with_llm(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities using Ollama"""
        try:
            # Specialized prompt for entity extraction
            system_prompt = """You are an expert entity extractor. Extract important entities from the given text and return them as JSON array.
            
            For each entity, provide:
            - name: The entity name
            - type: The entity type (person, place, organization, concept, event, etc.)
            - description: Brief description of the entity in context
            
            Only extract entities that are genuinely important to understanding or remembering the conversation.
            
            Return valid JSON array format: [{"name": "...", "type": "...", "description": "..."}]
            """
            response = await self.services.run_ollama(
                {
                    "model": "gemma3:12b",  # You can configure this
                    "prompt": f"System: {system_prompt}\n\nText: {content}\n\nEntities:",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent extraction
                        "num_predict": 500
                    }
                }
            )
            # response = await self.services.ollama_client.post(
            #     "/api/generate",
            #     json={
            #         "model": "gemma3:12b",  # You can configure this
            #         "prompt": f"System: {system_prompt}\n\nText: {content}\n\nEntities:",
            #         "stream": False,
            #         "options": {
            #             "temperature": 0.1,  # Low temperature for consistent extraction
            #             "num_predict": 500
            #         }
            #     }
            # )
            
            print("{response.status_code}\n"+str(response.json()))
            if response.status_code == 200:
                result = response.json()
                entities_text = result.get('response', '').strip()
                print(entities_text)
                try:
                    # entities = json.loads(entities_text)
                    entities = safe_parse_entities(entities_text)
                    return entities if isinstance(entities, list) else []
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse entities JSON: {entities_text}")
                    return []
            
        except Exception as e:
            logging.error(f"LLM entity extraction failed: {e}")
        
        return []
    
    async def store_entities_in_weaviate(self, entities: List[Dict[str, Any]], session_id: str):
        """Store entities in Weaviate for vector search"""
        try:
            for entity in entities:
                self.services.weaviate_client.data_object.create(
                    data_object={
                        "entityType": entity['type'],
                        "entityName": entity['name'],
                        "description": entity.get('description', ''),
                        "confidence": entity.get('confidence', 0.8),
                        "sourceSessionId": session_id
                    },
                    class_name="KnowledgeEntity"
                )
        except Exception as e:
            logging.error(f"Failed to store entities in Weaviate: {e}")
    
    async def get_related_entities(self, query: str, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get entities related to the query"""
        try:
            # Search in Weaviate first (vector similarity)
            weaviate_results = []
            if Config.ENABLE_VECTOR_SEARCH and self.services.weaviate_client:
                try:
                    result = self.services.weaviate_client.query \
                        .get("KnowledgeEntity", ["entityType", "entityName", "description", "confidence"]) \
                        .with_near_text({"concepts": [query]}) \
                        .with_where({
                            "path": ["sourceSessionId"],
                            "operator": "Equal",
                            "valueText": session_id
                        }) \
                        .with_limit(limit) \
                        .do()
                    
                    if "data" in result and "Get" in result["data"]:
                        weaviate_results = result["data"]["Get"]["KnowledgeEntity"]
                        
                except Exception as e:
                    logging.warning(f"Weaviate search failed: {e}")
            
            # Search in Neo4j (graph relationships)
            neo4j_results = []
            try:
                cypher = """
                    MATCH (c:Conversation {session_id: $session_id})-[:MENTIONED]->(e:Entity)
                    WHERE e.name CONTAINS $query OR e.description CONTAINS $query
                    RETURN e.name as name, e.type as type, e.description as description
                    ORDER BY e.name
                    LIMIT $limit
                """

                params = {"session_id": session_id, "query": query, "limit": limit}

                async with self.services.neo4j_driver.session() as session:
                    result = await session.run(cypher, params)
                    neo4j_results = [
                        {
                            "name": record["name"],
                            "type": record["type"],
                            "description": record["description"]
                        }
                        async for record in result
                    ]
                                    
            except Exception as e:
                logging.warning(f"Neo4j search failed: {e}")
            
            # Combine and deduplicate results
            all_results = weaviate_results + neo4j_results
            seen = set()
            unique_results = []
            
            for result in all_results:
                key = (result.get("entityName") or result.get("name"), result.get("entityType") or result.get("type"))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            return unique_results[:limit]
            
        except Exception as e:
            logging.error(f"Failed to get related entities: {e}")
            return []

class ToolManager:
    """Manages tool integration via MCP protocol"""
    
    def __init__(self, services: ServiceConnections):
        self.services = services
        self.available_tools: List[str] = []
    
    async def initialize(self):
        """Initialize available tools from MCP server"""
        try:
            response = await self.services.mcp_client.get("/tools")
            if response.status_code == 200:
                tools_data = response.json()
                self.available_tools = [tool['name'] for tool in tools_data.get('tools', [])]
                logging.info(f"Initialized {len(self.available_tools)} tools: {self.available_tools}")
        except Exception as e:
            logging.warning(f"Failed to initialize tools: {e}")
    
    async def detect_tool_usage(self, content: str) -> List[str]:
        """Detect if content requires tool usage"""
        # This would be more sophisticated in practice
        tool_indicators = {
            'search': ['search for', 'find', 'look up'],
            'calculate': ['calculate', 'compute', 'math'],
            'weather': ['weather', 'temperature', 'forecast'],
            'time': ['what time', 'current time', 'date'],
            'code': ['write code', 'programming', 'script']
        }
        
        detected_tools = []
        content_lower = content.lower()
        
        for tool, indicators in tool_indicators.items():
            if tool in self.available_tools and any(indicator in content_lower for indicator in indicators):
                detected_tools.append(tool)
        
        return detected_tools
    
    async def invoke_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool via MCP"""
        try:
            response = await self.services.mcp_client.post(
                f"/tools/{tool_name}/invoke",
                json=parameters
            )
            
            if response.status_code == 200:
                tool_usage.labels(tool_name=tool_name, status='success').inc()
                return response.json()
            else:
                tool_usage.labels(tool_name=tool_name, status='error').inc()
                return {"error": f"Tool invocation failed: {response.status_code}"}
                
        except Exception as e:
            logging.error(f"Tool invocation failed: {e}")
            tool_usage.labels(tool_name=tool_name, status='error').inc()
            return {"error": str(e)}


class OllamaAPIShim:
    """Main API shim that orchestrates all integrations with request queuing"""
    
    def __init__(self):
        self.memory_manager = MemoryManager(services)
        self.kg_manager = KnowledgeGraphManager(services)
        self.tool_manager = ToolManager(services)
        self.request_queue = RequestQueue(
            max_concurrent=Config.MAX_CONCURRENT_REQUESTS,
            max_queue_size=Config.MAX_QUEUE_SIZE
        )
        self.resource_monitor = ResourceMonitor()
        
        # Track pending requests with their futures
        self.pending_requests = {}
        
        # Set the request execution function
        self.request_queue._execute_request = self._execute_queued_request
        
    async def initialize(self):
        """Initialize all managers and start background tasks"""
        await self.tool_manager.initialize()
        
        # Start queue processor
        asyncio.create_task(self.request_queue.process_queue())
        
        # Start resource monitoring
        asyncio.create_task(self._monitor_resources())
    
    async def _monitor_resources(self):
        """Background task to monitor system resources"""
        while True:
            try:
                await self.resource_monitor.check_resources()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def process_request(self, request: OllamaRequest, session_id: str = None, priority: int = None) -> Dict[str, Any]:
        """Process incoming request with queuing and resource awareness"""
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Determine priority if not specified
        if priority is None:
            priority = self._calculate_priority(request, session_id)
        
        # Check if system is overloaded
        resources_available = await self.resource_monitor.check_resources()
        
        if not resources_available:
            # System overloaded - defer request
            deferred_requests.labels(reason='system_overload').inc()
            logging.info(f"Deferring request due to system overload. Queue size: {self.request_queue.queue.qsize()}")
        
        # Create a future to track this request's completion
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        # Package request data
        request_data = {
            'request_id': request_id,
            'request': request,
            'session_id': session_id,
            'timestamp': time.time(),
            'future': future
        }
        
        # Store the future for later retrieval
        self.pending_requests[request_id] = future
        
        try:
            # Enqueue request
            await self.request_queue.enqueue_request(request_data, priority)
            queued_requests.labels(priority_level=f"p{priority}").inc()
            
            # Wait for the request to be processed and return the result
            try:
                result = await future
                return result
            finally:
                # Clean up the pending request
                self.pending_requests.pop(request_id, None)
                
        except asyncio.QueueFull:
            # Clean up on queue full
            self.pending_requests.pop(request_id, None)
            queue_rejections.inc()
            raise HTTPException(status_code=503, detail="Server too busy. Please try again later.")
        except Exception as e:
            # Clean up on any other error
            self.pending_requests.pop(request_id, None)
            logging.error(f"Request processing failed: {e}")
            raise
    
    def _calculate_priority(self, request: OllamaRequest, session_id: str) -> int:
        """Calculate request priority based on various factors"""
        priority = 5  # Default priority
        
        # Adjust based on system load
        priority += self.resource_monitor.get_priority_adjustment()
        
        # Prioritize shorter prompts
        user_message = self.extract_user_message(request)
        if user_message and len(user_message) < 100:
            priority -= 1  # Higher priority for short messages
        elif user_message and len(user_message) > 1000:
            priority += 1  # Lower priority for long messages
        
        # Prioritize streaming requests (user is waiting)
        if request.stream:
            priority -= 1
        
        # Prioritize returning users (basic session-based logic)
        if session_id in self.memory_manager.session_cache:
            priority -= 1
        
        # Ensure priority stays within bounds
        return max(1, min(10, priority))
    
    async def _execute_queued_request(self, request_data: Dict[str, Any]) -> None:
        """Execute a queued request (called by the queue processor)"""
        request = request_data['request']
        session_id = request_data['session_id']
        request_id = request_data['request_id']
        future = request_data['future']
        
        start_time = time.time()
        
        try:
            # Extract user message
            user_message = self.extract_user_message(request)
            if not user_message:
                result = await self.forward_to_ollama(request)
                future.set_result(result)
                return
            
            # Get conversation context
            context = await self.memory_manager.get_or_create_session(session_id)
            
            # Add user message to memory
            await self.memory_manager.add_message(session_id, "user", user_message)
            
            # Search for relevant memories
            if Config.ENABLE_MEMORY:
                relevant_memories = await self.memory_manager.search_similar_memories(
                    user_message, session_id, limit=5
                )
                context.relevant_memories = relevant_memories
            
            # Extract and store knowledge entities
            if Config.ENABLE_KNOWLEDGE_GRAPH:
                entities = await self.kg_manager.extract_and_store_entities(
                    user_message, session_id, "user"
                )
                context.knowledge_entities = entities
                
                # Get related entities for context
                related_entities = await self.kg_manager.get_related_entities(
                    user_message, session_id
                )
                context.knowledge_entities.extend(related_entities)
            
            # Detect tool usage
            if Config.ENABLE_TOOLS:
                detected_tools = await self.tool_manager.detect_tool_usage(user_message)
                context.available_tools = detected_tools
            
            # Augment request with context
            augmented_request = await self.augment_request(request, context)
            
            # Forward to Ollama and wait for response
            response = await self.forward_to_ollama(augmented_request)
            
            # Process response
            if response and 'response' in response:
                assistant_message = response['response']
                
                # Store assistant response
                await self.memory_manager.add_message(session_id, "assistant", assistant_message)
                
                # Extract entities from response
                if Config.ENABLE_KNOWLEDGE_GRAPH:
                    await self.kg_manager.extract_and_store_entities(
                        assistant_message, session_id, "assistant"
                    )
            
            # Record metrics
            duration = time.time() - start_time
            request_duration.labels(
                method="POST",
                endpoint="/api/generate",
                model=request.model
            ).observe(duration)
            
            request_count.labels(
                method="POST",
                endpoint="/api/generate", 
                model=request.model,
                status="success"
            ).inc()
            
            # Set the result on the future
            future.set_result(response)
            
        except Exception as e:
            logging.error(f"Request processing failed: {e}")
            request_count.labels(
                method="POST",
                endpoint="/api/generate",
                model=request.model,
                status="error"
            ).inc()
            
            # Set the exception on the future
            future.set_exception(HTTPException(status_code=500, detail=str(e)))

    async def process_streaming_request(self, request: OllamaRequest, session_id: str = None) -> AsyncGenerator:
        """Process streaming request with immediate execution (bypass queue for user experience)"""
        
        # For streaming requests, we want immediate response to avoid user waiting
        # But we still do the processing asynchronously
        
        if not session_id:
            session_id = str(uuid.uuid4())
            
        # Check resources but don't defer streaming requests
        resources_available = await self.resource_monitor.check_resources()
        if not resources_available:
            logging.warning(f"Processing streaming request despite system overload for session {session_id}")
        
        # Process context in background while starting stream
        context_task = asyncio.create_task(self._prepare_streaming_context(request, session_id))
        
        try:
            # Start streaming immediately with basic request
            async for chunk in self._stream_with_context(request, session_id, context_task):
                yield chunk
                
        except Exception as e:
            logging.error(f"Streaming request failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    async def _prepare_streaming_context(self, request: OllamaRequest, session_id: str) -> ConversationContext:
        """Prepare context for streaming request"""
        user_message = self.extract_user_message(request)
        if not user_message:
            return None
            
        # Get conversation context
        context = await self.memory_manager.get_or_create_session(session_id)
        await self.memory_manager.add_message(session_id, "user", user_message)
        
        # Add context preparation
        if Config.ENABLE_MEMORY:
            context.relevant_memories = await self.memory_manager.search_similar_memories(
                user_message, session_id, limit=3
            )
        
        if Config.ENABLE_KNOWLEDGE_GRAPH:
            entities = await self.kg_manager.extract_and_store_entities(
                user_message, session_id, "user"
            )
            context.knowledge_entities = entities
        
        return context
    
    async def _stream_with_context(self, request: OllamaRequest, session_id: str, context_task):
        """Stream response while incorporating context"""
        
        # Wait briefly for context or timeout
        try:
            context = await asyncio.wait_for(context_task, timeout=2.0)
            if context:
                request = await self.augment_request(request, context)
        except asyncio.TimeoutError:
            logging.warning("Context preparation timed out, streaming with basic request")
        
        # Stream from Ollama
        full_response = ""
        
        # TODO: convret to run_ollama when streaming is supported 
        async with services.ollama_client.stream(
            "POST",
            "/api/generate", 
            json=request.model_dump(),
        ) as response:
            
            async for chunk in response.aiter_lines():
                if chunk:
                    try:
                        data = json.loads(chunk)
                        if "response" in data:
                            full_response += data["response"]
                        yield f"data: {chunk}\n\n"
                        
                        # Check if done
                        if data.get("done", False):
                            # Store complete response
                            if full_response:
                                await self.memory_manager.add_message(
                                    session_id, "assistant", full_response
                                )
                            break
                            
                    except json.JSONDecodeError:
                        continue
    
    def extract_user_message(self, request: OllamaRequest) -> Optional[str]:
        """Extract user message from request"""
        if request.prompt:
            return request.prompt
        elif request.messages:
            # Find the last user message
            for msg in reversed(request.messages):
                if msg.get('role') == 'user':
                    return msg.get('content')
        return None
    
    async def augment_request(self, request: OllamaRequest, context: ConversationContext) -> OllamaRequest:
        """Augment request with context and knowledge"""
        # Build enhanced system message
        system_parts = []
        
        # Base system message
        if request.system:
            system_parts.append(request.system)
        
        # Add memory context
        if context.relevant_memories:
            system_parts.append("## Relevant conversation history:")
            for memory in context.relevant_memories[:3]:  # Limit to avoid token overflow
                system_parts.append(f"- {memory['content'][:200]}...")
        
        # Add knowledge entities
        if context.knowledge_entities:
            system_parts.append("## Relevant knowledge entities:")
            for entity in context.knowledge_entities[:5]:
                name = entity.get('entityName') or entity.get('name', 'Unknown')
                entity_type = entity.get('entityType') or entity.get('type', 'Unknown')
                description = entity.get('description', '')
                system_parts.append(f"- {name} ({entity_type}): {description}")
        
        # Add available tools
        if context.available_tools:
            system_parts.append(f"## Available tools: {', '.join(context.available_tools)}")
            system_parts.append("You can mention if any of these tools would be helpful for the user's request.")
        
        # Create augmented request
        augmented = OllamaRequest(**request.model_dump())
        if system_parts:
            augmented.system = "\n\n".join(system_parts)
        
        return augmented
    
    async def forward_to_ollama(self, request: OllamaRequest) -> Dict[str, Any]:
        """Forward request to actual Ollama instance"""
        try:
            response = await services.run_ollama(
                request.model_dump()
            )
            # response = await services.ollama_client.post(
            #     "/api/generate",
            #     json=request.model_dump(),
            #     timeout=600.0  # 10 minutes timeout
            # )
            print("{response.status_code}\n"+str(response.json()))
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama error: {response.text}"
                )
                
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Ollama request timeout")
        except Exception as e:
            logging.error(f"Ollama forwarding failed: {e}")
            raise HTTPException(status_code=502, detail=f"Ollama unavailable: {str(e)}")


    def _get_resource_recommendations(self) -> List[str]:
        """Get recommendations based on current resource usage"""
        recommendations = []
        stats = self.resource_monitor.resource_stats
        
        if stats['cpu_percent'] > 80:
            recommendations.append("High CPU usage detected. Consider reducing concurrent requests.")
        
        if stats['memory_percent'] > 85:
            recommendations.append("High memory usage detected. Consider clearing caches or restarting services.")
        
        if stats['ollama_avg_response_time'] > 20:
            recommendations.append("Ollama response times are slow. Check Ollama service health.")
        
        if stats['system_overloaded']:
            recommendations.append("System is overloaded. Requests are being queued automatically.")
        
        if not recommendations:
            recommendations.append("System resources are healthy.")
        
        return recommendations
    
# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await services.initialize()
    shim = OllamaAPIShim()
    await shim.initialize()
    app.state.shim = shim
    
    logging.info("Ollama API Shim started successfully")
    
    yield
    
    # Shutdown
    await services.cleanup()
    logging.info("Ollama API Shim stopped")

app = FastAPI(
    title="Advanced Ollama API Shim",
    description="Ollama API with integrated knowledge graph, vector search, memory, and tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/generate")
async def generate(request: OllamaRequest, req: Request):
    """Main generation endpoint with queuing and resource awareness"""
    # Extract session ID and priority from headers
    session_id = req.headers.get("X-Session-ID") or str(uuid.uuid4())
    priority = req.headers.get("X-Priority")
    
    body_bytes = await req.body()   # <- await here
    body_str = body_bytes.decode("utf-8")
    print("Request body:", body_str)    # print to console

    # Convert priority header to int if provided
    if priority:
        try:
            priority = max(1, min(10, int(priority)))
        except ValueError:
            priority = None
    
    shim: OllamaAPIShim = app.state.shim
    
    if request.stream:
        return StreamingResponse(
            shim.process_streaming_request(request, session_id),
            media_type="text/plain"
        )
    else:
        return await shim.process_request(request, session_id, priority)

@app.post("/api/chat")
async def chat(request: OllamaRequest, req: Request):
    """Chat endpoint (Ollama-compatible) with queuing and resource awareness"""
    # Extract session ID and priority from headers
    session_id = req.headers.get("X-Session-ID") or str(uuid.uuid4())
    priority = req.headers.get("X-Priority")
    
    body_bytes = await req.body()   # <- await here
    body_str = body_bytes.decode("utf-8")
    print("Request body:", body_str)    # print to console
    
    # Convert priority header to int if provided
    if priority:
        try:
            priority = max(1, min(10, int(priority)))
        except ValueError:
            priority = None

    shim: OllamaAPIShim = app.state.shim

    # Handle streaming vs non-streaming requests
    if request.stream:
        return StreamingResponse(
            shim.process_streaming_request(request, session_id),
            media_type="text/plain"
        )
    else:
        response = await shim.process_request(request, session_id, priority)
        print(response.json())
        return response
    
@app.get("/api/shim/queue/status")
async def get_queue_status():
    """Get current queue status and statistics"""
    try:
        shim: OllamaAPIShim = app.state.shim
        queue_stats = shim.request_queue.get_stats()
        resource_stats = shim.resource_monitor.resource_stats
        
        return {
            "queue": queue_stats,
            "resources": resource_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shim/queue/priority")
async def update_request_priority(request_id: str, new_priority: int):
    """Update priority of a queued request"""
    if not 1 <= new_priority <= 10:
        raise HTTPException(status_code=400, detail="Priority must be between 1 and 10")
    
    try:
        # This would require implementing request lookup and priority update
        # For now, return a placeholder response
        return {
            "message": f"Priority update requested for {request_id}",
            "new_priority": new_priority,
            "status": "queued"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/shim/queue/cancel/{request_id}")
async def cancel_queued_request(request_id: str):
    """Cancel a queued request"""
    try:
        # This would require implementing request cancellation
        # For now, return a placeholder response
        return {
            "message": f"Cancellation requested for {request_id}",
            "status": "cancelled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shim/queue/flush")
async def flush_queue():
    """Flush all queued requests (admin operation)"""
    try:
        shim: OllamaAPIShim = app.state.shim
        
        # Get current queue size before flush
        queue_size_before = shim.request_queue.queue.qsize()
        
        # Clear the queue (this is a simplified implementation)
        while not shim.request_queue.queue.empty():
            try:
                item = shim.request_queue.queue.get_nowait()
                # Cancel the future
                item['future'].cancel()
            except asyncio.QueueEmpty:
                break
        
        return {
            "message": f"Flushed {queue_size_before} queued requests",
            "flushed_count": queue_size_before
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shim/resources")
async def get_resource_status():
    """Get current system resource status"""
    try:
        shim: OllamaAPIShim = app.state.shim
        await shim.resource_monitor.check_resources()  # Force check
        
        return {
            "resources": shim.resource_monitor.resource_stats,
            "thresholds": {
                "cpu_threshold": shim.resource_monitor.cpu_threshold,
                "memory_threshold": shim.resource_monitor.memory_threshold,
                "response_time_threshold": shim.resource_monitor.ollama_response_time_threshold
            },
            "recommendations": shim._get_resource_recommendations()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# def _get_resource_recommendations(self) -> List[str]:
#     """Get recommendations based on current resource usage"""
#     recommendations = []
#     stats = self.resource_monitor.resource_stats
    
#     if stats['cpu_percent'] > 80:
#         recommendations.append("High CPU usage detected. Consider reducing concurrent requests.")
    
#     if stats['memory_percent'] > 85:
#         recommendations.append("High memory usage detected. Consider clearing caches or restarting services.")
    
#     if stats['ollama_avg_response_time'] > 20:
#         recommendations.append("Ollama response times are slow. Check Ollama service health.")
    
#     if stats['system_overloaded']:
#         recommendations.append("System is overloaded. Requests are being queued automatically.")
    
#     if not recommendations:
#         recommendations.append("System resources are healthy.")
    
#     return recommendations

# ============================================================================
# VS CODE COMPATIBLE ENDPOINTS
# ============================================================================

@app.get("/v1/models")
async def list_models_openai():
    """OpenAI-compatible models endpoint for VS Code extensions"""
    try:
        # Get models from Ollama
        response = await services.ollama_client.get("/api/tags")
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch models from Ollama")
        
        ollama_models = response.json()
        
        # Convert to OpenAI format
        models = []
        for model in ollama_models.get("models", []):
            models.append({
                "id": model["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "permission": [],
                "root": model["name"],
                "parent": None
            })
        
        return {
            "object": "list",
            "data": models
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions_openai(request: dict, req: Request):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Extract session ID from headers
        session_id = req.headers.get("X-Session-ID") or str(uuid.uuid4())
        
        # Convert OpenAI format to Ollama format
        messages = request.get("messages", [])
        model = request.get("model", "llama2")
        stream = request.get("stream", False)
        max_tokens = request.get("max_tokens", 2048)
        temperature = request.get("temperature", 0.7)
        
        # Build system message and conversation
        system_parts = []
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation_messages.append(msg)
        
        # Create Ollama request
        if conversation_messages:
            # Use chat format if available
            ollama_request = OllamaRequest(
                model=model,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in conversation_messages],
                stream=stream,
                system="\n".join(system_parts) if system_parts else None,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
        else:
            # Fallback to prompt format
            prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
            ollama_request = OllamaRequest(
                model=model,
                prompt=prompt,
                stream=stream,
                system="\n".join(system_parts) if system_parts else None,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
        
        shim: OllamaAPIShim = app.state.shim
        
        if stream:
            return StreamingResponse(
                openai_streaming_wrapper(shim, ollama_request, session_id, model),
                media_type="text/plain"
            )
        else:
            # Process through shim
            result = await shim.process_request(ollama_request, session_id)
            
            # Convert to OpenAI format
            content = result.get("response", "") if result else ""
            
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(str(messages)) // 4,  # Rough estimate
                    "completion_tokens": len(content) // 4,
                    "total_tokens": (len(str(messages)) + len(content)) // 4
                }
            }
            
    except Exception as e:
        logging.error(f"OpenAI chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def openai_streaming_wrapper(shim: OllamaAPIShim, request: OllamaRequest, session_id: str, model: str):
    """Wrapper to convert Ollama streaming to OpenAI format"""
    try:
        async for chunk in shim.process_streaming_request(request, session_id):
            if chunk.startswith("data: "):
                try:
                    # Parse Ollama chunk
                    ollama_data = json.loads(chunk[6:])  # Remove "data: " prefix
                    
                    if "response" in ollama_data:
                        # Convert to OpenAI format
                        openai_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": ollama_data["response"]
                                },
                                "finish_reason": None
                            }]
                        }
                        
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                    
                    if ollama_data.get("done", False):
                        # Send final chunk
                        final_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        logging.error(f"OpenAI streaming wrapper failed: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/completions")
async def completions_openai(request: dict, req: Request):
    """OpenAI-compatible completions endpoint"""
    try:
        session_id = req.headers.get("X-Session-ID") or str(uuid.uuid4())
        
        prompt = request.get("prompt", "")
        model = request.get("model", "llama2")
        max_tokens = request.get("max_tokens", 2048)
        temperature = request.get("temperature", 0.7)
        stream = request.get("stream", False)
        
        ollama_request = OllamaRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        
        shim: OllamaAPIShim = app.state.shim
        
        if stream:
            return StreamingResponse(
                openai_completions_streaming_wrapper(shim, ollama_request, session_id, model),
                media_type="text/plain"
            )
        else:
            result = await shim.process_request(ollama_request, session_id)
            content = result.get("response", "") if result else ""
            
            return {
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "text": content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(content) // 4,
                    "total_tokens": (len(prompt) + len(content)) // 4
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def openai_completions_streaming_wrapper(shim: OllamaAPIShim, request: OllamaRequest, session_id: str, model: str):
    """Wrapper for completions streaming in OpenAI format"""
    try:
        async for chunk in shim.process_streaming_request(request, session_id):
            if chunk.startswith("data: "):
                try:
                    ollama_data = json.loads(chunk[6:])
                    
                    if "response" in ollama_data:
                        openai_chunk = {
                            "id": f"cmpl-{uuid.uuid4()}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "text": ollama_data["response"],
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                    
                    if ollama_data.get("done", False):
                        yield "data: [DONE]\n\n"
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/embeddings")
async def embeddings_openai(request: dict):
    """OpenAI-compatible embeddings endpoint"""
    try:
        input_text = request.get("input", "")
        model = request.get("model", "nomic-embed-text")
        
        # Handle both string and array inputs
        if isinstance(input_text, str):
            texts = [input_text]
        else:
            texts = input_text
        
        embeddings_data = []
        
        # Use our embedding cache system
        memory_manager = MemoryManager(services)
        
        for i, text in enumerate(texts):
            embedding = await memory_manager.get_embedding(text)
            embeddings_data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        return {
            "object": "list",
            "data": embeddings_data,
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(text) // 4 for text in texts),
                "total_tokens": sum(len(text) // 4 for text in texts)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# VS CODE EXTENSION SPECIFIC ENDPOINTS
# ============================================================================

@app.post("/v1/vscode/explain")
async def vscode_explain_code(request: dict, req: Request):
    """Explain code snippet for VS Code extensions"""
    try:
        session_id = req.headers.get("X-Session-ID") or f"vscode-{uuid.uuid4()}"
        
        code = request.get("code", "")
        language = request.get("language", "unknown")
        context = request.get("context", "")
        
        prompt = f"""Please explain the following {language} code:

```{language}
{code}
```

{f"Context: {context}" if context else ""}

Provide a clear, concise explanation of what this code does, including:
1. Overall purpose
2. Key components and their functions
3. Any notable patterns or techniques used
4. Potential improvements or considerations
"""
        
        ollama_request = OllamaRequest(
            model=request.get("model", "llama2"),
            prompt=prompt,
            stream=False,
            options={"temperature": 0.3}  # Lower temperature for more consistent explanations
        )
        
        shim: OllamaAPIShim = app.state.shim
        result = await shim.process_request(ollama_request, session_id)
        
        return {
            "explanation": result.get("response", "") if result else "",
            "language": language,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vscode/complete")
async def vscode_code_completion(request: dict, req: Request):
    """Code completion for VS Code extensions"""
    try:
        session_id = req.headers.get("X-Session-ID") or f"vscode-{uuid.uuid4()}"
        
        prefix = request.get("prefix", "")
        suffix = request.get("suffix", "")
        language = request.get("language", "")
        max_tokens = request.get("max_tokens", 100)
        
        # Build completion prompt
        prompt = f"""Complete the following {language} code:

{prefix}<CURSOR>{suffix}

Complete the code at the <CURSOR> position. Only provide the completion, not the full code."""
        
        ollama_request = OllamaRequest(
            model=request.get("model", "codellama"),
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.1,  # Low temperature for consistent completions
                "num_predict": max_tokens,
                "stop": ["\n\n", "```"]  # Stop at double newline or code block end
            }
        )
        
        shim: OllamaAPIShim = app.state.shim
        result = await shim.process_request(ollama_request, session_id)
        
        completion = result.get("response", "") if result else ""
        
        return {
            "completion": completion.strip(),
            "language": language,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vscode/review")
async def vscode_code_review(request: dict, req: Request):
    """Code review for VS Code extensions"""
    try:
        session_id = req.headers.get("X-Session-ID") or f"vscode-review-{uuid.uuid4()}"
        
        code = request.get("code", "")
        language = request.get("language", "")
        focus_areas = request.get("focus_areas", ["security", "performance", "maintainability"])
        
        focus_text = ", ".join(focus_areas)
        
        prompt = f"""Please review the following {language} code, focusing on {focus_text}:

```{language}
{code}
```

Provide a structured code review with:
1. **Strengths**: What the code does well
2. **Issues**: Problems or concerns found
3. **Suggestions**: Specific improvements
4. **Security**: Any security considerations
5. **Performance**: Performance implications

Be constructive and specific in your feedback."""
        
        ollama_request = OllamaRequest(
            model=request.get("model", "llama2"),
            prompt=prompt,
            stream=False,
            options={"temperature": 0.2}
        )
        
        shim: OllamaAPIShim = app.state.shim
        result = await shim.process_request(ollama_request, session_id)
        
        return {
            "review": result.get("response", "") if result else "",
            "language": language,
            "focus_areas": focus_areas,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vscode/refactor")
async def vscode_refactor_code(request: dict, req: Request):
    """Code refactoring suggestions for VS Code"""
    try:
        session_id = req.headers.get("X-Session-ID") or f"vscode-refactor-{uuid.uuid4()}"
        
        code = request.get("code", "")
        language = request.get("language", "")
        refactor_type = request.get("type", "general")  # general, performance, readability, etc.
        
        prompt = f"""Refactor the following {language} code to improve {refactor_type}:

```{language}
{code}
```

Provide:
1. **Refactored Code**: The improved version
2. **Changes Made**: Explanation of what was changed and why
3. **Benefits**: How these changes improve the code

Focus on {refactor_type} improvements while maintaining functionality."""
        
        ollama_request = OllamaRequest(
            model=request.get("model", "codellama"),
            prompt=prompt,
            stream=False,
            options={"temperature": 0.3}
        )
        
        shim: OllamaAPIShim = app.state.shim
        result = await shim.process_request(ollama_request, session_id)
        
        return {
            "refactoring": result.get("response", "") if result else "",
            "language": language,
            "type": refactor_type,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vscode/generate")
async def vscode_generate_code(request: dict, req: Request):
    """Generate code from natural language description"""
    try:
        session_id = req.headers.get("X-Session-ID") or f"vscode-generate-{uuid.uuid4()}"
        
        description = request.get("description", "")
        language = request.get("language", "python")
        style = request.get("style", "clean")  # clean, functional, object-oriented, etc.
        
        prompt = f"""Generate {language} code for the following requirement:

**Description**: {description}

**Requirements**:
- Write clean, {style} code
- Include appropriate comments
- Follow {language} best practices
- Make it production-ready

Provide only the code with brief comments explaining key parts."""
        
        ollama_request = OllamaRequest(
            model=request.get("model", "codellama"),
            prompt=prompt,
            stream=False,
            options={"temperature": 0.4}
        )
        
        shim: OllamaAPIShim = app.state.shim
        result = await shim.process_request(ollama_request, session_id)
        
        return {
            "generated_code": result.get("response", "") if result else "",
            "language": language,
            "description": description,
            "style": style,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/vscode/sessions/{session_id}")
async def get_vscode_session(session_id: str):
    """Get VS Code session context and history"""
    try:
        shim: OllamaAPIShim = app.state.shim
        context = await shim.memory_manager.get_or_create_session(session_id)
        
        # Filter for VS Code related messages
        vscode_messages = []
        for msg in context.messages:
            if any(keyword in msg.content.lower() for keyword in ['code', 'function', 'class', 'variable', 'debug']):
                vscode_messages.append({
                    "role": msg.role,
                    "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                })
        
        return {
            "session_id": session_id,
            "messages": vscode_messages[-10:],  # Last 10 relevant messages
            "entities": [e for e in context.knowledge_entities if 'code' in str(e).lower()],
            "context_summary": f"Session with {len(vscode_messages)} code-related interactions"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) #request(request, session_id)

async def generate_stream(request: OllamaRequest, session_id: str, shim: OllamaAPIShim):
    """Streaming generation with augmentation"""
    
    async def stream_generator():
        try:
            # Process request normally first to get context
            augmented_request = request.copy()
            
            # Get context
            user_message = shim.extract_user_message(request)
            if user_message:
                context = await shim.memory_manager.get_or_create_session(session_id)
                await shim.memory_manager.add_message(session_id, "user", user_message)
                
                # Add context
                if Config.ENABLE_MEMORY:
                    context.relevant_memories = await shim.memory_manager.search_similar_memories(
                        user_message, session_id, limit=3
                    )
                
                if Config.ENABLE_KNOWLEDGE_GRAPH:
                    entities = await shim.kg_manager.extract_and_store_entities(
                        user_message, session_id, "user"
                    )
                    context.knowledge_entities = entities
                
                augmented_request = await shim.augment_request(request, context)
            
            # Stream from Ollama
            async with services.ollama_client.stream(
                "POST",
                "/api/generate", 
                json=augmented_request.model_dump()
            ) as response:
                
                full_response = ""
                
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            data = json.loads(chunk)
                            if "response" in data:
                                full_response += data["response"]
                            yield f"data: {chunk}\n\n"
                            
                            # Check if done
                            if data.get("done", False):
                                # Store complete response
                                if full_response:
                                    await shim.memory_manager.add_message(
                                        session_id, "assistant", full_response
                                    )
                                break
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logging.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/api/tags")
async def get_tags():
    """Forward tags request to Ollama"""
    try:
        response = await services.ollama_client.get("/api/tags")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/pull")
async def pull_model(request: dict):
    """Forward model pull request to Ollama"""
    try:
        response = await services.ollama_client.post("/api/pull", json=request)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/push")
async def push_model(request: dict):
    """Forward model push request to Ollama"""
    try:
        response = await services.ollama_client.post("/api/push", json=request)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.delete("/api/delete")
async def delete_model(request: dict):
    """Forward model delete request to Ollama"""
    try:
        response = await services.ollama_client.delete("/api/delete", json=request)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/copy")
async def copy_model(request: dict):
    """Forward model copy request to Ollama"""
    try:
        response = await services.ollama_client.post("/api/copy", json=request)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/show")
async def show_model(request: dict):
    """Forward show model request to Ollama"""
    try:
        response = await services.ollama_client.post("/api/show", json=request)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/embeddings")
async def get_embeddings(request: dict):
    """Forward embeddings request to Ollama"""
    try:
        response = await services.ollama_client.post("/api/embeddings", json=request)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
        
# ============================================================================
# ADDITIONAL SHIM-SPECIFIC ENDPOINTS
# ============================================================================

@app.get("/api/shim/health")
async def health_check():
    """Health check endpoint with queue and resource status"""
    try:
        shim: OllamaAPIShim = app.state.shim
        
        # Check Ollama
        ollama_response = await services.ollama_client.get("/api/tags", timeout=5.0)
        ollama_healthy = ollama_response.status_code == 200
        
        # Check PostgreSQL
        pg_healthy = False
        try:
            async with services.pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                pg_healthy = True
        except:
            pass
        
        # Check Neo4j
        neo4j_healthy = False
        try:
            async with services.neo4j_driver.session() as session:
                await session.run("RETURN 1")
                neo4j_healthy = True
        except:
            pass
        
        # Check Weaviate
        weaviate_healthy = False
        try:
            if services.weaviate_client:
                services.weaviate_client.schema.get()
                weaviate_healthy = True
        except:
            pass
        
        # Get queue and resource status
        queue_stats = shim.request_queue.get_stats()
        resource_stats = shim.resource_monitor.resource_stats
        
        overall_status = "healthy"
        if not all([ollama_healthy, pg_healthy]):
            overall_status = "critical"
        elif resource_stats.get('system_overloaded', False):
            overall_status = "degraded"
        elif queue_stats['queue_size'] > queue_stats['max_queue_size'] * 0.8:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "services": {
                "ollama": ollama_healthy,
                "postgresql": pg_healthy, 
                "neo4j": neo4j_healthy,
                "weaviate": weaviate_healthy
            },
            "queue": {
                "size": queue_stats['queue_size'],
                "active_requests": queue_stats['active_requests'],
                "max_concurrent": queue_stats['max_concurrent'],
                "processing_capacity": f"{queue_stats['active_requests']}/{queue_stats['max_concurrent']}"
            },
            "resources": {
                "cpu_percent": resource_stats.get('cpu_percent', 0),
                "memory_percent": resource_stats.get('memory_percent', 0),
                "system_overloaded": resource_stats.get('system_overloaded', False)
            },
            "active_sessions": len(shim.memory_manager.session_cache),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/shim/sessions")
async def list_sessions():
    """List active conversation sessions"""
    try:
        sessions = []
        for session_id, context in app.state.shim.memory_manager.session_cache.items():
            sessions.append({
                "session_id": session_id,
                "message_count": len(context.messages),
                "last_message": context.messages[-1].timestamp.isoformat() if context.messages else None,
                "entities": len(context.knowledge_entities),
                "available_tools": context.available_tools
            })
        
        return {"sessions": sessions, "count": len(sessions)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shim/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get detailed session information"""
    try:
        context = await app.state.shim.memory_manager.get_or_create_session(session_id)
        
        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "metadata": msg.metadata
                }
                for msg in context.messages
            ],
            "knowledge_entities": context.knowledge_entities,
            "relevant_memories": context.relevant_memories,
            "available_tools": context.available_tools
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/shim/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    try:
        # Remove from cache
        if session_id in app.state.shim.memory_manager.session_cache:
            del app.state.shim.memory_manager.session_cache[session_id]
        
        # Remove from database
        async with services.pg_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conversations WHERE session_id = $1",
                session_id
            )
        
        # Remove from Neo4j
        async with services.neo4j_driver.session() as session:
            await session.run(
                "MATCH (c:Conversation {session_id: $session_id}) DETACH DELETE c",
                session_id=session_id
            )
        
        return {"message": "Session deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shim/search")
async def search_knowledge(q: str, session_id: str = None, limit: int = 10):
    """Search knowledge base"""
    try:
        results = {}
        
        if session_id:
            # Search memories
            memories = await app.state.shim.memory_manager.search_similar_memories(
                q, session_id, limit
            )
            results["memories"] = memories
            
            # Search entities
            entities = await app.state.shim.kg_manager.get_related_entities(
                q, session_id, limit
            )
            results["entities"] = entities
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shim/tools/{tool_name}")
async def invoke_tool(tool_name: str, parameters: dict):
    """Manually invoke a tool"""
    try:
        result = await app.state.shim.tool_manager.invoke_tool(tool_name, parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shim/tools")
async def list_tools():
    """List available tools"""
    return {"tools": app.state.shim.tool_manager.available_tools}

# ============================================================================
# METRICS ENDPOINT
# ============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(registry),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

# ============================================================================
# ENDPOINT ENDPOINT
# ============================================================================

@app.get("/endpoints")
def list_endpoints():
    routes = []
    for route in app.routes:
        if hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return routes

# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from starlette.types import Receive

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# # in-memory request log
# request_log = []
# # Helper: decide per-request whether to buffer or stream
# def should_buffer(request: Request, max_buffer_size: int = 10_000_000) -> bool:
#     """
#     Buffer small requests, stream large or chunked requests.
#     """
#     transfer_encoding = request.headers.get("transfer-encoding", "").lower()
#     content_length = request.headers.get("content-length")

#     # If transfer-encoding chunked or no content-length, stream
#     if transfer_encoding == "chunked" or content_length is None:
#         return False

#     # If content-length exceeds limit, stream
#     try:
#         if int(content_length) > max_buffer_size:
#             return False
#     except ValueError:
#         return False

#     return True  # buffer small bodies

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     if should_buffer(request):
#         # --- Buffered Mode ---
#         body = await request.body()  # read fully

#         # Re-inject body for downstream consumers
#         async def receive():
#             return {"type": "http.request", "body": body, "more_body": False}

#         request = Request(request.scope, receive)

#         response: Response = await call_next(request)

#         request_log.append({
#             "method": request.method,
#             "url": str(request.url),
#             "query_params": dict(request.query_params),
#             "headers": dict(request.headers),
#             "body": body.decode("utf-8", errors="ignore"),
#             "mode": "buffered",
#         })

#     else:
#         # --- Streaming Mode ---
#         body_chunks = []
#         original_receive = request._receive

#         async def receive_wrapper():
#             message = await original_receive()
#             if message["type"] == "http.request":
#                 chunk = message.get("body", b"")
#                 if chunk:
#                     body_chunks.append(chunk)
#             return message

#         request._receive = receive_wrapper
#         response: Response = await call_next(request)

#         full_body = b"".join(body_chunks)
#         request_log.append({
#             "method": request.method,
#             "url": str(request.url),
#             "query_params": dict(request.query_params),
#             "headers": dict(request.headers),
#             "body": full_body.decode("utf-8", errors="ignore"),
#             "mode": "streamed",
#         })

#     return response

# @app.get("/requests", response_class=HTMLResponse)
# async def show_requests(request: Request):
#     return templates.TemplateResponse("requests.html", {"request": request, "logs": request_log})


# ============================================================================
# Passthrough ENDPOINT
# ============================================================================

# Catch-all route for everything else → transparent pass-through
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(full_path: str, request: Request):
    # Default route prefers CPU if no model-specific logic
    instance_name = "cpu"
    target_url = f"http://ollama:11434/{full_path}"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            body = await request.body()
            headers = dict(request.headers)
            response = await client.request(
                request.method, target_url, content=body, headers=headers
            )
            return httpx.Response(
                status_code=response.status_code,
                content=response.content,
                headers=response.headers
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )