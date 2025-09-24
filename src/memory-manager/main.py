from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
from neo4j import GraphDatabase
import redis
import requests
import json
import logging
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Memory Manager", version="1.0.0")

class MemoryItem(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    memory_type: str = "conversation"

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    memory_types: List[str] = ["conversation", "knowledge", "context"]

class MemoryManager:
    def __init__(self):
        self.postgres_url = os.getenv("POSTGRES_URL")
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        
        # Initialize connections
        self.redis_client = redis.from_url(self.redis_url)
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def get_db_connection(self):
        """Get PostgreSQL connection"""
        return await asyncpg.connect(self.postgres_url)
    
    async def store_memory(self, item: MemoryItem) -> str:
        """Store memory item in appropriate storage"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(item.content).tolist()
            
            # Store in PostgreSQL with vector
            conn = await self.get_db_connection()
            try:
                memory_id = await conn.fetchval("""
                    INSERT INTO memories (content, embedding, metadata, memory_type, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    RETURNING id
                """, item.content, embedding, json.dumps(item.metadata), item.memory_type)
                
                # Store in Redis for fast access
                self.redis_client.setex(
                    f"memory:{memory_id}",
                    3600,  # 1 hour TTL
                    json.dumps({
                        "content": item.content,
                        "metadata": item.metadata,
                        "memory_type": item.memory_type
                    })
                )
                
                logger.info(f"Stored memory item {memory_id}")
                return str(memory_id)
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")
    
    async def query_memories(self, query: QueryRequest) -> List[Dict[str, Any]]:
        """Query memories using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query.query).tolist()
            
            # Search in PostgreSQL using cosine similarity
            conn = await self.get_db_connection()
            try:
                results = await conn.fetch("""
                    SELECT id, content, metadata, memory_type,
                           1 - (embedding <=> $1) as similarity
                    FROM memories
                    WHERE memory_type = ANY($2)
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, query_embedding, query.memory_types, query.limit)
                
                memories = []
                for row in results:
                    memories.append({
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]),
                        "memory_type": row["memory_type"],
                        "similarity": float(row["similarity"])
                    })
                
                return memories
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Error querying memories: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to query memories: {str(e)}")

memory_manager = MemoryManager()

@app.post("/store")
async def store_memory(item: MemoryItem):
    """Store a memory item"""
    memory_id = await memory_manager.store_memory(item)
    return {"memory_id": memory_id, "status": "stored"}

@app.post("/query")
async def query_memories(request: QueryRequest):
    """Query memories by similarity"""
    memories = await memory_manager.query_memories(request)
    return {"memories": memories, "count": len(memories)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "memory-manager"}

@app.on_event("startup")
async def startup_event():
    """Initialize database tables"""
    try:
        conn = await asyncpg.connect(memory_manager.postgres_url)
        await conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384),
                metadata JSONB,
                memory_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_memories_embedding 
            ON memories USING ivfflat (embedding vector_cosine_ops);
            
            CREATE INDEX IF NOT EXISTS idx_memories_type 
            ON memories (memory_type);
        """)
        await conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
