# agentic-core/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import asyncpg
from neo4j import AsyncGraphDatabase
import httpx
import schedule
import time
from typing import Optional, Dict, Any, List
import json
import logging
from datetime import datetime, timedelta
import os
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Core", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MODEL_ROUTER_URL = os.getenv("MODEL_ROUTER_URL")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
TOOL_SERVER_URL = os.getenv("TOOL_SERVER_URL")
GRAFITI_URL = os.getenv("GRAFITI_URL")
PROACTIVE_MODE = os.getenv("PROACTIVE_MODE", "true").lower() == "true"
THOUGHT_INTERVAL = int(os.getenv("THOUGHT_INTERVAL", "300"))

class AgentRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    reasoning: str
    actions_taken: List[str]
    knowledge_updated: bool
    session_id: str
    timestamp: datetime

class AgentMemory:
    def __init__(self):
        self.pg_pool = None
        self.neo4j_driver = None
        
    async def initialize(self):
        """Initialize database connections"""
        self.pg_pool = await asyncpg.create_pool(DATABASE_URL)
        self.neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
    async def store_interaction(self, query: str, response: str, user_id: str, session_id: str):
        """Store interaction in both SQL and graph databases"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO interactions (query, response, user_id, session_id, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """, query, response, user_id, session_id, datetime.now())
        
        # Store in Neo4j as well
        async with self.neo4j_driver.session() as session:
            await session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (s:Session {id: $session_id})
                MERGE (u)-[:HAS_SESSION]->(s)
                CREATE (i:Interaction {
                    query: $query,
                    response: $response,
                    timestamp: datetime()
                })
                MERGE (s)-[:CONTAINS]->(i)
            """, user_id=user_id, session_id=session_id, query=query, response=response)

    async def get_relevant_context(self, query: str, user_id: str) -> Dict[str, Any]:
        """Retrieve relevant context from memory systems"""
        # Get recent interactions
        async with self.pg_pool.acquire() as conn:
            recent_interactions = await conn.fetch("""
                SELECT query, response, timestamp
                FROM interactions
                WHERE user_id = $1
                ORDER BY timestamp DESC
                LIMIT 5
            """, user_id)
        
        # Get graph context
        async with self.neo4j_driver.session() as session:
            result = await session.run("""
                MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(s:Session)-[:CONTAINS]->(i:Interaction)
                WHERE i.query CONTAINS $query_term
                RETURN i.query as query, i.response as response
                ORDER BY i.timestamp DESC
                LIMIT 3
            """, user_id=user_id, query_term=query[:50])
            
            graph_context = [record async for record in result]
        
        return {
            "recent_interactions": recent_interactions,
            "graph_context": graph_context,
            "user_id": user_id
        }

class AgentCore:
    def __init__(self):
        self.memory = AgentMemory()
        self.is_thinking = False
        
    async def initialize(self):
        """Initialize the agent core"""
        await self.memory.initialize()
        
        if PROACTIVE_MODE:
            # Schedule proactive thinking
            schedule.every(THOUGHT_INTERVAL).seconds.do(self.proactive_think)
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process an agent request"""
        session_id = request.session_id or f"session_{int(time.time())}"
        user_id = request.user_id or "anonymous"
        
        # Get relevant context
        context = await self.memory.get_relevant_context(request.query, user_id)
        
        # Route through model router
        async with httpx.AsyncClient() as client:
            model_response = await client.post(f"{MODEL_ROUTER_URL}/route", json={
                "prompt": request.query,
                "context": json.dumps(context),
                "parameters": {}
            })
            
            if model_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Model routing failed")
            
            model_result = model_response.json()
        
        # Determine if tools are needed
        actions_taken = []
        if self._needs_tools(request.query):
            tool_results = await self._execute_tools(request.query, context)
            actions_taken.extend(tool_results)
        
        # Update knowledge graph
        knowledge_updated = await self._update_knowledge_graph(
            request.query, model_result["response"], context
        )
        
        # Store interaction
        await self.memory.store_interaction(
            request.query, model_result["response"], user_id, session_id
        )
        
        return AgentResponse(
            response=model_result["response"],
            reasoning=f"Used {model_result['model_used']} model (complexity: {model_result['complexity_score']:.2f})",
            actions_taken=actions_taken,
            knowledge_updated=knowledge_updated,
            session_id=session_id,
            timestamp=datetime.now()
        )
    
    def _needs_tools(self, query: str) -> bool:
        """Determine if query needs tool execution"""
        tool_keywords = [
            'search', 'calculate', 'compute', 'fetch', 'retrieve',
            'analyze file', 'send email', 'schedule', 'reminder',
            'weather', 'news', 'current', 'today'
        ]
        return any(keyword in query.lower() for keyword in tool_keywords)
    
    async def _execute_tools(self, query: str, context: Dict) -> List[str]:
        """Execute appropriate tools based on query"""
        actions = []
        
        try:
            # Try MCP server first
            async with httpx.AsyncClient() as client:
                mcp_response = await client.post(f"{MCP_SERVER_URL}/execute", json={
                    "query": query,
                    "context": context
                }, timeout=30.0)
                
                if mcp_response.status_code == 200:
                    result = mcp_response.json()
                    actions.append(f"MCP: {result.get('action', 'executed')}")
                
                # Also try generic tool server
                tool_response = await client.post(f"{TOOL_SERVER_URL}/execute", json={
                    "query": query,
                    "context": context
                }, timeout=30.0)
                
                if tool_response.status_code == 200:
                    result = tool_response.json()
                    actions.append(f"Tool: {result.get('action', 'executed')}")
                    
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            actions.append(f"Tool execution failed: {str(e)}")
        
        return actions
    
    async def _update_knowledge_graph(self, query: str, response: str, context: Dict) -> bool:
        """Update knowledge graph with new information"""
        try:
            async with httpx.AsyncClient() as client:
                kg_response = await client.post(f"{GRAFITI_URL}/update", json={
                    "query": query,
                    "response": response,
                    "context": context
                }, timeout=30.0)
                
                return kg_response.status_code == 200
                
        except Exception as e:
            logger.error(f"Knowledge graph update error: {e}")
            return False
    
    async def proactive_think(self):
        """Proactive thinking process"""
        if self.is_thinking:
            return
            
        self.is_thinking = True
        logger.info("Starting proactive thinking cycle...")
        
        try:
            # Analyze recent patterns
            async with self.memory.pg_pool.acquire() as conn:
                recent_patterns = await conn.fetch("""
                    SELECT query, COUNT(*) as frequency
                    FROM interactions
                    WHERE timestamp > $1
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT 10
                """, datetime.now() - timedelta(hours=24))
            
            # Generate proactive insights
            if recent_patterns:
                insights_query = f"Analyze these interaction patterns and suggest proactive actions: {recent_patterns}"
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(f"{MODEL_ROUTER_URL}/route", json={
                        "prompt": insights_query,
                        "context": "proactive_analysis",
                        "parameters": {"temperature": 0.7}
                    })
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"Proactive insight: {result['response'][:200]}...")
                        
                        # Store proactive thought
                        await self.memory.store_interaction(
                            "PROACTIVE_ANALYSIS",
                            result["response"],
                            "system",
                            f"proactive_{int(time.time())}"
                        )
        
        except Exception as e:
            logger.error(f"Proactive thinking error: {e}")
        finally:
            self.is_thinking = False

# Global agent instance
agent = AgentCore()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await agent.initialize()
    logger.info("Agentic Core initialized successfully")

@app.post("/process", response_model=AgentResponse)
async def process_request(request: AgentRequest):
    """Process an agent request"""
    try:
        return await agent.process_request(request)
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "proactive_mode": PROACTIVE_MODE}

@app.post("/think")
async def trigger_proactive_thinking():
    """Manually trigger proactive thinking"""
    asyncio.create_task(agent.proactive_think())
    return {"status": "thinking_initiated"}

@app.get("/memory/stats")
async def memory_stats():
    """Get memory system statistics"""
    try:
        async with agent.memory.pg_pool.acquire() as conn:
            interaction_count = await conn.fetchval("SELECT COUNT(*) FROM interactions")
            user_count = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM interactions")
        
        async with agent.memory.neo4j_driver.session() as session:
            result = await session.run("MATCH (n) RETURN COUNT(n) as node_count")
            node_count = await result.single()
            
        return {
            "interactions": interaction_count,
            "users": user_count,
            "graph_nodes": node_count["node_count"] if node_count else 0
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    # Run scheduled tasks in background
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    import threading
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)