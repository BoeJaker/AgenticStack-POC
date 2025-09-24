from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Graphiti imports (you'll need to install: pip install graphiti-ai)
from graphiti import Graphiti
from graphiti.nodes import Node
from graphiti.edges import Edge

OLLAMA_URL = "http://ollama:11434"
app = FastAPI()

# Initialize Graphiti client
# Configure with your preferred database (Neo4j, FalkorDB, etc.)
graphiti_client = Graphiti(
    host="your-graph-db-host",  # Configure your graph database
    port=7687,
    user="neo4j",
    password="password"
)

# Map OpenAI model names to Ollama local models
MODEL_MAP = {
    "gpt-3.5-turbo": "llama2",
    "gpt-4": "mistral",
    # Add aliases as needed
}

def map_model(model: str) -> str:
    return MODEL_MAP.get(model, model)

def openai_error(message: str, code: str = "invalid_request_error"):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
    )

class ConversationMemory:
    """Enhanced conversation memory using Graphiti"""
    
    def __init__(self, graphiti_client):
        self.graphiti = graphiti_client
    
    async def add_conversation_turn(self, user_id: str, messages: List[Dict], model: str, response: str):
        """Add conversation turn to knowledge graph"""
        timestamp = datetime.now().isoformat()
        turn_id = str(uuid.uuid4())
        
        # Create user node if doesn't exist
        user_node = Node(
            id=f"user_{user_id}",
            name=f"User {user_id}",
            node_type="User",
            properties={"created_at": timestamp}
        )
        
        # Create conversation turn node
        turn_node = Node(
            id=f"turn_{turn_id}",
            name=f"Conversation Turn {turn_id}",
            node_type="ConversationTurn",
            properties={
                "messages": json.dumps(messages),
                "model_used": model,
                "response": response,
                "timestamp": timestamp,
                "turn_count": len(messages)
            }
        )
        
        # Create relationship between user and turn
        user_turn_edge = Edge(
            source_node_id=user_node.id,
            target_node_id=turn_node.id,
            edge_type="HAD_CONVERSATION",
            properties={
                "timestamp": timestamp,
                "valid_at": timestamp
            }
        )
        
        # Extract entities and topics from messages
        await self._extract_and_link_entities(messages, turn_node, timestamp)
        
        # Add to graph
        await self.graphiti.add_node(user_node)
        await self.graphiti.add_node(turn_node)
        await self.graphiti.add_edge(user_turn_edge)
    
    async def _extract_and_link_entities(self, messages: List[Dict], turn_node: Node, timestamp: str):
        """Extract entities from conversation and create knowledge graph relationships"""
        # This would use your Ollama models for NER and topic extraction
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                
                # Simple keyword extraction (you could enhance this with Ollama NER)
                # For now, let's create topic nodes for key concepts
                words = content.lower().split()
                topics = [word for word in words if len(word) > 4]  # Simple topic extraction
                
                for topic in topics[:5]:  # Limit to prevent spam
                    topic_node = Node(
                        id=f"topic_{topic}",
                        name=topic.title(),
                        node_type="Topic",
                        properties={"first_mentioned": timestamp}
                    )
                    
                    topic_edge = Edge(
                        source_node_id=turn_node.id,
                        target_node_id=topic_node.id,
                        edge_type="DISCUSSES",
                        properties={
                            "timestamp": timestamp,
                            "context": content[:200]  # Store context snippet
                        }
                    )
                    
                    await self.graphiti.add_node(topic_node)
                    await self.graphiti.add_edge(topic_edge)
    
    async def get_conversation_context(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve recent conversation context for user"""
        query = """
        MATCH (u:User {id: $user_id})-[r:HAD_CONVERSATION]->(t:ConversationTurn)
        RETURN t.properties.messages, t.properties.response, t.properties.timestamp
        ORDER BY t.properties.timestamp DESC
        LIMIT $limit
        """
        
        result = await self.graphiti.query(query, {"user_id": f"user_{user_id}", "limit": limit})
        return [{"messages": json.loads(row["t.properties.messages"]), 
                 "response": row["t.properties.response"],
                 "timestamp": row["t.properties.timestamp"]} for row in result]
    
    async def get_related_topics(self, user_id: str, current_message: str) -> List[str]:
        """Find related topics from user's conversation history"""
        # Simple approach - you could enhance with semantic similarity
        words = current_message.lower().split()
        topics = [word for word in words if len(word) > 4]
        
        if not topics:
            return []
        
        query = """
        MATCH (u:User {id: $user_id})-[:HAD_CONVERSATION]->(t:ConversationTurn)-[d:DISCUSSES]->(topic:Topic)
        WHERE topic.name IN $topics
        RETURN DISTINCT topic.name, COUNT(d) as frequency
        ORDER BY frequency DESC
        LIMIT 10
        """
        
        result = await self.graphiti.query(query, {
            "user_id": f"user_{user_id}", 
            "topics": [t.title() for t in topics]
        })
        return [row["topic.name"] for row in result]

# Initialize conversation memory
conversation_memory = ConversationMemory(graphiti_client)

# --------------------------
# Enhanced Chat Completions with Memory
# --------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = map_model(body.get("model", "llama2"))
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    # Extract user_id from custom header or generate one
    user_id = request.headers.get("x-user-id", "anonymous")
    use_memory = request.headers.get("x-use-memory", "false").lower() == "true"

    if not messages:
        return openai_error("`messages` is required.")
    
    enhanced_messages = messages.copy()
    
    # Enhanced memory integration
    if use_memory and len(messages) > 0:
        last_message = messages[-1].get("content", "")
        
        # Get conversation context
        context = await conversation_memory.get_conversation_context(user_id, limit=3)
        if context:
            context_summary = "Previous conversation context:\n"
            for ctx in context[-2:]:  # Last 2 turns
                context_summary += f"Q: {ctx['messages'][-1].get('content', '')[:100]}...\n"
                context_summary += f"A: {ctx['response'][:100]}...\n\n"
            
            # Inject context as system message
            enhanced_messages.insert(0, {
                "role": "system", 
                "content": f"{context_summary}Please use this context to provide more personalized responses."
            })
        
        # Get related topics
        related_topics = await conversation_memory.get_related_topics(user_id, last_message)
        if related_topics:
            enhanced_messages.insert(-1, {
                "role": "system",
                "content": f"Related topics from previous conversations: {', '.join(related_topics)}"
            })

    ollama_payload = {"model": model, "messages": enhanced_messages, "stream": stream}

    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            # For streaming, we'll collect the response to store in memory
            collected_response = ""
            
            async def event_stream():
                nonlocal collected_response
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            # Parse the line to extract response content
                            try:
                                data = json.loads(line)
                                if data.get("message", {}).get("content"):
                                    collected_response += data["message"]["content"]
                            except:
                                pass
                            yield f"data: {line}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    # Store conversation in memory after streaming
                    if use_memory and collected_response:
                        asyncio.create_task(conversation_memory.add_conversation_turn(
                            user_id, messages, model, collected_response
                        ))
            
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload)
            if resp.status_code != 200:
                return openai_error(f"Ollama error: {resp.text}")

            data = resp.json()
            now = int(time.time())
            
            response_content = data.get("message", {}).get("content", "")
            
            # Store conversation in memory
            if use_memory and response_content:
                asyncio.create_task(conversation_memory.add_conversation_turn(
                    user_id, messages, model, response_content
                ))

            return JSONResponse({
                "id": f"chatcmpl-{now}",
                "object": "chat.completion",
                "created": now,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": data.get("message", {}),
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(str(enhanced_messages)),
                    "completion_tokens": len(str(data.get("message", {}))),
                    "total_tokens": len(str(enhanced_messages)) + len(str(data.get("message", {})))
                }
            })

# --------------------------
# Memory Management Endpoints
# --------------------------
@app.get("/v1/memory/conversations/{user_id}")
async def get_user_conversations(user_id: str, limit: int = 10):
    """Get conversation history for a user"""
    try:
        conversations = await conversation_memory.get_conversation_context(user_id, limit)
        return {"conversations": conversations}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/v1/memory/topics/{user_id}")
async def get_user_topics(user_id: str):
    """Get topics discussed by a user"""
    try:
        query = """
        MATCH (u:User {id: $user_id})-[:HAD_CONVERSATION]->(t:ConversationTurn)-[d:DISCUSSES]->(topic:Topic)
        RETURN topic.name, COUNT(d) as frequency, MAX(d.timestamp) as last_mentioned
        ORDER BY frequency DESC, last_mentioned DESC
        """
        
        result = await graphiti_client.query(query, {"user_id": f"user_{user_id}"})
        topics = [{
            "name": row["topic.name"],
            "frequency": row["frequency"],
            "last_mentioned": row["last_mentioned"]
        } for row in result]
        
        return {"topics": topics}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/v1/memory/search")
async def semantic_search_memory(request: Request):
    """Search conversation memory semantically"""
    body = await request.json()
    query = body.get("query", "")
    user_id = body.get("user_id")
    limit = body.get("limit", 10)
    
    try:
        # Use Graphiti's hybrid search capabilities
        results = await graphiti_client.search(
            query_text=query,
            search_type="hybrid",  # semantic + BM25 + graph
            filters={"user_id": f"user_{user_id}"} if user_id else None,
            limit=limit
        )
        
        return {"results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --------------------------
# Original endpoints (completions, embeddings, models, moderations, transcriptions)
# Keep your existing implementations but could enhance with memory too
# --------------------------

# ... (keep your existing endpoints)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)