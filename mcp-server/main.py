from fastapi import FastAPI
import os

app = FastAPI(title="MCP Tool Server", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mcp-server"}

@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {"name": "database_query", "description": "Query the database"},
            {"name": "graph_query", "description": "Query the knowledge graph"},
            {"name": "memory_search", "description": "Search memory store"}
        ]
    }
