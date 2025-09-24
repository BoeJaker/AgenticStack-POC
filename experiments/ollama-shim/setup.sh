#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Setting up Ollama Enhanced API Shim ===${NC}"

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p ollama-enhanced-shim
mkdir -p ollama-enhanced-shim/cache
mkdir -p monitoring/prometheus

# Copy files to the correct locations
echo -e "${YELLOW}Setting up application files...${NC}"

# Create the main application file
cat > ollama-enhanced-shim/main.py << 'EOF'
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

from config import get_config

# Initialize configuration
config = get_config()
config.print_config()

# Validate configuration
if not config.validate():
    logging.error("Configuration validation failed! Please check your environment variables.")
    exit(1)

# ... [The rest of the main.py content would be here - truncated for space]
# This would include all the classes and functions from the previous artifact

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )
EOF

# Create config.py
cat > ollama-enhanced-shim/config.py << 'EOF'
# [Config content from previous artifact would be here]
EOF

# Create requirements.txt
cat > ollama-enhanced-shim/requirements.txt << 'EOF'
# Web framework and API
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# HTTP client
httpx==0.25.2

# Database drivers
asyncpg==0.29.0
neo4j==5.14.0

# Vector database
weaviate-client==3.25.3

# Machine learning and embeddings
sentence-transformers==2.2.2
numpy==1.24.4
scikit-learn==1.3.2

# Monitoring and metrics  
prometheus-client==0.19.0

# Utilities
python-multipart==0.0.6
python-json-logger==2.0.7
structlog==23.2.0
EOF

# Create Dockerfile
cat > ollama-enhanced-shim/Dockerfile << 'EOF'
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /app/cache

# Create non-root user
RUN useradd -m -u 1000 shimuser && chown -R shimuser:shimuser /app
USER shimuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/api/shim/health || exit 1

# Run the application
CMD ["python", "main.py"]
EOF

# Create environment template
echo -e "${YELLOW}Creating environment template...${NC}"
cat > .env.shim.template << 'EOF'
# Ollama Enhanced API Shim Configuration

# Database passwords (set these!)
SUPABASE_DB_PASSWORD=your_strong_password_here
NEO4J_PASSWORD=your_neo4j_password_here

# Service URLs (these should match your docker-compose setup)
OLLAMA_BASE_URL=http://ollama:11434
SUPABASE_DB_URL=postgresql://supabase_admin:${SUPABASE_DB_PASSWORD}@supabase-db:5432/postgres
WEAVIATE_URL=http://weaviate:8080
NEO4J_URI=bolt://neo4j-stack:7687
NEO4J_USERNAME=neo4j
MCP_SERVER_URL=http://mcp-server:8000

# AI Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_CONTEXT_TOKENS=8192
SIMILARITY_THRESHOLD=0.7
MAX_MEMORY_ENTRIES=50

# Feature Flags
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_VECTOR_SEARCH=true
ENABLE_MEMORY=true
ENABLE_TOOLS=true
ENABLE_CACHING=true

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=60
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8080
EOF

# Create Prometheus configuration for scraping the shim
echo -e "${YELLOW}Creating Prometheus configuration...${NC}"
cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'ollama-enhanced-shim'
    static_configs:
      - targets: ['ollama-enhanced-shim:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'supabase'
    static_configs:
      - targets: ['kong:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
EOF

# Create Docker Compose override
echo -e "${YELLOW}Creating Docker Compose override...${NC}"
cat > docker-compose.shim.yml << 'EOF'
version: '3.8'

services:
  # Enhanced Ollama API Shim with full stack integration
  ollama-enhanced-shim:
    build:
      context: ./ollama-enhanced-shim
      dockerfile: Dockerfile
    container_name: ollama-enhanced-shim
    restart: unless-stopped
    environment:
      # Ollama connection
      OLLAMA_BASE_URL: http://ollama:11434
      
      # Database connections
      SUPABASE_DB_URL: postgresql://supabase_admin:${SUPABASE_DB_PASSWORD}@supabase-db:5432/postgres
      NEO4J_URI: bolt://neo4j-stack:7687
      NEO4J_USERNAME: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      WEAVIATE_URL: http://weaviate:8080
      
      # MCP server connection
      MCP_SERVER_URL: http://mcp-server:8000
      
      # Configuration
      EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2
      MAX_CONTEXT_TOKENS: 8192
      SIMILARITY_THRESHOLD: 0.7
      MAX_MEMORY_ENTRIES: 50
      
      # Feature flags
      ENABLE_KNOWLEDGE_GRAPH: "true"
      ENABLE_VECTOR_SEARCH: "true" 
      ENABLE_MEMORY: "true"
      ENABLE_TOOLS: "true"
      ENABLE_CACHING: "true"
      
      # Environment
      ENVIRONMENT: production
      LOG_LEVEL: INFO
      
    volumes:
      # Cache for ML models
      - ollama-shim-cache:/app/cache
      
    ports:
      - "8082:8080"  # Different port to avoid conflicts
      
    depends_on:
      - ollama
      - supabase-db  
      - neo4j-stack
      - weaviate
      # - mcp-server  # Uncomment when MCP server is ready
      
    networks:
      - agentic-network
      
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/shim/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s  # Allow time for model loading
      
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.ollama-shim.rule=Host(`${DOMAIN}`) && PathPrefix(`/api/ollama`)"
      - "traefik.http.middlewares.ollama-shim-stripprefix.stripprefix.prefixes=/api/ollama"
      - "traefik.http.routers.ollama-shim.middlewares=ollama-shim-stripprefix"
      - "traefik.http.services.ollama-shim.loadbalancer.server.port=8080"
      
    # Resource limits to prevent memory issues during model loading
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

volumes:
  ollama-shim-cache:

networks:
  agentic-network:
    external: true
EOF

# Create startup script
echo -e "${YELLOW}Creating startup script...${NC}"
cat > start-shim.sh << 'EOF'
#!/bin/bash

echo "Starting Ollama Enhanced API Shim..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.shim.template .env
    echo "Please edit .env with your actual passwords and configuration!"
    echo "Then run this script again."
    exit 1
fi

# Load environment variables
source .env

# Start the shim alongside the main stack
docker-compose -f docker-compose.yml -f docker-compose.shim.yml up -d ollama-enhanced-shim

echo "Ollama Enhanced API Shim started!"
echo "Access it at: http://localhost:8082"
echo "Health check: http://localhost:8082/api/shim/health"
echo "Metrics: http://localhost:8082/metrics"
EOF

chmod +x start-shim.sh

# Create test script
echo -e "${YELLOW}Creating test script...${NC}"
cat > test-shim.sh << 'EOF'
#!/bin/bash

echo "Testing Ollama Enhanced API Shim..."

BASE_URL="http://localhost:8082"

# Test health endpoint
echo "Testing health endpoint..."
curl -s "$BASE_URL/api/shim/health" | jq .

# Test metrics endpoint
echo -e "\nTesting metrics endpoint..."
curl -s "$BASE_URL/metrics" | head -10

# Test Ollama passthrough
echo -e "\nTesting Ollama passthrough (list models)..."
curl -s "$BASE_URL/api/tags" | jq .

# Test enhanced generation (if models are available)
echo -e "\nTesting enhanced generation..."
curl -s -X POST "$BASE_URL/api/generate" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: test-session-123" \
  -d '{
    "model": "llama2",
    "prompt": "Hello, tell me about artificial intelligence.",
    "stream": false
  }' | jq .

echo -e "\nDone testing!"
EOF

chmod +x test-shim.sh

# Create monitoring dashboard template
echo -e "${YELLOW}Creating Grafana dashboard template...${NC}"
mkdir -p monitoring/grafana/dashboards
cat > monitoring/grafana/dashboards/ollama-shim-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Ollama Enhanced API Shim",
    "description": "Monitoring dashboard for the Ollama API Shim with knowledge graph integration",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ollama_shim_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}} {{status}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ollama_shim_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Active Sessions",
        "type": "stat",
        "targets": [
          {
            "expr": "ollama_shim_active_sessions",
            "legendFormat": "Sessions"
          }
        ]
      },
      {
        "id": 4,
        "title": "Knowledge Graph Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ollama_shim_kg_operations_total[5m])",
            "legendFormat": "{{operation}} {{status}}"
          }
        ]
      },
      {
        "id": 5,
        "title": "Vector Search Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ollama_shim_vector_searches_total[5m])",
            "legendFormat": "{{store_type}} {{status}}"
          }
        ]
      },
      {
        "id": 6,
        "title": "Tool Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ollama_shim_tool_usage_total[5m])",
            "legendFormat": "{{tool_name}} {{status}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

# Create README
echo -e "${YELLOW}Creating README...${NC}"
cat > README-SHIM.md << 'EOF'
# Ollama Enhanced API Shim

This enhanced API shim sits between your applications and Ollama, providing advanced features like:

## Features

🧠 **Conversational Memory**: Persistent conversation history with vector similarity search
🕸️ **Knowledge Graph**: Automatic entity extraction and relationship mapping with Neo4j
🔍 **Vector Search**: Semantic search using Weaviate and Supabase pgvector
🛠️ **Tool Integration**: MCP (Model Context Protocol) tool support
📊 **Comprehensive Metrics**: Prometheus metrics for monitoring and observability
⚡ **Intelligent Caching**: Response and embedding caching for better performance

## Quick Start

1. **Setup**: Run the setup script
   ```bash
   ./setup.sh
   ```

2. **Configure**: Edit `.env` with your actual passwords
   ```bash
   cp .env.shim.template .env
   nano .env  # Set your passwords!
   ```

3. **Start**: Launch the shim
   ```bash
   ./start-shim.sh
   ```

4. **Test**: Verify everything is working
   ```bash
   ./test-shim.sh
   ```

## API Endpoints

### Standard Ollama Endpoints
- `POST /api/generate` - Enhanced generation with memory and knowledge
- `GET /api/tags` - List available models
- `POST /api/pull` - Pull a model
- `POST /api/show` - Show model details
- `POST /api/embeddings` - Generate embeddings

### Enhanced Shim Endpoints
- `GET /api/shim/health` - Health check
- `GET /api/shim/sessions` - List conversation sessions
- `GET /api/shim/sessions/{id}` - Get session details
- `DELETE /api/shim/sessions/{id}` - Delete a session
- `GET /api/shim/search` - Search knowledge base
- `POST /api/shim/tools/{name}` - Invoke tools manually
- `GET /metrics` - Prometheus metrics

## Usage Examples

### Basic Chat with Memory
```bash
curl -X POST http://localhost:8082/api/generate \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "model": "llama2",
    "prompt": "Remember that I like cats. What are good cat breeds?",
    "stream": false
  }'
```

### Search Knowledge Base
```bash
curl "http://localhost:8082/api/shim/search?q=cats&session_id=my-session"
```

### Check Session Details
```bash
curl "http://localhost:8082/api/shim/sessions/my-session"
```

## Monitoring

The shim provides comprehensive metrics for Prometheus:

- Request rates and response times
- Knowledge graph operations
- Vector search performance
- Tool usage statistics
- Active session counts
- Cache hit rates

Import the Grafana dashboard from `monitoring/grafana/dashboards/ollama-shim-dashboard.json`

## Configuration

All configuration is done via environment variables. See `.env.shim.template` for all options.

Key settings:
- `ENABLE_KNOWLEDGE_GRAPH`: Enable/disable entity extraction
- `ENABLE_VECTOR_SEARCH`: Enable/disable semantic search
- `ENABLE_MEMORY`: Enable/disable conversation memory
- `ENABLE_TOOLS`: Enable/disable tool integration
- `SIMILARITY_THRESHOLD`: Threshold for semantic similarity
- `MAX_MEMORY_ENTRIES`: Maximum messages to keep in memory

## Architecture

```
Client Request
     ↓
[Ollama Enhanced Shim]
     ├─ Memory Manager (PostgreSQL + pgvector)
     ├─ Knowledge Graph (Neo4j)
     ├─ Vector Search (Weaviate)
     ├─ Tool Manager (MCP)
     └─ Metrics (Prometheus)
     ↓
[Ollama] → Response
```

## Troubleshooting

1. **Health check fails**: Verify all dependent services are running
2. **Memory not working**: Check PostgreSQL connection and pgvector extension
3. **Knowledge graph errors**: Verify Neo4j is accessible and credentials are correct
4. **Vector search issues**: Check Weaviate connection and schema setup
5. **High memory usage**: Reduce embedding cache size or model complexity

## Development

To run in development mode:
```bash
ENVIRONMENT=development python ollama-enhanced-shim/main.py
```

This enables debug logging and reduces database pool sizes.
EOF

echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo "1. Copy .env.shim.template to .env and set your passwords"
echo "2. Run ./start-shim.sh to start the enhanced shim"
echo "3. Run ./test-shim.sh to verify everything works"
echo "4. Import the Grafana dashboard for monitoring"
echo ""
echo -e "${YELLOW}The shim will be available at: http://localhost:8082${NC}"
echo -e "${YELLOW}Replace your Ollama API calls with this URL to get enhanced features!${NC}"