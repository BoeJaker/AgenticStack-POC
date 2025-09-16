# Agentic AI Docker Stack

A comprehensive, self-hosted agentic AI system with knowledge graph memory, proactive thinking, and multi-modal capabilities.

## 🏗️ Architecture Overview

### Main Stack
![Stack-Architecture](images/Stack.svg)

### Agent Architecture
![Agent-Architecture](images/architecture.svg)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB+ RAM recommended
- 100GB+ storage space
- (Optional) NVIDIA GPU for heavy models

### 1. Clone and Setup
```bash
git clone <repository>
cd agentic-ai-stack
make setup
```

### 2. Start the Stack
```bash
make up
```

### 3. Initialize and Configure
```bash
# Wait for services to start
make health

# Setup AI models
make setup-models

# Check status
make status
```

### 4. Access Services
- **Main UI**: http://localhost/webui
- **Workflow Designer**: http://localhost/n8n  
- **AI Flow Builder**: http://localhost/flowise
- **Graph Database**: http://localhost:7474
- **Monitoring**: http://localhost/grafana

## 🧠 Key Features

### Intelligent Model Routing
- **Light models** (llama3.2:3b) for simple queries
- **Heavy models** (llama3.1:8b+) for complex reasoning
- **Automatic complexity detection** using linguistic analysis
- **Load balancing** and **performance optimization**

### Proactive AI Agent
- **Autonomous thinking** cycles every 5 minutes (configurable)
- **Pattern analysis** of user interactions
- **Proactive suggestions** and insights
- **Context-aware** memory system

### Advanced Memory System
- **Vector embeddings** for semantic search
- **Knowledge graph** relationships in Neo4j
- **SQL database** for structured data
- **Multi-modal memory** integration

### Tool Integration
- **MCP (Model Context Protocol)** server
- **Generic tool server** with LangChain support
- **Web search**, **calculations**, **file analysis**
- **Email sending**, **weather data**, **custom tools**

### Workflow Automation
- **N8N integration** for complex workflows
- **Flowise** for visual AI flow design
- **Webhook triggers** and **scheduled tasks**
- **Event-driven** automation

## 📊 Service Details

### Core Services

| Service | Purpose | Port | Health Check |
|---------|---------|------|--------------|
| Traefik | API Gateway & Load Balancer | 80, 443, 8080 | `/api/overview` |
| Agentic Core | Main AI Agent Logic | 8000 | `/health` |
| Model Router | Intelligence Routing | 8000 | `/health` |
| Ollama | LLM Runtime | 11434 | `/api/tags` |

### Data Services

| Service | Purpose | Port | Health Check |
|---------|---------|------|--------------|
| Supabase DB | Primary Database (Postgres + Vector) | 5432 | Connection test |
| Neo4j | Knowledge Graph | 7474, 7687 | Browser UI |
| Qdrant | Vector Search Engine | 6333 | `/health` |

### Tool Services

| Service | Purpose | Port | Health Check |
|---------|---------|------|--------------|
| MCP Server | Model Context Protocol Tools | 8000 | `/health` |
| Generic Tool Server | LangChain & Custom Tools | 8001 | `/health` |

### UI Services

| Service | Purpose | Port | Access |
|---------|---------|------|--------|
| Open WebUI | Chat Interface | 3000 | `/webui` |
| N8N | Workflow Automation | 5678 | `/n8n` |
| Flowise | AI Flow Designer | 3000 | `/flowise` |

## 🔧 Configuration

### Environment Variables

Key variables in `.env`:

```bash
# Domain & Security
DOMAIN=localhost
JWT_SECRET=your-super-secret-jwt-secret

# Database Passwords
SUPABASE_DB_PASSWORD=secure_password
NEO4J_PASSWORD=secure_password

# AI Configuration
COMPLEXITY_THRESHOLD=0.7
PROACTIVE_MODE=true
THOUGHT_INTERVAL=300

# Model Selection
DEFAULT_LIGHT_MODEL=llama3.2:3b
DEFAULT_HEAVY_MODEL=llama3.1:8b
```

### Model Configuration

The system uses a **tiered model approach**:

1. **Ultra-Light**: `llama3.2:1b` - Simple responses
2. **Light**: `llama3.2:3b` - Standard queries  
3. **Medium**: `llama3.1:8b` - Complex reasoning
4. **Heavy**: `llama3.1:70b` - Advanced analysis (GPU required)

Configure in `model-router` service based on your hardware.

## 🛠️ Management Commands

### Service Management
```bash
make up              # Start all services
make down            # Stop all services  
make restart         # Restart all services
make logs            # View logs
make logs-follow     # Follow logs real-time
```

### Health & Monitoring
```bash
make health          # Check all service health
make status          # Show container status
make resources       # Show resource usage
make network-test    # Test internal connectivity
```

### Model Management
```bash
make models-list     # List downloaded models
make model-pull MODEL=llama3.1:70b  # Download specific model
make setup-models    # Download default models
```

### Development
```bash
make dev-logs        # Follow core service logs
make dev-shell-agent # Access agent container shell
make dev-shell-db    # Access database shell
```

### Backup & Restore
```bash
make backup          # Create full backup
make restore DATE=20241201_143022  # Restore from backup
```

## 🔍 Monitoring & Observability

### Grafana Dashboards
- **Request metrics** and **response times**
- **Model usage** and **complexity distribution** 
- **Database performance** and **connection pools**
- **Memory and CPU** utilization
- **Tool execution** success rates

### Prometheus Metrics
- Custom metrics for AI operations
- Service health monitoring
- Performance tracking
- Alert rules for critical issues

### Log Aggregation
- Centralized logging for all services
- Structured logs with correlation IDs
- Error tracking and debugging

## 🧪 Testing & Development

### Performance Testing
```bash
make test-performance    # Run performance tests
```

### API Testing
```bash
# Test model router
curl -X POST http://localhost/api/model/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "parameters": {}}'

# Test agentic core  
curl -X POST http://localhost/api/agent/process \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather like?", "user_id": "test_user"}'
```

### Adding Custom Tools
1. Create tool class in `tool-server/tools/`
2. Inherit from `BaseTool`
3. Implement `execute()` method
4. Register in `tool-server/main.py`

### LangChain Integration
1. Place LangChain code in `langchain-code/` directory
2. Create `tools.py` with `AVAILABLE_TOOLS` dict
3. Tools automatically loaded on startup

## 📈 Scaling & Production

### Horizontal Scaling
```bash
make scale-heavy     # Scale up for heavy workloads
make scale-light     # Scale down for light usage
```

### GPU Support
Uncomment GPU sections in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Production Deployment  
```bash
make prod-deploy     # Deploy with SSL and production configs
make ssl-setup       # Configure SSL certificates
```

## 🔐 Security Considerations

- **Change all default passwords** in `.env`
- **Use strong JWT secrets**
- **Configure SSL/TLS** for production
- **Network isolation** with Docker networks
- **Regular security updates**

## 🚨 Troubleshooting

### Common Issues

**Services won't start:**
```bash
make logs            # Check error messages
make clean && make up  # Clean restart
```

**Database connection issues:**
```bash
make dev-shell-db    # Test database access
make init-db         # Reinitialize database
```

**Model loading problems:**
```bash
docker-compose logs ollama  # Check Ollama logs
make models-list     # Verify models downloaded
```

**Out of memory:**
- Reduce model sizes in configuration
- Increase Docker memory limits
- Use lighter models for development

### Performance Optimization

- **Enable GPU** for heavy models
- **Adjust complexity thresholds** based on hardware
- **Scale services** based on load
- **Monitor resource usage** with Grafana

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes following the architecture
4. Test with `make test-performance`
5. Submit pull request

## 📝 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review logs with `make logs`
3. Open GitHub issue with logs and configuration
4. Join community discussions

---


**Built for the future of autonomous AI systems** 🤖✨



