# Makefile for Agentic AI Docker Stack
.PHONY: help build up down logs clean init-db setup-models health status

# Default target
help:
	@echo "Agentic AI Docker Stack Management"
	@echo "=================================="
	@echo "Available commands:"
	@echo "  setup          - Initial setup of the entire stack"
	@echo "  build          - Build all custom Docker images"
	@echo "  up             - Start all services"
	@echo "  down           - Stop all services"
	@echo "  restart        - Restart all services"
	@echo "  logs           - Show logs for all services"
	@echo "  logs-follow    - Follow logs in real-time"
	@echo "  clean          - Clean up volumes and networks"
	@echo "  init-db        - Initialize databases"
	@echo "  setup-models   - Download and setup AI models"
	@echo "  health         - Check health of all services"
	@echo "  status         - Show status of all services"
	@echo "  scale          - Scale services up/down"
	@echo "  backup         - Backup databases"
	@echo "  restore        - Restore from backup"

# Environment setup
.env:
	@echo "Creating .env file from template..."
	@cp .env.template .env
	@echo "Please edit .env file with your configuration"

# Initial setup
setup: .env
	@echo "Setting up Agentic AI Stack..."
	@mkdir -p data/supabase data/neo4j data/ollama data/n8n data/flowise
	@mkdir -p logs monitoring/grafana/dashboards monitoring/grafana/datasources
	@chmod -R 755 data logs
	@make build
	@make init-db
	@make setup-models
	@echo "Setup complete! Run 'make up' to start the stack"

# Build all custom images
build:
	@echo "Building custom Docker images..."
	docker-compose build --parallel

# Start services
up:
	@echo "Starting Agentic AI Stack..."
	docker-compose up -d
	@echo "Services started. Run 'make health' to check status"

# Stop services
down:
	@echo "Stopping Agentic AI Stack..."
	docker-compose down

# Restart services
restart:
	@make down
	@make up

# View logs
logs:
	docker-compose logs

# Follow logs in real-time
logs-follow:
	docker-compose logs -f

# Clean up everything
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "Cleanup complete"

# Deep clean (removes volumes too)
clean-all: clean
	@echo "Removing all data volumes..."
	docker volume rm $(shell docker volume ls -q | grep agentic) 2>/dev/null || true
	@echo "Deep cleanup complete"

# Initialize databases
init-db:
	@echo "Initializing databases..."
	@echo "Waiting for database to be ready..."
	@sleep 10
	docker-compose exec -T supabase-db psql -U supabase_admin -d postgres -f /docker-entrypoint-initdb.d/01_create_databases.sql || true
	docker-compose exec -T supabase-db psql -U supabase_admin -d postgres -f /docker-entrypoint-initdb.d/02_enable_extensions.sql || true
	docker-compose exec -T supabase-db psql -U supabase_admin -d postgres -f /docker-entrypoint-initdb.d/03_create_agent_schema.sql || true
	@echo "Database initialization complete"

# Setup AI models
setup-models:
	@echo "Setting up AI models..."
	@echo "Pulling lightweight models..."
	docker-compose exec ollama ollama pull llama3.2:3b || echo "Ollama not ready yet"
	@echo "Pulling heavier models..."
	docker-compose exec ollama ollama pull llama3.1:8b || echo "Ollama not ready yet"
	@echo "Pulling embedding model..."
	docker-compose exec ollama ollama pull nomic-embed-text || echo "Ollama not ready yet"
	@echo "Models setup complete"

# Health check for all services
health:
	@echo "Checking service health..."
	@echo "=================================="
	@curl -s http://localhost:8080/api/overview 2>/dev/null && echo "✓ Traefik: Healthy" || echo "✗ Traefik: Down"
	@curl -s http://localhost:5432 2>/dev/null && echo "✓ Supabase DB: Healthy" || echo "✗ Supabase DB: Down"
	@curl -s http://localhost:7474 2>/dev/null && echo "✓ Neo4j: Healthy" || echo "✗ Neo4j: Down"
	@curl -s http://localhost:11434/api/tags 2>/dev/null && echo "✓ Ollama: Healthy" || echo "✗ Ollama: Down"
	@curl -s http://localhost/n8n 2>/dev/null && echo "✓ N8N: Healthy" || echo "✗ N8N: Down"
	@curl -s http://localhost/flowise 2>/dev/null && echo "✓ Flowise: Healthy" || echo "✗ Flowise: Down"
	@curl -s http://localhost/webui 2>/dev/null && echo "✓ Open WebUI: Healthy" || echo "✗ Open WebUI: Down"
	@echo "=================================="

# Show service status
status:
	@echo "Service Status:"
	@echo "==============="
	docker-compose ps

# Scale services
scale-light:
	@echo "Scaling to light configuration..."
	docker-compose up -d --scale agentic-core=1 --scale model-router=1

scale-heavy:
	@echo "Scaling to heavy configuration..."
	docker-compose up -d --scale agentic-core=2 --scale model-router=2

# Backup databases
backup:
	@echo "Creating backups..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	docker-compose exec -T supabase-db pg_dump -U supabase_admin -d agent > backups/$(shell date +%Y%m%d_%H%M%S)/agent_db.sql
	docker-compose exec -T supabase-db pg_dump -U supabase_admin -d tools > backups/$(shell date +%Y%m%d_%H%M%S)/tools_db.sql
	docker-compose exec -T supabase-db pg_dump -U supabase_admin -d mcp > backups/$(shell date +%Y%m%d_%H%M%S)/mcp_db.sql
	docker-compose exec neo4j neo4j-admin database dump --to-path=/backups neo4j > backups/$(shell date +%Y%m%d_%H%M%S)/neo4j.dump
	@echo "Backups created in backups/ directory"

# Restore from backup
restore:
	@echo "Available backups:"
	@ls -la backups/
	@echo "Please specify backup date: make restore-from DATE=YYYYMMDD_HHMMSS"

restore-from:
	@echo "Restoring from backup $(DATE)..."
	docker-compose exec -T supabase-db psql -U supabase_admin -d agent < backups/$(DATE)/agent_db.sql
	docker-compose exec -T supabase-db psql -U supabase_admin -d tools < backups/$(DATE)/tools_db.sql
	docker-compose exec -T supabase-db psql -U supabase_admin -d mcp < backups/$(DATE)/mcp_db.sql
	@echo "Database restore complete"

# Development helpers
dev-logs:
	docker-compose logs -f agentic-core model-router

dev-shell-agent:
	docker-compose exec agentic-core /bin/bash

dev-shell-db:
	docker-compose exec supabase-db psql -U supabase_admin -d agent

dev-shell-neo4j:
	docker-compose exec neo4j cypher-shell -u neo4j -p $(NEO4J_PASSWORD)

# Model management
models-list:
	docker-compose exec ollama ollama list

models-pull:
	@echo "Available models to pull:"
	@echo "  llama3.2:1b   - Ultra lightweight"
	@echo "  llama3.2:3b   - Lightweight (default)"
	@echo "  llama3.1:8b   - Medium (default heavy)"
	@echo "  llama3.1:70b  - Heavy (requires GPU)"
	@echo "  codellama:7b  - Code specialized"
	@echo "  mistral:7b    - Alternative model"
	@echo ""
	@echo "Usage: make model-pull MODEL=llama3.1:70b"

model-pull:
	docker-compose exec ollama ollama pull $(MODEL)

# Monitoring
monitoring-up:
	@echo "Starting monitoring stack..."
	docker-compose up -d prometheus grafana
	@echo "Grafana available at http://localhost/grafana (admin/admin)"

monitoring-down:
	docker-compose stop prometheus grafana

# Update stack
update:
	@echo "Updating Docker images..."
	docker-compose pull
	@make build
	@echo "Update complete. Run 'make restart' to apply changes"

# Performance testing
test-performance:
	@echo "Running performance tests..."
	@echo "Testing model router..."
	@curl -X POST http://localhost/api/model/route \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello, this is a simple test", "parameters": {}}' \
		2>/dev/null | jq . || echo "Model router not responding"
	@echo "Testing agentic core..."
	@curl -X POST http://localhost/api/agent/process \
		-H "Content-Type: application/json" \
		-d '{"query": "What is the weather like?", "user_id": "test_user"}' \
		2>/dev/null | jq . || echo "Agentic core not responding"

# SSL Setup (for production)
ssl-setup:
	@echo "Setting up SSL certificates..."
	@mkdir -p ssl
	@echo "Please place your SSL certificates in the ssl/ directory"
	@echo "Required files: ssl/cert.pem, ssl/key.pem"

# Production deployment
prod-deploy: ssl-setup
	@echo "Deploying to production..."
	@cp docker-compose.yml docker-compose.prod.yml
	@echo "Updating production configuration..."
	@sed -i 's/traefik:v2.10/traefik:v2.10/g' docker-compose.prod.yml
	@docker-compose -f docker-compose.prod.yml up -d
	@echo "Production deployment complete"

# Quick start for development
quick-start:
	@echo "Quick starting development environment..."
	@make setup
	@make up
	@echo "Waiting for services to be ready..."
	@sleep 30
	@make setup-models
	@make health
	@echo ""
	@echo "Quick start complete!"
	@echo "Services available at:"
	@echo "  Web UI: http://localhost/webui"
	@echo "  N8N: http://localhost/n8n"
	@echo "  Flowise: http://localhost/flowise"
	@echo "  Neo4j: http://localhost:7474"
	@echo "  Grafana: http://localhost/grafana"

# Resource monitoring
resources:
	@echo "Docker Resource Usage:"
	@echo "======================"
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Network diagnostics
network-test:
	@echo "Testing internal network connectivity..."
	docker-compose exec agentic-core curl -s http://model-router:8000/health || echo "❌ Agent -> Model Router"
	docker-compose exec model-router curl -s http://ollama:11434/api/tags || echo "❌ Model Router -> Ollama"
	docker-compose exec agentic-core curl -s http://supabase-db:5432 || echo "❌ Agent -> Database"
	docker-compose exec agentic-core curl -s http://neo4j:7687 || echo "❌ Agent -> Neo4j"
	@echo "Network test complete"