#!/usr/bin/env python3
"""
Configuration management for Ollama API Shim
"""

import os
from typing import Optional

class Config:
    """Configuration settings with environment variable overrides"""
    
    # Service endpoints
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "postgresql://supabase_admin:your_password@supabase-db:5432/postgres")
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j-stack:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "your_password")
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
    
    # AI Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    MAX_MEMORY_ENTRIES: int = int(os.getenv("MAX_MEMORY_ENTRIES", "50"))
    
    # Feature flags
    ENABLE_KNOWLEDGE_GRAPH: bool = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
    ENABLE_VECTOR_SEARCH: bool = os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"
    ENABLE_MEMORY: bool = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
    ENABLE_TOOLS: bool = os.getenv("ENABLE_TOOLS", "true").lower() == "true"
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Cache and performance
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/app/cache")
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    
    # Queue configuration
    MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))
    QUEUE_TIMEOUT: int = int(os.getenv("QUEUE_TIMEOUT", "300"))  # 5 minutes
    
    # Resource monitoring thresholds
    CPU_THRESHOLD: float = float(os.getenv("CPU_THRESHOLD", "80.0"))
    MEMORY_THRESHOLD: float = float(os.getenv("MEMORY_THRESHOLD", "85.0"))
    RESPONSE_TIME_THRESHOLD: float = float(os.getenv("RESPONSE_TIME_THRESHOLD", "30.0"))
    
    # Database pool settings
    DB_POOL_MIN_SIZE: int = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
    DB_POOL_MAX_SIZE: int = int(os.getenv("DB_POOL_MAX_SIZE", "20"))
    
    # Model routing thresholds
    COMPLEXITY_THRESHOLD: float = float(os.getenv("COMPLEXITY_THRESHOLD", "0.7"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration"""
        required_vars = [
            cls.OLLAMA_BASE_URL,
            cls.SUPABASE_DB_URL,
            cls.NEO4J_URI,
            cls.NEO4J_PASSWORD
        ]
        
        return all(var and var != "your_password" for var in required_vars)
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without secrets)"""
        print("=== Ollama API Shim Configuration ===")
        print(f"Ollama URL: {cls.OLLAMA_BASE_URL}")
        print(f"Weaviate URL: {cls.WEAVIATE_URL}")
        print(f"Neo4j URI: {cls.NEO4J_URI}")
        print(f"MCP Server URL: {cls.MCP_SERVER_URL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Max Context Tokens: {cls.MAX_CONTEXT_TOKENS}")
        print(f"Similarity Threshold: {cls.SIMILARITY_THRESHOLD}")
        print("Performance Settings:")
        print(f"  Max Concurrent Requests: {cls.MAX_CONCURRENT_REQUESTS}")
        print(f"  Max Queue Size: {cls.MAX_QUEUE_SIZE}")
        print(f"  CPU Threshold: {cls.CPU_THRESHOLD}%")
        print(f"  Memory Threshold: {cls.MEMORY_THRESHOLD}%")
        print("Feature Flags:")
        print(f"  Knowledge Graph: {cls.ENABLE_KNOWLEDGE_GRAPH}")
        print(f"  Vector Search: {cls.ENABLE_VECTOR_SEARCH}")
        print(f"  Memory: {cls.ENABLE_MEMORY}")
        print(f"  Tools: {cls.ENABLE_TOOLS}")
        print(f"  Caching: {cls.ENABLE_CACHING}")
        print("=====================================")

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    LOG_LEVEL: str = "DEBUG"
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 5
    MAX_CONCURRENT_REQUESTS: int = 3
    MAX_QUEUE_SIZE: int = 20

class ProductionConfig(Config):
    """Production environment configuration"""
    LOG_LEVEL: str = "INFO"
    DB_POOL_MIN_SIZE: int = 10
    DB_POOL_MAX_SIZE: int = 50
    MAX_CONCURRENT_REQUESTS: int = 20
    MAX_QUEUE_SIZE: int = 500

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()