from fastapi import FastAPI
import os
import importlib.util
from typing import Dict, Any

app = FastAPI(title="Generic Tool Server", version="1.0.0")

class ToolServer:
    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        self.tools = self.load_custom_tools()
    
    def load_custom_tools(self):
        """Load custom tools from langchain_tools directory"""
        tools = {}
        tools_dir = "/app/langchain_
