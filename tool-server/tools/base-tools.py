# tool-server/tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import asyncio

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = "Base tool"
        self.parameters_schema = {}
    
    @abstractmethod
    async def execute(self, query: str, context: Dict[str, Any], parameters: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters"""
        pass

# tool-server/tools/web_search.py
import httpx
from .base_tool import BaseTool
from typing import Dict, Any
import json

class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.description = "Search the web for information"
        self.parameters_schema = {
            "query": {"type": "string", "description": "Search query"}
        }
    
    async def execute(self, query: str, context: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search"""
        search_query = parameters.get("query", query)
        
        try:
            # Using DuckDuckGo API as an example (no API key required)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": search_query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant information
                    results = []
                    if data.get("RelatedTopics"):
                        for topic in data["RelatedTopics"][:5]:
                            if isinstance(topic, dict) and "Text" in topic:
                                results.append({
                                    "text": topic["Text"],
                                    "url": topic.get("FirstURL", "")
                                })
                    
                    return {
                        "query": search_query,
                        "results": results,
                        "abstract": data.get("Abstract", ""),
                        "answer": data.get("Answer", "")
                    }
                else:
                    return {"error": f"Search failed with status {response.status_code}"}
                    
        except Exception as e:
            return {"error": str(e)}

# tool-server/tools/calculator.py
import sympy
from .base_tool import BaseTool
from typing import Dict, Any
import re

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.description = "Perform mathematical calculations"
        self.parameters_schema = {
            "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
        }
    
    async def execute(self, query: str, context: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mathematical calculation"""
        # Extract mathematical expression from query or parameters
        expression = parameters.get("expression", query)
        
        # Clean up the expression
        expression = self._extract_math_expression(expression)
        
        try:
            # Use sympy for safe mathematical evaluation
            result = sympy.sympify(expression)
            evaluated = float(result.evalf()) if result.is_real else str(result)
            
            return {
                "expression": expression,
                "result": evaluated,
                "formatted": f"{expression} = {evaluated}"
            }
            
        except Exception as e:
            return {"error": f"Mathematical evaluation failed: {str(e)}"}
    
    def _extract_math_expression(self, text: str) -> str:
        """Extract mathematical expression from text"""
        # Remove common words and keep mathematical symbols
        text = re.sub(r'\b(calculate|compute|what is|equals?|result of)\b', '', text.lower())
        text = re.sub(r'[^\d+\-*/().^√πe\s]', '', text)
        text = text.replace('√', 'sqrt').replace('π', 'pi').replace('^', '**')
        return text.strip()

# tool-server/tools/weather.py
import httpx
from .base_tool import BaseTool
from typing import Dict, Any
import os

class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.description = "Get weather information for a location"
        self.parameters_schema = {
            "location": {"type": "string", "description": "Location for weather data"}
        }
        self.api_key = os.getenv("OPENWEATHER_API_KEY")  # Optional API key
    
    async def execute(self, query: str, context: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get weather information"""
        location = parameters.get("location", self._extract_location(query))
        
        if not location:
            return {"error": "No location specified"}
        
        try:
            if self.api_key:
                return await self._get_weather_openweather(location)
            else:
                return await self._get_weather_free(location)
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_weather_openweather(self, location: str) -> Dict[str, Any]:
        """Get weather from OpenWeatherMap API"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "location": data["name"],
                    "temperature": data["main"]["temp"],
                    "description": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"]
                }
            else:
                return {"error": "Weather API request failed"}
    
    async def _get_weather_free(self, location: str) -> Dict[str, Any]:
        """Get weather from free service (wttr.in)"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://wttr.in/{location}?format=j1",
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                current = data["current_condition"][0]
                
                return {
                    "location": location,
                    "temperature": current["temp_C"],
                    "description": current["weatherDesc"][0]["value"],
                    "humidity": current["humidity"],
                    "pressure": current["pressure"]
                }
            else:
                return {"error": "Weather service unavailable"}
    
    def _extract_location(self, text: str) -> str:
        """Extract location from query text"""
        # Simple location extraction
        words = text.lower().split()
        weather_words = ["weather", "temperature", "forecast", "in", "at", "for"]
        
        # Find words after weather-related terms
        for i, word in enumerate(words):
            if word in weather_words and i + 1 < len(words):
                return " ".join(words[i + 1:])
        
        return ""

# tool-server/tools/email_sender.py
from .base_tool import BaseTool
from typing import Dict, Any
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailSenderTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.description = "Send emails"
        self.parameters_schema = {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"}
        }
        
        # Email configuration from environment
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")
    
    async def execute(self, query: str, context: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send email"""
        if not self.sender_email or not self.sender_password:
            return {"error": "Email credentials not configured"}
        
        to_email = parameters.get("to")
        subject = parameters.get("subject", "Message from AI Agent")
        body = parameters.get("body", query)
        
        if not to_email:
            return {"error": "Recipient email address required"}
        
        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = to_email
            message["Subject"] = subject
            
            message.attach(MIMEText(body, "plain"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            return {
                "status": "sent",
                "to": to_email,
                "subject": subject
            }
            
        except Exception as e:
            return {"error": f"Email sending failed: {str(e)}"}

# tool-server/tools/file_analyzer.py
import aiofiles
import pandas as pd
from .base_tool import BaseTool
from typing import Dict, Any
import os
import mimetypes
from pathlib import Path

class FileAnalyzerTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.description = "Analyze uploaded files"
        self.parameters_schema = {
            "file_path": {"type": "string", "description": "Path to file to analyze"}
        }
    
    async def execute(self, query: str, context: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze file content"""
        file_path = parameters.get("file_path")
        
        if not file_path:
            return {"error": "File path required"}
        
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        try:
            file_info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "type": mimetypes.guess_type(file_path)[0]
            }
            
            # Determine file type and analyze accordingly
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension in ['.csv']:
                analysis = await self._analyze_csv(file_path)
            elif file_extension in ['.txt', '.md']:
                analysis = await self._analyze_text(file_path)
            elif file_extension in ['.json']:
                analysis = await self._analyze_json(file_path)
            else:
                analysis = {"type": "unknown", "message": "Unsupported file type for analysis"}
            
            return {
                "file_info": file_info,
                "analysis": analysis
            }
            
        except Exception as e:
            return {"error": f"File analysis failed: {str(e)}"}
    
    async def _analyze_csv(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            return {
                "type": "csv",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head().to_dict('records')
            }
        except Exception as e:
            return {"type": "csv", "error": str(e)}
    
    async def _analyze_text(self, file_path: str) -> Dict[str, Any]:
        """Analyze text file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
                
            return {
                "type": "text",
                "character_count": len(content),
                "word_count": len(content.split()),
                "line_count": len(content.splitlines()),
                "preview": content[:500] + "..." if len(content) > 500 else content
            }
        except Exception as e:
            return {"type": "text", "error": str(e)}
    
    async def _analyze_json(self, file_path: str) -> Dict[str, Any]:
        """Analyze JSON file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
                
            import json
            data = json.loads(content)
            
            return {
                "type": "json",
                "structure": self._get_json_structure(data),
                "size": len(content),
                "keys": list(data.keys()) if isinstance(data, dict) else "Not a JSON object"
            }
        except Exception as e:
            return {"type": "json", "error": str(e)}
    
    def _get_json_structure(self, obj, max_depth=3, current_depth=0):
        """Get JSON structure recursively"""
        if current_depth >= max_depth:
            return "..."
        
        if isinstance(obj, dict):
            return {k: self._get_json_structure(v, max_depth, current_depth + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._get_json_structure(obj[0], max_depth, current_depth + 1)] if obj else []
        else:
            return type(obj).__name__