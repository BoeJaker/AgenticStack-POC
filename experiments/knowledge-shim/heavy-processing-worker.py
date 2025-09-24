"""
Heavy Processing Worker - Tier 1 & 2 Classification and Deep Analysis
Processes queued jobs for advanced knowledge graph extraction
"""

import ast
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aioredis
import httpx
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
import tree_sitter
from tree_sitter import Language, Parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
WORKER_NAME = os.getenv("WORKER_NAME", "heavy-worker-1")
PROCESSING_MODEL = os.getenv("PROCESSING_MODEL", "llama2")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))

class Tier1StructuredExtractor:
    """Tier 1: Structured LLM-based extraction with JSON output"""
    
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url
        self.model = model
    
    async def extract_structured_knowledge(self, prompt_text: str, response_text: str) -> Dict:
        """Use LLM to extract structured knowledge with strict JSON schema"""
        
        extraction_prompt = self._build_extraction_prompt(prompt_text, response_text)
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": self.model,
                    "prompt": extraction_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistency
                        "top_p": 0.9,
                        "seed": 42  # Reproducible results
                    }
                }
                
                response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Parse the JSON response
                extracted_json = json.loads(result.get("response", "{}"))
                return self._validate_and_clean_extraction(extracted_json)
                
        except Exception as e:
            logger.error(f"Tier 1 extraction failed: {e}")
            return {"error": str(e), "confidence": 0.0}
    
    def _build_extraction_prompt(self, prompt_text: str, response_text: str) -> str:
        """Build structured extraction prompt with examples"""
        return f"""You are a knowledge extraction system. Extract structured information from the conversation below and return ONLY valid JSON.

REQUIRED JSON SCHEMA:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "PERSON|ORGANIZATION|LOCATION|TECHNOLOGY|CONCEPT|PRODUCT",
      "description": "brief description",
      "confidence": 0.0-1.0,
      "context": "surrounding text context",
      "canonical_form": "normalized name",
      "aliases": ["alternative names"]
    }}
  ],
  "relationships": [
    {{
      "subject": "entity name",
      "predicate": "relationship type", 
      "object": "entity name",
      "type": "SEMANTIC|TEMPORAL|CAUSAL|HIERARCHICAL|FUNCTIONAL",
      "confidence": 0.0-1.0,
      "evidence": "supporting text",
      "direction": "BIDIRECTIONAL|FORWARD|REVERSE"
    }}
  ],
  "temporal_events": [
    {{
      "event": "what happened",
      "timestamp": "when (if extractable)",
      "entities_involved": ["entity names"],
      "event_type": "CREATION|MODIFICATION|DELETION|COMMUNICATION|DECISION",
      "confidence": 0.0-1.0
    }}
  ],
  "code_structures": [
    {{
      "name": "function/class/variable name",
      "type": "FUNCTION|CLASS|VARIABLE|MODULE|IMPORT",
      "language": "programming language",
      "signature": "function signature or declaration",
      "docstring": "documentation if present",
      "dependencies": ["what it depends on"],
      "purpose": "what it does",
      "confidence": 0.0-1.0
    }}
  ],
  "topics": [
    {{
      "topic": "main topic or theme",
      "category": "TECHNICAL|BUSINESS|PERSONAL|EDUCATIONAL|DEBUG",
      "subtopics": ["related subtopics"],
      "confidence": 0.0-1.0
    }}
  ],
  "intentions": [
    {{
      "intent": "what the user wants",
      "type": "QUESTION|REQUEST|COMMAND|INFORMATION|HELP|CREATION",
      "confidence": 0.0-1.0,
      "fulfilled": true/false
    }}
  ],
  "metadata": {{
    "processing_complexity": "LOW|MEDIUM|HIGH",
    "domain": "primary domain or field",
    "language_detected": "primary language",
    "requires_web_search": true/false,
    "contains_sensitive_info": true/false,
    "overall_confidence": 0.0-1.0
  }}
}}

CONVERSATION TO ANALYZE:
USER: {prompt_text[:2000]}

ASSISTANT: {response_text[:2000]}

Return ONLY the JSON object with extracted information. Be precise and conservative with confidence scores."""

    def _validate_and_clean_extraction(self, extracted_data: Dict) -> Dict:
        """Validate and clean the extracted JSON data"""
        
        # Provide defaults for required fields
        cleaned = {
            "entities": extracted_data.get("entities", []),
            "relationships": extracted_data.get("relationships", []),
            "temporal_events": extracted_data.get("temporal_events", []),
            "code_structures": extracted_data.get("code_structures", []),
            "topics": extracted_data.get("topics", []),
            "intentions": extracted_data.get("intentions", []),
            "metadata": extracted_data.get("metadata", {}),
            "processing_tier": "tier_1_structured",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Clean and validate entities
        for entity in cleaned["entities"]:
            entity["confidence"] = max(0.0, min(1.0, entity.get("confidence", 0.5)))
            entity["type"] = entity.get("type", "CONCEPT").upper()
            if not entity.get("canonical_form"):
                entity["canonical_form"] = entity.get("name", "").lower().strip()
        
        # Clean and validate relationships
        for rel in cleaned["relationships"]:
            rel["confidence"] = max(0.0, min(1.0, rel.get("confidence", 0.5)))
            rel["type"] = rel.get("type", "SEMANTIC").upper()
            rel["direction"] = rel.get("direction", "FORWARD").upper()
        
        # Set overall confidence based on individual confidences
        all_confidences = []
        for item_list in [cleaned["entities"], cleaned["relationships"], cleaned["temporal_events"]]:
            all_confidences.extend([item.get("confidence", 0.5) for item in item_list])
        
        if all_confidences:
            cleaned["metadata"]["overall_confidence"] = sum(all_confidences) / len(all_confidences)
        else:
            cleaned["metadata"]["overall_confidence"] = 0.3
        
        return cleaned

class WebCrawler:
    """Web crawler for processing referenced URLs"""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "KnowledgeGraph-Bot/1.0 (+https://example.com/bot)"
            },
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def crawl_url(self, url: str) -> Dict:
        """Crawl a single URL and extract structured information"""
        
        try:
            # Respect delay
            await asyncio.sleep(self.delay)
            
            response = await self.session.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            
            if "text/html" in content_type:
                return await self._parse_html(url, response.text)
            elif any(ct in content_type for ct in ["application/json", "text/json"]):
                return await self._parse_json(url, response.text)
            elif "text/plain" in content_type:
                return await self._parse_text(url, response.text)
            else:
                return {
                    "url": url,
                    "status": "unsupported_content_type",
                    "content_type": content_type,
                    "size": len(response.text)
                }
                
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            return {
                "url": url,
                "status": "error",
                "error": str(e),
                "crawled_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def _parse_html(self, url: str, html_content: str) -> Dict:
        """Parse HTML content and extract structured data"""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Extract Open Graph and meta tags
        og_data = {}
        for meta in soup.find_all('meta', property=re.compile(r'^og:')):
            property_name = meta.get('property', '').replace('og:', '')
            og_data[property_name] = meta.get('content', '')
        
        # Extract main content
        main_content = ""
        content_selectors = ['main', 'article', '.content', '#content', '.post-content']
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text(separator=' ', strip=True)
                break
        
        if not main_content:
            main_content = soup.get_text(separator=' ', strip=True)
        
        # Clean and limit content
        main_content = re.sub(r'\s+', ' ', main_content)[:5000]
        
        # Extract links
        internal_links = []
        external_links = []
        base_domain = urlparse(url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            link_domain = urlparse(absolute_url).netloc
            
            link_data = {
                "url": absolute_url,
                "text": link.get_text().strip(),
                "title": link.get('title', '')
            }
            
            if link_domain == base_domain:
                internal_links.append(link_data)
            else:
                external_links.append(link_data)
        
        return {
            "url": url,
            "status": "success",
            "title": title_text,
            "content": main_content,
            "content_length": len(main_content),
            "opengraph": og_data,
            "internal_links": internal_links[:20],  # Limit links
            "external_links": external_links[:10],
            "images": [img.get('src', '') for img in soup.find_all('img', src=True)[:10]],
            "headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])[:10]],
            "domain": base_domain,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "content_type": "text/html"
        }
    
    async def _parse_json(self, url: str, json_content: str) -> Dict:
        """Parse JSON content"""
        try:
            data = json.loads(json_content)
            return {
                "url": url,
                "status": "success",
                "content": json_content[:2000],  # Limit size
                "structured_data": data if isinstance(data, dict) else {"data": data},
                "content_type": "application/json",
                "crawled_at": datetime.now(timezone.utc).isoformat()
            }
        except json.JSONDecodeError as e:
            return {
                "url": url,
                "status": "json_parse_error",
                "error": str(e),
                "content": json_content[:500]
            }
    
    async def _parse_text(self, url: str, text_content: str) -> Dict:
        """Parse plain text content"""
        return {
            "url": url,
            "status": "success",
            "content": text_content[:5000],  # Limit size
            "content_length": len(text_content),
            "content_type": "text/plain",
            "crawled_at": datetime.now(timezone.utc).isoformat()
        }

class CodeAnalyzer:
    """Advanced code analysis using AST and tree-sitter"""
    
    def __init__(self):
        self.python_parser = None
        self._setup_parsers()
    
    def _setup_parsers(self):
        """Setup tree-sitter parsers if available"""
        try:
            # This would require tree-sitter-python to be built
            # For now, we'll use Python's built-in AST
            pass
        except Exception as e:
            logger.warning(f"Tree-sitter setup failed: {e}")
    
    def analyze_python_code(self, code: str, filename: str = "<string>") -> Dict:
        """Analyze Python code using AST"""
        try:
            tree = ast.parse(code, filename=filename)
            analyzer = PythonASTAnalyzer()
            return analyzer.visit(tree)
        except SyntaxError as e:
            return {
                "error": "syntax_error",
                "message": str(e),
                "line": e.lineno,
                "offset": e.offset
            }
        except Exception as e:
            return {
                "error": "analysis_error",
                "message": str(e)
            }
    
    def detect_language(self, code: str) -> str:
        """Detect programming language from code content"""
        # Simple heuristic-based detection
        if any(keyword in code for keyword in ['def ', 'import ', 'from ', 'class ', '__init__']):
            return "python"
        elif any(keyword in code for keyword in ['function ', 'const ', 'let ', 'var ', '=>']):
            return "javascript"
        elif any(keyword in code for keyword in ['public class', 'private ', 'public static void main']):
            return "java"
        elif any(keyword in code for keyword in ['#include', 'int main(', 'printf(', 'cout <<']):
            return "c/cpp"
        elif any(keyword in code for keyword in ['SELECT ', 'FROM ', 'WHERE ', 'INSERT INTO']):
            return "sql"
        else:
            return "unknown"

class PythonASTAnalyzer(ast.NodeVisitor):
    """Python AST analyzer to extract code structures"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.variables = []
        self.calls = []
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        func_info = {
            "name": node.name,
            "type": "method" if self.current_class else "function",
            "line_start": node.lineno,
            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            "args": [arg.arg for arg in node.args.args],
            "decorators": [ast.unparse(dec) if hasattr(ast, 'unparse') else "decorator" for dec in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "parent_class": self.current_class,
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }
        self.functions.append(func_info)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        class_info = {
            "name": node.name,
            "type": "class",
            "line_start": node.lineno,
            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            "bases": [ast.unparse(base) if hasattr(ast, 'unparse') else "base" for base in node.bases],
            "decorators": [ast.unparse(dec) if hasattr(ast, 'unparse') else "decorator" for dec in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "methods": []
        }
        
        old_class = self.current_class
        self.current_class = node.name
        self.classes.append(class_info)
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "type": "import",
                "module": alias.name,
                "alias": alias.asname,
                "line": node.lineno
            })
    
    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append({
                "type": "from_import",
                "module": node.module,
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "level": node.level
            })
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.append({
                    "name": target.id,
                    "type": "variable",
                    "line": node.lineno,
                    "scope": self.current_class or "module"
                })
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append({
                "function": node.func.id,
                "line": node.lineno,
                "args_count": len(node.args)
            })
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            self.calls.append({
                "function": attr_name,
                "line": node.lineno,
                "args_count": len(node.args),
                "is_method": True
            })
        self.generic_visit(node)
    
    def get_results(self):
        return {
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "variables": self.variables,
            "function_calls": self.calls,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }

class HeavyProcessingWorker:
    """Main worker class for heavy processing tasks"""
    
    def __init__(self):
        self.redis_client = None
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.tier1_extractor = Tier1StructuredExtractor(OLLAMA_URL, PROCESSING_MODEL)
        self.code_analyzer = CodeAnalyzer()
        self.running = True
    
    async def start(self):
        """Start the worker"""
        self.redis_client = await aioredis.from_url(REDIS_URL)
        logger.info(f"Heavy processing worker {WORKER_NAME} started")
        
        while self.running:
            try:
                # Blocking pop from Redis queue
                job_data = await self.redis_client.brpop("heavy_processing_queue", timeout=5)
                
                if job_data:
                    queue_name, job_json = job_data
                    job = json.loads(job_json)
                    await self.process_job(job)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
    
    async def process_job(self, job: Dict):
        """Process a single heavy processing job"""
        session_id = job.get("session_id")
        logger.info(f"Processing heavy job for session {session_id}")
        
        try:
            # Tier 1: Structured extraction
            tier1_results = await self.tier1_extractor.extract_structured_knowledge(
                job.get("prompt_text", ""),
                job.get("response_text", "")
            )
            
            # Ingest Tier 1 results
            await self.ingest_tier1_results(session_id, tier1_results)
            
            # Web crawling if URLs detected
            urls = self.extract_urls_from_job(job)
            if urls:
                await self.process_urls(session_id, urls)
            
            # Code analysis if code detected
            code_blocks = self.extract_code_from_job(job, tier1_results)
            if code_blocks:
                await self.process_code_blocks(session_id, code_blocks)
            
            # Enqueue Tier 2 processing if confidence is low or complexity is high
            if (tier1_results.get("metadata", {}).get("overall_confidence", 1.0) < 0.7 or
                tier1_results.get("metadata", {}).get("processing_complexity") == "HIGH"):
                await self.enqueue_tier2_processing(session_id, job, tier1_results)
            
            logger.info(f"Completed heavy processing for session {session_id}")
            
        except Exception as e:
            logger.error(f"Heavy processing failed for session {session_id}: {e}")
            await self.record_processing_error(session_id, str(e))
    
    async def ingest_tier1_results(self, session_id: str, results: Dict):
        """Ingest Tier 1 structured results into Neo4j"""
        
        def ingest_transaction(tx):
            # Update processing job
            tx.run("""
                MATCH (j:ProcessingJob {session_id: $session_id})
                SET j.tier1_completed = true,
                    j.tier1_timestamp = $timestamp,
                    j.tier1_confidence = $confidence,
                    j.tier1_entities_count = $entities_count,
                    j.tier1_relationships_count = $relationships_count
            """,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                confidence=results.get("metadata", {}).get("overall_confidence", 0.0),
                entities_count=len(results.get("entities", [])),
                relationships_count=len(results.get("relationships", []))
            )
            
            # Batch insert enhanced entities
            if results.get("entities"):
                entities_data = []
                for entity in results["entities"]:
                    entities_data.append({
                        "name": entity["name"],
                        "canonical_form": entity.get("canonical_form", entity["name"].lower()),
                        "type": entity.get("type", "CONCEPT"),
                        "description": entity.get("description", ""),
                        "confidence": entity.get("confidence", 0.5),
                        "context": entity.get("context", ""),
                        "aliases": entity.get("aliases", []),
                        "session_id": session_id,
                        "processing_tier": "tier_1",
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    })
                
                tx.run("""
                    UNWIND $entities_data AS entity
                    MATCH (p:Prompt {session_id: entity.session_id})
                    MERGE (e:Entity {canonical_form: entity.canonical_form})
                    ON CREATE SET e += {
                        name: entity.name,
                        type: entity.type,
                        first_seen: entity.updated_at,
                        occurrence_count: 1
                    }
                    ON MATCH SET e.occurrence_count = coalesce(e.occurrence_count, 0) + 1
                    SET e += {
                        description: entity.description,
                        tier1_confidence: entity.confidence,
                        context: entity.context,
                        aliases: entity.aliases,
                        processing_tier: entity.processing_tier,
                        updated_at: entity.updated_at
                    }
                    MERGE (p)-[:MENTIONS_TIER1 {
                        confidence: entity.confidence,
                        context: entity.context,
                        created_at: entity.updated_at
                    }]->(e)
                """, entities_data=entities_data)
            
            # Batch insert relationships
            if results.get("relationships"):
                relationships_data = []
                for rel in results["relationships"]:
                    relationships_data.append({
                        "subject": rel["subject"],
                        "predicate": rel["predicate"],
                        "object": rel["object"],
                        "type": rel.get("type", "SEMANTIC"),
                        "confidence": rel.get("confidence", 0.5),
                        "evidence": rel.get("evidence", ""),
                        "direction": rel.get("direction", "FORWARD"),
                        "session_id": session_id,
                        "created_at": datetime.now(timezone.utc).isoformat()
                    })
                
                tx.run("""
                    UNWIND $relationships_data AS rel
                    MATCH (subj:Entity) WHERE subj.name = rel.subject OR rel.subject IN subj.aliases
                    MATCH (obj:Entity) WHERE obj.name = rel.object OR rel.object IN obj.aliases
                    MERGE (subj)-[r:RELATED_TO_TIER1 {
                        predicate: rel.predicate,
                        type: rel.type,
                        session_id: rel.session_id
                    }]->(obj)
                    SET r += {
                        confidence: rel.confidence,
                        evidence: rel.evidence,
                        direction: rel.direction,
                        created_at: rel.created_at,
                        updated_at: rel.created_at
                    }
                """, relationships_data=relationships_data)
        
        # Execute transaction
        with self.neo4j_driver.session() as session:
            session.write_transaction(ingest_transaction)
    
    def extract_urls_from_job(self, job: Dict) -> List[str]:
        """Extract URLs that need crawling"""
        urls = []
        text = f"{job.get('prompt_text', '')} {job.get('response_text', '')}"
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        found_urls = re.findall(url_pattern, text)
        
        # Filter out common non-content URLs
        filtered_urls = []
        skip_patterns = [
            r'\.(?:jpg|jpeg|png|gif|css|js|ico|pdf|zip|exe),
            r'(?:facebook|twitter|instagram|linkedin)\.com',
            r'(?:ads|analytics|tracking)',
        ]
        
        for url in found_urls:
            if not any(re.search(pattern, url, re.IGNORECASE) for pattern in skip_patterns):
                filtered_urls.append(url)
        
        return filtered_urls[:5]  # Limit to 5 URLs per job
    
    def extract_code_from_job(self, job: Dict, tier1_results: Dict) -> List[Dict]:
        """Extract code blocks for analysis"""
        code_blocks = []
        
        # From tier1 results
        for code_struct in tier1_results.get("code_structures", []):
            code_blocks.append({
                "code": code_struct.get("signature", "") + "\n" + code_struct.get("docstring", ""),
                "language": code_struct.get("language", "unknown"),
                "type": code_struct.get("type", "UNKNOWN"),
                "name": code_struct.get("name", "unnamed"),
                "source": "tier1_extraction"
            })
        
        # Extract from original text using regex
        text = f"{job.get('prompt_text', '')} {job.get('response_text', '')}"
        
        # Find code blocks in markdown format
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            language = match.group(1) or "unknown"
            code = match.group(2).strip()
            if code and len(code) > 10:  # Skip trivial code
                code_blocks.append({
                    "code": code,
                    "language": language,
                    "type": "CODE_BLOCK",
                    "name": f"code_block_{len(code_blocks)}",
                    "source": "markdown_extraction"
                })
        
        return code_blocks
    
    async def process_urls(self, session_id: str, urls: List[str]):
        """Process and crawl URLs"""
        async with WebCrawler(delay=CRAWL_DELAY) as crawler:
            for url in urls:
                try:
                    crawl_result = await crawler.crawl_url(url)
                    await self.ingest_crawl_result(session_id, crawl_result)
                    
                    # If successful crawl, extract entities from content
                    if crawl_result.get("status") == "success" and crawl_result.get("content"):
                        content_entities = await self.extract_entities_from_content(
                            crawl_result["content"], url
                        )
                        if content_entities:
                            await self.ingest_content_entities(session_id, url, content_entities)
                    
                except Exception as e:
                    logger.error(f"Failed to process URL {url}: {e}")
    
    async def ingest_crawl_result(self, session_id: str, crawl_result: Dict):
        """Ingest web crawl results into Neo4j"""
        
        def ingest_transaction(tx):
            # Update or create Document node
            tx.run("""
                MERGE (d:Document {url: $url})
                SET d += {
                    title: $title,
                    content: $content,
                    content_length: $content_length,
                    domain: $domain,
                    status: $status,
                    content_type: $content_type,
                    crawled_at: $crawled_at,
                    opengraph_data: $opengraph,
                    headings: $headings,
                    image_count: $image_count,
                    internal_links_count: $internal_links_count,
                    external_links_count: $external_links_count,
                    updated_at: datetime()
                }
            """,
                url=crawl_result["url"],
                title=crawl_result.get("title", ""),
                content=crawl_result.get("content", "")[:10000],  # Limit size
                content_length=crawl_result.get("content_length", 0),
                domain=crawl_result.get("domain", ""),
                status=crawl_result.get("status", "unknown"),
                content_type=crawl_result.get("content_type", ""),
                crawled_at=crawl_result.get("crawled_at", ""),
                opengraph=json.dumps(crawl_result.get("opengraph", {})),
                headings=crawl_result.get("headings", []),
                image_count=len(crawl_result.get("images", [])),
                internal_links_count=len(crawl_result.get("internal_links", [])),
                external_links_count=len(crawl_result.get("external_links", []))
            )
            
            # Link to original prompt
            tx.run("""
                MATCH (p:Prompt {session_id: $session_id})
                MATCH (d:Document {url: $url})
                MERGE (p)-[:CRAWLED {
                    crawled_at: $crawled_at,
                    status: $status
                }]->(d)
            """,
                session_id=session_id,
                url=crawl_result["url"],
                crawled_at=crawl_result.get("crawled_at", ""),
                status=crawl_result.get("status", "unknown")
            )
        
        with self.neo4j_driver.session() as session:
            session.write_transaction(ingest_transaction)
    
    async def extract_entities_from_content(self, content: str, source_url: str) -> Dict:
        """Extract entities from crawled content using LLM"""
        if len(content) < 100:
            return {}
        
        # Truncate content for processing
        content = content[:3000]
        
        extraction_prompt = f"""Extract key entities from this web content. Return JSON only:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "PERSON|ORGANIZATION|LOCATION|TECHNOLOGY|CONCEPT|PRODUCT",
      "confidence": 0.0-1.0,
      "context": "surrounding text"
    }}
  ],
  "main_topics": ["topic1", "topic2"],
  "summary": "brief summary of content"
}}

Content from {source_url}:
{content}

Return only JSON:"""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.tier1_extractor.model,
                    "prompt": extraction_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2}
                }
                
                response = await client.post(f"{self.tier1_extractor.ollama_url}/api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                
                return json.loads(result.get("response", "{}"))
        except Exception as e:
            logger.error(f"Failed to extract entities from {source_url}: {e}")
            return {}
    
    async def ingest_content_entities(self, session_id: str, source_url: str, entities_data: Dict):
        """Ingest entities extracted from web content"""
        
        def ingest_transaction(tx):
            if not entities_data.get("entities"):
                return
            
            entities_list = []
            for entity in entities_data["entities"]:
                entities_list.append({
                    "name": entity["name"],
                    "type": entity.get("type", "CONCEPT"),
                    "confidence": entity.get("confidence", 0.5),
                    "context": entity.get("context", ""),
                    "source_url": source_url,
                    "session_id": session_id,
                    "extraction_source": "web_content",
                    "created_at": datetime.now(timezone.utc).isoformat()
                })
            
            # Batch create entities from web content
            tx.run("""
                UNWIND $entities_list AS entity
                MERGE (e:Entity {name: entity.name})
                ON CREATE SET e += {
                    type: entity.type,
                    first_seen: entity.created_at,
                    occurrence_count: 1
                }
                ON MATCH SET e.occurrence_count = coalesce(e.occurrence_count, 0) + 1
                SET e += {
                    web_confidence: entity.confidence,
                    web_context: entity.context,
                    extraction_source: entity.extraction_source,
                    updated_at: entity.created_at
                }
                WITH e, entity
                MATCH (d:Document {url: entity.source_url})
                MERGE (d)-[:HAS_ENTITY {
                    confidence: entity.confidence,
                    context: entity.context,
                    extracted_at: entity.created_at
                }]->(e)
                WITH e, entity
                MATCH (p:Prompt {session_id: entity.session_id})
                MERGE (p)-[:DISCOVERED_VIA_WEB {
                    source_url: entity.source_url,
                    confidence: entity.confidence,
                    discovered_at: entity.created_at
                }]->(e)
            """, entities_list=entities_list)
            
            # Store summary if available
            if entities_data.get("summary"):
                tx.run("""
                    MATCH (d:Document {url: $url})
                    SET d.ai_summary = $summary,
                        d.main_topics = $topics,
                        d.summary_generated_at = $timestamp
                """,
                    url=source_url,
                    summary=entities_data["summary"],
                    topics=entities_data.get("main_topics", []),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
        
        with self.neo4j_driver.session() as session:
            session.write_transaction(ingest_transaction)
    
    async def process_code_blocks(self, session_id: str, code_blocks: List[Dict]):
        """Process and analyze code blocks"""
        
        for code_block in code_blocks:
            try:
                language = code_block.get("language", "unknown").lower()
                code = code_block["code"]
                
                if language == "python":
                    analysis_result = self.code_analyzer.analyze_python_code(code)
                else:
                    # Basic analysis for other languages
                    analysis_result = {
                        "language": language,
                        "line_count": len(code.split('\n')),
                        "char_count": len(code),
                        "complexity": "LOW" if len(code) < 500 else "MEDIUM" if len(code) < 2000 else "HIGH"
                    }
                
                await self.ingest_code_analysis(session_id, code_block, analysis_result)
                
            except Exception as e:
                logger.error(f"Failed to analyze code block: {e}")
    
    async def ingest_code_analysis(self, session_id: str, code_block: Dict, analysis: Dict):
        """Ingest code analysis results"""
        
        def ingest_transaction(tx):
            # Create or update CodeElement node
            tx.run("""
                MERGE (c:CodeElement {name: $name, session_id: $session_id})
                SET c += {
                    code_text: $code,
                    language: $language,
                    type: $type,
                    source: $source,
                    analysis_result: $analysis,
                    line_count: $line_count,
                    complexity: $complexity,
                    analyzed_at: $analyzed_at,
                    updated_at: datetime()
                }
                WITH c
                MATCH (p:Prompt {session_id: $session_id})
                MERGE (p)-[:CONTAINS_ANALYZED_CODE {
                    language: $language,
                    complexity: $complexity,
                    analyzed_at: $analyzed_at
                }]->(c)
            """,
                name=code_block["name"],
                session_id=session_id,
                code=code_block["code"][:5000],  # Limit size
                language=code_block["language"],
                type=code_block["type"],
                source=code_block["source"],
                analysis=json.dumps(analysis),
                line_count=len(code_block["code"].split('\n')),
                complexity=analysis.get("complexity", "UNKNOWN"),
                analyzed_at=datetime.now(timezone.utc).isoformat()
            )
            
            # If Python analysis, create function and class nodes
            if "functions" in analysis:
                for func in analysis["functions"]:
                    tx.run("""
                        MATCH (c:CodeElement {name: $code_name, session_id: $session_id})
                        MERGE (f:Function {name: $func_name, parent_code: $code_name})
                        SET f += {
                            type: $func_type,
                            args: $args,
                            line_start: $line_start,
                            line_end: $line_end,
                            docstring: $docstring,
                            is_async: $is_async,
                            parent_class: $parent_class,
                            session_id: $session_id,
                            created_at: $created_at
                        }
                        MERGE (c)-[:DEFINES_FUNCTION]->(f)
                    """,
                        code_name=code_block["name"],
                        session_id=session_id,
                        func_name=func["name"],
                        func_type=func["type"],
                        args=func["args"],
                        line_start=func["line_start"],
                        line_end=func["line_end"],
                        docstring=func.get("docstring", ""),
                        is_async=func.get("is_async", False),
                        parent_class=func.get("parent_class"),
                        created_at=datetime.now(timezone.utc).isoformat()
                    )
        
        with self.neo4j_driver.session() as session:
            session.write_transaction(ingest_transaction)
    
    async def enqueue_tier2_processing(self, session_id: str, original_job: Dict, tier1_results: Dict):
        """Enqueue Tier 2 deep processing"""
        tier2_job = {
            "session_id": session_id,
            "tier": "tier_2_deep",
            "original_job": original_job,
            "tier1_results": tier1_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.redis_client.lpush("tier2_processing_queue", json.dumps(tier2_job))
        logger.info(f"Enqueued Tier 2 processing for session {session_id}")
    
    async def record_processing_error(self, session_id: str, error_message: str):
        """Record processing errors in Neo4j"""
        
        def error_transaction(tx):
            tx.run("""
                MATCH (j:ProcessingJob {session_id: $session_id})
                SET j.error_occurred = true,
                    j.error_message = $error_message,
                    j.error_timestamp = $timestamp,
                    j.completed = false
            """,
                session_id=session_id,
                error_message=error_message,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        with self.neo4j_driver.session() as session:
            session.write_transaction(error_transaction)
    
    async def stop(self):
        """Stop the worker gracefully"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        self.neo4j_driver.close()

async def main():
    """Main worker entry point"""
    worker = HeavyProcessingWorker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await worker.stop()
        logger.info("Heavy processing worker stopped")

if __name__ == "__main__":
    asyncio.run(main())