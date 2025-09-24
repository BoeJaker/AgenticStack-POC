"""
Graphiti Knowledge Graph System
A comprehensive implementation for consuming heterogeneous data and serving as LLM memory
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import aiofiles
from datetime import datetime, timezone
import hashlib
import re
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

# Core Graphiti imports
from graphiti import Graphiti
from graphiti.nodes import EntityNode, EpisodeNode
from graphiti.edges import Edge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Represents a data source with metadata"""
    source_id: str
    source_type: str  # 'chat', 'corpus', 'web_crawl', 'file'
    metadata: Dict[str, Any]
    content: str
    timestamp: datetime
    
class WebCrawler:
    """Simple web crawler for extracting content"""
    
    def __init__(self, max_depth: int = 2, delay: float = 1.0):
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls = set()
        
    async def crawl_url(self, url: str, session: aiohttp.ClientSession) -> Optional[str]:
        """Crawl a single URL and extract text content"""
        if url in self.visited_urls:
            return None
            
        try:
            self.visited_urls.add(url)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.text()
                    # Simple text extraction (in production, use proper HTML parsing)
                    text = re.sub(r'<[^>]+>', ' ', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
        
        return None
    
    async def crawl_site(self, base_url: str, max_pages: int = 10) -> List[DataSource]:
        """Crawl a website and return data sources"""
        data_sources = []
        
        async with aiohttp.ClientSession() as session:
            content = await self.crawl_url(base_url, session)
            if content:
                data_sources.append(DataSource(
                    source_id=hashlib.md5(base_url.encode()).hexdigest(),
                    source_type='web_crawl',
                    metadata={'url': base_url, 'crawl_depth': 0},
                    content=content,
                    timestamp=datetime.now(timezone.utc)
                ))
        
        return data_sources

class LLMInterface:
    """Interface for LLM interactions (Ollama/API support)"""
    
    def __init__(self, provider: str = "ollama", base_url: str = "http://localhost:11434"):
        self.provider = provider
        self.base_url = base_url
        
    async def generate_response(self, prompt: str, model: str = "llama2") -> str:
        """Generate response from LLM"""
        if self.provider == "ollama":
            return await self._ollama_generate(prompt, model)
        elif self.provider == "openai":
            return await self._openai_generate(prompt, model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _ollama_generate(self, prompt: str, model: str) -> str:
        """Generate using Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    result = await response.json()
                    return result.get("response", "")
        except Exception as e:
            logger.error(f"Error with Ollama generation: {e}")
            return ""
    
    async def _openai_generate(self, prompt: str, model: str) -> str:
        """Generate using OpenAI API"""
        # Implement OpenAI API call here
        # This is a placeholder implementation
        return "OpenAI response placeholder"
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using LLM"""
        prompt = f"""
        Extract entities from the following text. Return JSON format with entity name, type, and description:
        
        Text: {text}
        
        Return only valid JSON array of entities.
        """
        
        response = await self.generate_response(prompt)
        try:
            # Parse LLM response to extract entities
            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except json.JSONDecodeError:
            logger.error("Failed to parse entity extraction response")
            return []

class GraphitiKnowledgeGraph:
    """Main Knowledge Graph system using Graphiti"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        """Initialize the knowledge graph"""
        self.graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
        self.llm_interface = LLMInterface()
        self.crawler = WebCrawler()
        self.data_sources: List[DataSource] = []
        
    async def initialize(self):
        """Initialize the knowledge graph"""
        await self.graphiti.build_indices()
        logger.info("Knowledge graph initialized")
    
    async def add_data_source(self, data_source: DataSource):
        """Add a data source to the knowledge graph"""
        self.data_sources.append(data_source)
        
        # Process the data source and add to graph
        await self._process_data_source(data_source)
        
    async def _process_data_source(self, data_source: DataSource):
        """Process a data source and add entities/relationships to graph"""
        try:
            # Extract entities using LLM
            entities = await self.llm_interface.extract_entities(data_source.content)
            
            # Create episode node for this data source
            episode_node = EpisodeNode(
                name=f"episode_{data_source.source_id}",
                created_at=data_source.timestamp,
                content=data_source.content[:1000],  # Truncate for storage
                source_description=f"{data_source.source_type}: {data_source.metadata}"
            )
            
            # Add episode to graph
            await self.graphiti.add_episode(episode_node)
            
            # Add extracted entities
            for entity_data in entities:
                entity_node = EntityNode(
                    name=entity_data.get('name', 'unknown'),
                    labels=[entity_data.get('type', 'entity')],
                    created_at=data_source.timestamp
                )
                
                await self.graphiti.add_entity(entity_node)
                
                # Create relationship between episode and entity
                edge = Edge(
                    source_node_id=episode_node.uuid,
                    target_node_id=entity_node.uuid,
                    edge_type="CONTAINS_ENTITY",
                    created_at=data_source.timestamp
                )
                
                await self.graphiti.add_edge(edge)
                
        except Exception as e:
            logger.error(f"Error processing data source {data_source.source_id}: {e}")
    
    async def consume_chat_data(self, messages: List[Dict[str, str]], 
                               conversation_id: str = None):
        """Consume chat/conversation data"""
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" 
                                     for msg in messages])
        
        data_source = DataSource(
            source_id=conversation_id or hashlib.md5(conversation_text.encode()).hexdigest(),
            source_type='chat',
            metadata={'conversation_id': conversation_id, 'message_count': len(messages)},
            content=conversation_text,
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.add_data_source(data_source)
        logger.info(f"Added chat data: {len(messages)} messages")
    
    async def consume_corpus_data(self, corpus_path: Union[str, Path], 
                                 chunk_size: int = 1000):
        """Consume textual corpus data"""
        corpus_path = Path(corpus_path)
        
        if corpus_path.is_file():
            await self._process_file(corpus_path, chunk_size)
        elif corpus_path.is_dir():
            for file_path in corpus_path.rglob("*.txt"):
                await self._process_file(file_path, chunk_size)
        
        logger.info(f"Processed corpus data from {corpus_path}")
    
    async def _process_file(self, file_path: Path, chunk_size: int):
        """Process a single file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Split into chunks if too large
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                data_source = DataSource(
                    source_id=f"{file_path.stem}_chunk_{i}",
                    source_type='corpus',
                    metadata={'file_path': str(file_path), 'chunk_index': i},
                    content=chunk,
                    timestamp=datetime.now(timezone.utc)
                )
                await self.add_data_source(data_source)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    async def consume_web_crawl_data(self, urls: List[str]):
        """Consume web crawl data"""
        for url in urls:
            data_sources = await self.crawler.crawl_site(url)
            for data_source in data_sources:
                await self.add_data_source(data_source)
        
        logger.info(f"Crawled {len(urls)} URLs")
    
    async def query_memory(self, query: str, context_window: int = 5) -> Dict[str, Any]:
        """Query the knowledge graph as LLM memory"""
        try:
            # Search for relevant episodes and entities
            episodes = await self.graphiti.search_episodes(query, limit=context_window)
            entities = await self.graphiti.search_entities(query, limit=context_window)
            
            # Build context from results
            context = {
                'query': query,
                'episodes': [self._episode_to_dict(ep) for ep in episodes],
                'entities': [self._entity_to_dict(ent) for ent in entities],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return {'query': query, 'episodes': [], 'entities': [], 'error': str(e)}
    
    def _episode_to_dict(self, episode: EpisodeNode) -> Dict[str, Any]:
        """Convert episode node to dictionary"""
        return {
            'id': str(episode.uuid),
            'name': episode.name,
            'content': episode.content,
            'created_at': episode.created_at.isoformat(),
            'source_description': episode.source_description
        }
    
    def _entity_to_dict(self, entity: EntityNode) -> Dict[str, Any]:
        """Convert entity node to dictionary"""
        return {
            'id': str(entity.uuid),
            'name': entity.name,
            'labels': entity.labels,
            'created_at': entity.created_at.isoformat()
        }
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the knowledge graph memory"""
        try:
            # Get counts and statistics
            stats = await self.graphiti.get_statistics()
            
            return {
                'total_episodes': stats.get('episode_count', 0),
                'total_entities': stats.get('entity_count', 0),
                'total_edges': stats.get('edge_count', 0),
                'data_sources': len(self.data_sources),
                'source_types': list(set(ds.source_type for ds in self.data_sources)),
                'last_updated': max([ds.timestamp for ds in self.data_sources]).isoformat() if self.data_sources else None
            }
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {'error': str(e)}
    
    async def enhance_llm_response(self, query: str, llm_response: str) -> str:
        """Enhance LLM response with knowledge graph context"""
        # Query memory for relevant context
        memory_context = await self.query_memory(query)
        
        if memory_context['episodes'] or memory_context['entities']:
            # Create enhanced prompt with memory context
            context_text = self._format_memory_context(memory_context)
            
            enhanced_prompt = f"""
            Based on the following context from memory, enhance this response:
            
            Original Query: {query}
            Original Response: {llm_response}
            
            Memory Context:
            {context_text}
            
            Provide an enhanced response that incorporates relevant information from memory.
            """
            
            enhanced_response = await self.llm_interface.generate_response(enhanced_prompt)
            return enhanced_response if enhanced_response else llm_response
        
        return llm_response
    
    def _format_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """Format memory context for LLM prompt"""
        context_parts = []
        
        if memory_context['episodes']:
            context_parts.append("Relevant Episodes:")
            for ep in memory_context['episodes'][:3]:  # Limit to top 3
                context_parts.append(f"- {ep['name']}: {ep['content'][:200]}...")
        
        if memory_context['entities']:
            context_parts.append("Relevant Entities:")
            for ent in memory_context['entities'][:5]:  # Limit to top 5
                context_parts.append(f"- {ent['name']} ({', '.join(ent['labels'])})")
        
        return "\n".join(context_parts)

# Usage Example and API Wrapper
class GraphitiKGAPI:
    """API wrapper for the Knowledge Graph system"""
    
    def __init__(self):
        self.kg = GraphitiKnowledgeGraph()
    
    async def initialize(self):
        """Initialize the system"""
        await self.kg.initialize()
    
    async def add_chat_conversation(self, messages: List[Dict[str, str]], 
                                  conversation_id: str = None):
        """Add chat conversation to knowledge graph"""
        await self.kg.consume_chat_data(messages, conversation_id)
        return {"status": "success", "message": "Chat data added"}
    
    async def add_corpus(self, corpus_path: str, chunk_size: int = 1000):
        """Add corpus data to knowledge graph"""
        await self.kg.consume_corpus_data(corpus_path, chunk_size)
        return {"status": "success", "message": "Corpus data added"}
    
    async def add_web_crawl(self, urls: List[str]):
        """Add web crawl data to knowledge graph"""
        await self.kg.consume_web_crawl_data(urls)
        return {"status": "success", "message": "Web crawl data added"}
    
    async def query_memory(self, query: str, context_window: int = 5):
        """Query the knowledge graph memory"""
        return await self.kg.query_memory(query, context_window)
    
    async def get_summary(self):
        """Get knowledge graph summary"""
        return await self.kg.get_memory_summary()
    
    async def chat_with_memory(self, query: str, model: str = "llama2"):
        """Chat with LLM enhanced by knowledge graph memory"""
        # Get basic LLM response
        basic_response = await self.kg.llm_interface.generate_response(query, model)
        
        # Enhance with memory
        enhanced_response = await self.kg.enhance_llm_response(query, basic_response)
        
        # Add this interaction to memory
        await self.add_chat_conversation([
            {"role": "user", "content": query},
            {"role": "assistant", "content": enhanced_response}
        ])
        
        return {
            "query": query,
            "response": enhanced_response,
            "memory_enhanced": len(enhanced_response) > len(basic_response)
        }

# Example usage
async def main():
    """Example usage of the GraphitiKG system"""
    
    # Initialize the system
    kg_api = GraphitiKGAPI()
    await kg_api.initialize()
    
    # Add different types of data
    
    # 1. Add chat conversation
    chat_messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data."}
    ]
    await kg_api.add_chat_conversation(chat_messages, "conv_001")
    
    # 2. Add corpus data (assuming you have text files)
    # await kg_api.add_corpus("./documents/")
    
    # 3. Add web crawl data
    # await kg_api.add_web_crawl(["https://example.com/ai-articles"])
    
    # Query memory
    memory_result = await kg_api.query_memory("machine learning")
    print("Memory Query Result:", json.dumps(memory_result, indent=2))
    
    # Chat with memory enhancement
    chat_result = await kg_api.chat_with_memory("Tell me more about AI")
    print("Chat Result:", json.dumps(chat_result, indent=2))
    
    # Get system summary
    summary = await kg_api.get_summary()
    print("System Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(main())