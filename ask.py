#!/usr/bin/env python3
"""
Interactive Log File Query Tool

Query your indexed log files using natural language and get AI-generated answers.
Uses the embeddings created by index.py and generates responses using Ollama.

Usage:
    python ask.py [output_file.md]

Arguments:
    output_file.md    Optional. Markdown file to save Q&A pairs (default: logfile_queries.md)
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any, Optional, cast
from datetime import datetime
from pathlib import Path

import chromadb
from rich.console import Console
from rich.markdown import Markdown
from ollama import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'dolphincoder:15b')
EMBEDDING_SERVER = os.getenv('EMBEDDING_SERVER', 'http://localhost:5000')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5')
CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_code')
USE_LOCAL_EMBEDDINGS = os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() == 'true'
USE_LOCAL_OLLAMA = os.getenv('USE_LOCAL_OLLAMA', 'true').lower() == 'true'

# Constants
DEFAULT_OUTPUT_FILE = "logfile_queries.md"
DEFAULT_TOP_K = 5
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Initialize console
console = Console()


class QueryHandler:
    """Handle querying and embedding generation"""
    
    def __init__(self):
        self.ollama_client = Client(host=OLLAMA_HOST)
        self.collection: Optional[Any] = None
        self.embedding_type: Optional[str] = None
        self.embedding_model: str = EMBEDDING_MODEL
        self._local_model: Optional[Any] = None
        self._load_database()
        self._load_metadata()
        
    def _load_database(self) -> None:
        """Load ChromaDB collection"""
        # Try to load from configured path first
        db_paths = [CHROMA_PATH]
        
        # Add fallback paths for backward compatibility
        fallback_paths = [
            "./chroma_ollama",
            "./chroma_code_optimized", 
            "./chroma_db",
            "./chroma_code"
        ]
        
        for path in fallback_paths:
            if path not in db_paths:
                db_paths.append(path)
        
        for db_path in db_paths:
            if os.path.exists(db_path):
                try:
                    client = chromadb.PersistentClient(path=db_path)
                    # Try different collection names
                    for collection_name in ["code", "logs"]:
                        try:
                            self.collection = client.get_collection(collection_name)
                            console.print(f"[green]✓ Using database at: {db_path}[/green]")
                            console.print(f"[green]✓ Collection: {collection_name}[/green]")
                            return
                        except Exception:
                            continue
                except Exception:
                    continue
        
        raise Exception(
            f"No indexed database found. Please run: python index.py /path/to/repository"
        )
    
    def _load_metadata(self) -> None:
        """Load indexing metadata if available"""
        metadata_path = Path(CHROMA_PATH) / "index_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.embedding_type = metadata.get('embedding_type', 'unknown')
                    self.embedding_model = metadata.get('embedding_model', EMBEDDING_MODEL)
                    console.print(f"[cyan]Embedding type: {self.embedding_type}[/cyan]")
                    console.print(f"[cyan]Embedding model: {self.embedding_model}[/cyan]")
            except Exception:
                self._guess_embedding_type()
        else:
            self._guess_embedding_type()
    
    def _guess_embedding_type(self) -> None:
        """Guess embedding type from database path"""
        db_path = ""
        if self.collection and hasattr(self.collection, '_client'):
            client = getattr(self.collection, '_client')
            if hasattr(client, '_path'):
                db_path = str(getattr(client, '_path'))
        
        if "ollama" in db_path:
            self.embedding_type = "ollama"
            self.embedding_model = OLLAMA_EMBEDDING_MODEL
        elif "optimized" in db_path:
            self.embedding_type = "local"
            self.embedding_model = EMBEDDING_MODEL
        elif "remote" in db_path or "gpu" in db_path:
            self.embedding_type = "remote"
            self.embedding_model = EMBEDDING_MODEL
        else:
            # Default to configuration
            if USE_LOCAL_EMBEDDINGS:
                self.embedding_type = "local"
            else:
                self.embedding_type = "remote"
            self.embedding_model = EMBEDDING_MODEL
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using the appropriate method"""
        if self.embedding_type == "ollama":
            return self._get_embedding_ollama(text)
        elif self.embedding_type == "local":
            return self._get_embedding_local(text)
        else:
            return self._get_embedding_remote(text)
    
    def _get_embedding_ollama(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()['embedding']
        except Exception as e:
            console.print(f"[red]Error getting Ollama embedding: {e}[/red]")
            raise
    
    def _get_embedding_local(self, text: str) -> List[float]:
        """Get embedding using local model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            if not self._local_model:
                console.print("[yellow]Loading local embedding model...[/yellow]")
                self._local_model = SentenceTransformer(self.embedding_model, trust_remote_code=True)
                self._local_model.max_seq_length = 512
            
            embedding = self._local_model.encode([text], show_progress_bar=False)
            # Handle different return types from encode
            if hasattr(embedding, 'tolist'):
                # If it's a numpy array
                return cast(List[float], embedding[0].tolist())
            elif isinstance(embedding, list):
                if len(cast(List[Any], embedding)) > 0:
                    return cast(List[float], embedding[0])
                else:
                    return []
            else:
                # Handle tensor or other types  
                try:
                    # Try to access first element
                    first_item = embedding[0]
                    return cast(List[float], list(first_item))
                except (IndexError, TypeError):
                    # If we can't access it, try converting the whole thing
                    return cast(List[float], list(embedding))
        except ImportError:
            console.print("[red]Local embeddings require sentence-transformers. Falling back to remote.[/red]")
            self.embedding_type = "remote"
            return self._get_embedding_remote(text)
    
    def _get_embedding_remote(self, text: str) -> List[float]:
        """Get embedding from remote server"""
        try:
            response = requests.post(
                f"{EMBEDDING_SERVER}/embed",
                json={"texts": [text], "model": self.embedding_model},
                timeout=60
            )
            response.raise_for_status()
            return response.json()['embeddings'][0]
        except Exception as e:
            console.print(f"[red]Error getting remote embedding: {e}[/red]")
            raise
    
    def query_codebase(self, question: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Query the logs and generate a response"""
        # Get embedding for the question
        try:
            q_embed = self.get_embedding(question)
        except Exception as e:
            return f"Error generating embedding: {e}"
        
        # Query ChromaDB
        if not self.collection:
            return "Database collection not loaded."
            
        results = self.collection.query(
            query_embeddings=[q_embed],
            n_results=top_k
        )
        
        # Build context from results
        context = ""
        if results.get('documents') and results.get('metadatas'):
            documents = cast(List[Any], results['documents'][0] if results['documents'] else [])
            metadatas = cast(List[Dict[str, Any]], results['metadatas'][0] if results['metadatas'] else [])
            for doc, meta in zip(documents, metadatas):
                source = str(meta.get('source', meta.get('file', 'Unknown')) if meta else 'Unknown')
                chunk_idx = str(meta.get('chunk_index', '') if meta else '')
                
                context += f"File: {source}"
                if chunk_idx:
                    context += f" (chunk {chunk_idx})"
                context += f"\n{str(doc)}\n\n"
        
        if not context:
            return "No relevant data found for your question."
        
        # Generate response using Ollama
        prompt = f"""You are a helpful systems administrator. Here's some relevant server log context:

{context}

Now answer this question: {question}

Format your entire response using markdown with appropriate headers, code blocks, bullet points, etc."""
        
        try:
            response = cast(Any, self.ollama_client.chat(  # type: ignore[misc]
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a systems administrator that provides clear, accurate answers based on the provided server log context. Always format your responses in well-structured markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            ))
            # Handle the response - expect dict with 'message' key
            if hasattr(response, 'get') and response.get('message'):
                message = response.get('message', {})
                if hasattr(message, 'get'):
                    content = message.get('content', '')
                    return str(content) if content else "No content in response"
                return str(message)
            return "Error: Invalid response format"
        except Exception as e:
            return f"Error generating response: {e}"


def write_to_markdown(question: str, answer: str, filename: str) -> None:
    """Append question and answer to a markdown file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the file if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Log File Query Log\n\n")
            f.write("This file contains questions and answers about the Log Files.\n\n")
    
    # Append the Q&A pair
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"## Question [{timestamp}]\n\n")
        f.write(f"**Q:** {question}\n\n")
        f.write(f"**A:**\n\n{answer}\n\n")
        f.write("---\n\n")


def main() -> None:
    """Main interactive loop"""
    # Parse command line arguments
    if len(sys.argv) > 2:
        console.print("[red]Usage: python ask.py [output_file.md][/red]")
        sys.exit(1)
    
    output_file = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_OUTPUT_FILE
    
    console.print(f"\n[bold cyan]Log Query Tool[/bold cyan]")
    console.print(f"Output file: [cyan]{output_file}[/cyan]")
    console.print(f"Ollama model: [cyan]{OLLAMA_MODEL}[/cyan]")
    console.print("\nType 'exit' or 'quit' to stop.\n")
    
    # Initialize query handler
    try:
        handler = QueryHandler()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    
    # Interactive loop
    while True:
        try:
            question = input("\n[?] Ask a question about the log files: ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print(f"\n[green]✓ All responses saved to {output_file}[/green]")
                break
            
            if not question.strip():
                continue
            
            # Generate answer
            console.print("\n[yellow]Searching log files and generating response...[/yellow]")
            answer = handler.query_codebase(question)
            
            # Write to file
            write_to_markdown(question, answer, output_file)
            
            # Display answer
            console.print("\n[bold]Answer:[/bold]\n")
            console.print(Markdown(answer))
            
            console.print(f"\n[dim]Response saved to {output_file}[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()