#!/usr/bin/env python3
"""
Interactive Log File Query Tool

Query your indexed log files using natural language and get AI-generated answers.
Uses the embeddings created by index.py and generates responses using Ollama.

Usage:
    python ask.py [output_file.md]

Arguments:
    output_file.md    Optional. Markdown file to save Q&A pairs (default: ask-YYYY-Month-DD-HH-MM.md)
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
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text:latest')
EMBEDDING_SERVER = os.getenv('EMBEDDING_SERVER', 'http://localhost:5000')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5')
CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
USE_LOCAL_EMBEDDINGS = os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() == 'true'
USE_LOCAL_OLLAMA = os.getenv('USE_LOCAL_OLLAMA', 'true').lower() == 'true'
DEFAULT_OUTPUT_FILE_PREFIX = os.getenv('DEFAULT_OUTPUT_FILE_PREFIX', 'ask')
DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', '5'))

# Generate default output filename with timestamp
def get_default_output_file() -> str:
    """Generate a timestamp-based output filename"""
    timestamp = datetime.now().strftime("%Y-%B-%d-%H-%M")
    return f"{DEFAULT_OUTPUT_FILE_PREFIX}-{timestamp}.md"

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
        db_path = CHROMA_PATH
        
        if os.path.exists(db_path):
            try:
                client = chromadb.PersistentClient(path=db_path)
                self.collection = client.get_collection("vectors")
                console.print(f"[green]✓ Using database at: {db_path}[/green]")
                console.print(f"[green]✓ Collection: vectors[/green]")
                return
            except Exception as e:
                console.print(f"[red]Error loading collection 'vectors': {e}[/red]")
        
        raise Exception(
            f"No indexed database found at {db_path}. Please run: python index.py /path/to/repository"
        )
    
    def _load_metadata(self) -> None:
        """Load indexing metadata to determine embedding type"""
        metadata_path = Path(CHROMA_PATH) / "index_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.embedding_type = metadata.get('embedding_type')
                    self.embedding_model = metadata.get('embedding_model', EMBEDDING_MODEL)
                    
                    if not self.embedding_type:
                        raise ValueError("Missing embedding_type in metadata")
                    
                    console.print(f"[cyan]Embedding type: {self.embedding_type}[/cyan]")
                    console.print(f"[cyan]Embedding model: {self.embedding_model}[/cyan]")
            except Exception as e:
                console.print(f"[red]Error loading metadata: {e}[/red]")
                console.print("[red]Please re-index your repository with: python index.py /path/to/repository[/red]")
                raise Exception("Invalid or missing metadata. Re-indexing required.")
        else:
            console.print(f"[red]No metadata file found at {metadata_path}[/red]")
            console.print("[red]Please index your repository first with: python index.py /path/to/repository[/red]")
            raise Exception("No index metadata found. Please run indexing first.")
    
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
            if not self._local_model:
                console.print("[yellow]Loading local embedding model...[/yellow]")
                from trust_manager import safe_sentence_transformer_load  # type: ignore[import]
                self._local_model = safe_sentence_transformer_load(self.embedding_model)
                self._local_model.max_seq_length = 512  # type: ignore[attr-defined]
            
            embedding = self._local_model.encode([text], show_progress_bar=False)  # type: ignore[attr-defined]
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
            # Get trust setting for this model
            from trust_manager import TrustManager
            trust_manager = TrustManager()
            trust_remote_code = trust_manager.get_trust_setting(self.embedding_model, interactive=True)
            
            response = requests.post(
                f"{EMBEDDING_SERVER}/embed",
                json={
                    "texts": [text], 
                    "model": self.embedding_model,
                    "trust_remote_code": trust_remote_code
                },
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
                ], 
                think=False
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
    
    output_file = sys.argv[1] if len(sys.argv) == 2 else get_default_output_file()
    
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