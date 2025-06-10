#!/usr/bin/env python3
"""
Unified Log File Indexer with Multiple Embedding Options

This script indexes a path of log files for semantic search using various embedding strategies:
- Local embeddings using SentenceTransformer
- Ollama embeddings using Ollama's API
- Remote embeddings using a dedicated embedding server

Usage:
    python index.py /path/to/repository [options]

Options:
    --local-embeddings    Use local SentenceTransformer (default)
    --ollama-embeddings   Use Ollama's embedding API
    --remote-embeddings   Use remote embedding server
    --model MODEL         Specify embedding model (overrides .env)
    --chunk-size SIZE     Size of log chunks (default: 2000)
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
from datetime import datetime
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# Load environment variables
load_dotenv()

# Configuration from environment
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
EMBEDDING_SERVER = os.getenv('EMBEDDING_SERVER', 'http://localhost:5000')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5')
CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_code')
USE_LOCAL_EMBEDDINGS = os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() == 'true'
USE_LOCAL_OLLAMA = os.getenv('USE_LOCAL_OLLAMA', 'true').lower() == 'true'

# Constants
DEFAULT_CHUNK_SIZE = 2000
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

console = Console()


class EmbeddingHandler:
    """Base class for embedding handlers"""
    
    def __init__(self, model: Optional[str] = None):
        self.model: str = model or EMBEDDING_MODEL
        
    def embed(self, _texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        raise NotImplementedError
        
    def check_availability(self) -> bool:
        """Check if the embedding service is available"""
        raise NotImplementedError


class LocalEmbeddingHandler(EmbeddingHandler):
    """Handle embeddings using local SentenceTransformer"""
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(model)
        self.transformer: Optional[Any] = None
        self.device: str = 'cpu'
        try:
            import torch
            
            # Check for CUDA availability
            if torch.cuda.is_available():
                console.print(f"[green]✓ CUDA available: {torch.cuda.get_device_name(0)}[/green]")
                self.device = 'cuda'
            else:
                console.print("[yellow]! CUDA not available, using CPU[/yellow]")
                self.device = 'cpu'
                
            from trust_manager import safe_sentence_transformer_load  # type: ignore[import]
            self.transformer = safe_sentence_transformer_load(self.model, device=self.device)
            self.transformer.max_seq_length = 512  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        if not self.transformer:
            raise RuntimeError("Transformer not initialized")
        embeddings = self.transformer.encode(texts, batch_size=32, show_progress_bar=False)
        # The encode method returns numpy array, convert to list
        return embeddings.tolist()
    
    def check_availability(self) -> bool:
        """Local embeddings are always available if initialized"""
        return True


class OllamaEmbeddingHandler(EmbeddingHandler):
    """Handle embeddings using Ollama's API"""
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(model or OLLAMA_EMBEDDING_MODEL)
        self.base_url: str = OLLAMA_HOST
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama API"""
        embeddings: List[List[float]] = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)
            except Exception as e:
                console.print(f"[red]Error generating embedding: {e}[/red]")
                # Return zero embedding on error
                embeddings.append([0.0] * 384)  # Typical embedding size
                
        return embeddings
    
    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if the embedding model is available
            models = response.json()["models"]
            model_names = [model["name"] for model in models]
            
            if self.model not in model_names:
                console.print(f"[yellow]Model {self.model} not found. Available models: {model_names}[/yellow]")
                console.print(f"[yellow]Please run: ollama pull {self.model}[/yellow]")
                return False
                
            return True
        except Exception as e:
            console.print(f"[red]Cannot connect to Ollama at {self.base_url}: {e}[/red]")
            return False


class RemoteEmbeddingHandler(EmbeddingHandler):
    """Handle embeddings using remote embedding server"""
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(model)
        self.base_url: str = EMBEDDING_SERVER
        self.max_retries: int = 3
        self.retry_delay: int = 1
        self.trust_remote_code: bool = self._get_trust_setting()
    
    def _get_trust_setting(self) -> bool:
        """Get trust_remote_code setting for this model"""
        from trust_manager import TrustManager
        trust_manager = TrustManager()
        return trust_manager.get_trust_setting(self.model, interactive=True)
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using remote server with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/embed",
                    json={
                        "texts": texts, 
                        "model": self.model,
                        "trust_remote_code": self.trust_remote_code
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["embeddings"]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    console.print(f"[yellow]Retry {attempt + 1}/{self.max_retries} after {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
                else:
                    console.print(f"[red]Failed to get embeddings after {self.max_retries} attempts: {e}[/red]")
                    raise
        # This should never be reached, but satisfies type checker
        return []
    
    def check_availability(self) -> bool:
        """Check if remote embedding server is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            console.print(f"[green]✓ Remote embedding server available at {self.base_url}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Cannot connect to embedding server at {self.base_url}: {e}[/red]")
            return False


def is_indexable_file(file_path: Path) -> bool:
    """Determine if a file can be indexed by examining its content"""
    try:
        # Skip if file is too large (> 100MB)
        if file_path.stat().st_size > 100 * 1024 * 1024:
            return False
            
        with open(file_path, 'rb') as f:
            # Read first 8KB to check content
            chunk = f.read(8192)
            if not chunk:
                return False  # Empty file
            
            # Check for null bytes (indicates binary content)
            if b'\x00' in chunk:
                return False
            
            # Try to decode as text using common encodings
            for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                try:
                    chunk.decode(encoding)
                    return True
                except UnicodeDecodeError:
                    continue
            
            return False
                
    except (IOError, OSError, PermissionError):
        return False


def collect_files(repo_path: Path) -> List[Path]:
    """Collect all indexable files from the repository by scanning content"""
    files: List[Path] = []
    
    # Filter out common directories to ignore
    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.env', 'venv', 'env', '.venv', 
                   'target', 'build', 'dist', '.svn', '.hg', '.idea', '.vscode'}
    
    # Recursively scan all files
    for file_path in repo_path.rglob('*'):
        if file_path.is_file():
            # Skip files in ignored directories
            if any(ignored in file_path.parts for ignored in ignore_dirs):
                continue
                
            # Check if file is indexable by content
            if is_indexable_file(file_path):
                files.append(file_path)
    
    return sorted(files)


def chunk_code(content: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split logs into chunks"""
    chunks: List[str] = []
    lines = content.split('\n')
    current_chunk: List[str] = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def process_repository(
    repo_path: Path,
    embedding_handler: EmbeddingHandler,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_size: int = 200
) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]], List[str]]:
    """Process repository and generate embeddings"""
    
    files = collect_files(repo_path)
    console.print(f"\n[cyan]Found {len(files)} files to index[/cyan]")
    
    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []
    all_ids: List[str] = []
    
    # Process files and create chunks
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=len(files))
        
        for file_path in files:
            try:
                # Try different encodings to read the file
                content = None
                for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        content = file_path.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    console.print(f"[yellow]Could not decode {file_path}, skipping[/yellow]")
                    continue
                chunks = chunk_code(content, chunk_size)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Skip empty chunks
                        all_chunks.append(chunk)
                        all_metadata.append({
                            "source": str(file_path.relative_to(repo_path)),
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        })
                        all_ids.append(f"{file_path.name}:{i}")
                        
            except Exception as e:
                console.print(f"[yellow]Error processing {file_path}: {e}[/yellow]")
                
            progress.update(task, advance=1)
    
    console.print(f"[cyan]Created {len(all_chunks)} chunks[/cyan]")
    
    # Generate embeddings in batches
    all_embeddings: List[List[float]] = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=len(all_chunks))
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_embeddings = embedding_handler.embed(batch)
            all_embeddings.extend(batch_embeddings)
            progress.update(task, advance=len(batch))
    
    return all_chunks, all_embeddings, all_metadata, all_ids


def save_to_chromadb(
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: List[Dict[str, Any]],
    ids: List[str],
    persist_directory: str
) -> None:
    """Save embeddings to ChromaDB"""
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name="code")
    except Exception:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name="code",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add data in batches
    batch_size = 500
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Saving to ChromaDB...", total=len(chunks))
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            collection.add(
                documents=chunks[i:end_idx],
                embeddings=cast(Any, embeddings[i:end_idx]),
                metadatas=cast(Any, metadata[i:end_idx]),
                ids=ids[i:end_idx]
            )
            progress.update(task, advance=end_idx - i)
    
    console.print(f"[green]✓ Saved {len(chunks)} embeddings to {persist_directory}[/green]")


def save_metadata(repo_path: Path, embedding_type: str, model: str, chunk_size: int, chroma_path: str) -> None:
    """Save indexing metadata for later reference"""
    metadata: Dict[str, Any] = {
        "indexed_at": datetime.now().isoformat(),
        "repository": str(repo_path),
        "embedding_type": embedding_type,
        "embedding_model": model,
        "chunk_size": chunk_size,
        "chroma_path": str(Path(chroma_path).resolve())
    }
    
    metadata_path = Path(chroma_path) / "index_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"[green]✓ Saved metadata to {metadata_path}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index log files for semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('repository', type=str, help='Path to the repository to index')
    
    # Embedding type selection
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        '--local-embeddings', 
        action='store_true', 
        default=USE_LOCAL_EMBEDDINGS,
        help='Use local SentenceTransformer embeddings (default based on .env)'
    )
    embedding_group.add_argument(
        '--ollama-embeddings', 
        action='store_true',
        help='Use Ollama embedding API'
    )
    embedding_group.add_argument(
        '--remote-embeddings', 
        action='store_true',
        help='Use remote embedding server'
    )
    
    # Other options
    parser.add_argument(
        '--model', 
        type=str, 
        help=f'Embedding model to use (default: {EMBEDDING_MODEL} or {OLLAMA_EMBEDDING_MODEL} for Ollama)'
    )
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=DEFAULT_CHUNK_SIZE,
        help=f'Size of log file chunks (default: {DEFAULT_CHUNK_SIZE})'
    )
    parser.add_argument(
        '--chroma-path',
        type=str,
        default=CHROMA_PATH,
        help=f'Path to ChromaDB storage (default: {CHROMA_PATH})'
    )
    
    args = parser.parse_args()
    
    # Use the specified chroma_path
    chroma_path = args.chroma_path
    
    # Validate repository path
    repo_path = Path(args.repository).resolve()
    if not repo_path.exists():
        console.print(f"[red]Error: Repository path does not exist: {repo_path}[/red]")
        sys.exit(1)
    
    console.print(f"\n[bold cyan]Log Indexer[/bold cyan]")
    console.print(f"Repository: {repo_path}")
    
    # Determine embedding type
    if args.ollama_embeddings:
        embedding_type = "ollama"
        handler = OllamaEmbeddingHandler(args.model)
    elif args.remote_embeddings:
        embedding_type = "remote"
        handler = RemoteEmbeddingHandler(args.model)
    else:
        embedding_type = "local"
        handler = LocalEmbeddingHandler(args.model)
    
    console.print(f"Embedding type: [cyan]{embedding_type}[/cyan]")
    console.print(f"Embedding model: [cyan]{handler.model}[/cyan]")
    console.print(f"Chunk size: [cyan]{args.chunk_size}[/cyan]")
    console.print(f"ChromaDB path: [cyan]{chroma_path}[/cyan]")
    
    # Check availability
    if not handler.check_availability():
        console.print("[red]Embedding service not available. Exiting.[/red]")
        sys.exit(1)
    
    # Process repository
    start_time = time.time()
    
    try:
        chunks, embeddings, metadata, ids = process_repository(
            repo_path,
            handler,
            chunk_size=args.chunk_size
        )
        
        # Save to ChromaDB
        save_to_chromadb(chunks, embeddings, metadata, ids, chroma_path)
        
        # Save metadata
        save_metadata(repo_path, embedding_type, handler.model, args.chunk_size, chroma_path)
        
        elapsed_time = time.time() - start_time
        console.print(f"\n[green]✓ Indexing completed in {elapsed_time:.2f} seconds[/green]")
        
    except Exception as e:
        console.print(f"\n[red]Error during indexing: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()