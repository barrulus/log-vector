"""
Shared configuration for log-vector.

All environment variables and defaults are centralised here so that
index.py, ask.py, and embedding_server.py share a single source of truth.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Ollama
OLLAMA_HOST: str = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
OLLAMA_EMBEDDING_MODEL: str = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text:latest')

# Embedding
EMBEDDING_SERVER: str = os.getenv('EMBEDDING_SERVER', 'http://localhost:5000')
EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5')

# Storage
CHROMA_PATH: str = os.getenv('CHROMA_PATH', './chroma_db')
DEFAULT_CHUNK_SIZE: int = int(os.getenv('DEFAULT_CHUNK_SIZE', '2000'))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '200'))

# Query
DEFAULT_OUTPUT_FILE_PREFIX: str = os.getenv('DEFAULT_OUTPUT_FILE_PREFIX', 'ask')
DEFAULT_TOP_K: int = int(os.getenv('DEFAULT_TOP_K', '5'))
