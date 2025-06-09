# Log File Analysis System

A powerful semantic search system for log files that enables natural language queries over your logs using vector embeddings and local LLM integration.

## Features

- **Multiple Embedding Strategies**: Choose between local SentenceTransformer, Ollama embeddings, or remote embedding server
- **Flexible Storage**: Uses ChromaDB for persistent vector storage with configurable paths
- **Local LLM Integration**: Generates AI responses using Ollama with customizable models
- **Interactive Query Interface**: Rich terminal interface with markdown rendering
- **GPU Acceleration**: Optional GPU support for faster embedding generation
- **Comprehensive File Support**: Indexes `.py`, `.log`, `.js`, `.ts`, `.md`, `.sql`, `.html`, `.csv` files
- **Environment Configuration**: Fully configurable via `.env` files

## Quick Start

### 1. Install Dependencies

I advise you run in a venv.

```bash
python -m venv venv
source venv/bin/activate
```

or on Windows

```ps1
python.exe -m venv venv
.\venv\Scripts\Activate
```

then

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and customize it:

```bash
cp .env_example .env
```

Edit `.env` to configure your setup:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=dolphincoder:15b

# Embedding Configuration  
EMBEDDING_SERVER=http://localhost:5000
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# ChromaDB Configuration
CHROMA_PATH=./chroma_code

# Default Settings
USE_LOCAL_EMBEDDINGS=true
USE_LOCAL_OLLAMA=true
```

### 3. Index Your Log Files

Index a directory using local embeddings (default):

```bash
python index.py /path/to/your/logs
```

Or specify embedding type:

```bash
# Use local SentenceTransformer embeddings
python index.py /path/to/logs --local-embeddings

# Use Ollama embeddings
python index.py /path/to/logs --ollama-embeddings

# Use remote embedding server
python index.py /path/to/logs --remote-embeddings
```

Additional options:

```bash
# Custom model and chunk size
python index.py /path/to/logs --model custom-model --chunk-size 1500

# Custom ChromaDB path
python index.py /path/to/logs --chroma-path ./my_custom_db
```

### 4. Query Your Logs

Start the interactive query interface:

```bash
python ask.py
```

Or specify a custom output file:

```bash
python ask.py my_queries.md
```

## Architecture

### Core Components

1. **Unified Indexer (`index.py`)**
   - Processes repositories and creates vector embeddings
   - Supports multiple embedding strategies via handler classes
   - Chunks code into configurable segments (default: 2000 characters)
   - Stores embeddings in ChromaDB with metadata tracking

2. **Query Interface (`ask.py`)**  
   - Interactive CLI for natural language log queries
   - Auto-detects embedding type from metadata
   - Generates responses using Ollama's local LLM
   - Saves all Q&A pairs with timestamps

3. **Embedding Server (`embedding_server.py`)**
   - Optional remote embedding service with GPU support
   - RESTful API with health checks and server info
   - Configurable via command-line arguments
   - Supports batch processing and model caching

### Embedding Handlers

- **LocalEmbeddingHandler**: Uses SentenceTransformer with automatic GPU detection
- **OllamaEmbeddingHandler**: Leverages Ollama's embedding API
- **RemoteEmbeddingHandler**: Connects to remote embedding server with retry logic

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model for responses | `dolphincoder:15b` |
| `EMBEDDING_SERVER` | Remote embedding server URL | `http://localhost:5000` |
| `EMBEDDING_MODEL` | Embedding model name | `nomic-ai/nomic-embed-text-v1.5` |
| `CHROMA_PATH` | ChromaDB storage path | `./chroma_code` |
| `USE_LOCAL_EMBEDDINGS` | Default embedding strategy | `true` |
| `USE_LOCAL_OLLAMA` | Use local Ollama instance | `true` |

### Command Line Options

#### Indexing (`index.py`)

```bash
python index.py <repository> [options]

Options:
  --local-embeddings     Use local SentenceTransformer (default)
  --ollama-embeddings    Use Ollama embedding API  
  --remote-embeddings    Use remote embedding server
  --model MODEL          Override embedding model
  --chunk-size SIZE      Text chunk size (default: 2000)
  --chroma-path PATH     ChromaDB storage path
```

#### Embedding Server (`embedding_server.py`)

```bash
python embedding_server.py [options]

Options:
  --host HOST            Bind host (default: 0.0.0.0)
  --port PORT            Bind port (default: 5000)  
  --model MODEL          SentenceTransformer model
  --max-length LENGTH    Max sequence length (default: 512)
  --batch-size SIZE      Encoding batch size (default: 32)
  --debug                Enable debug mode
```

## Advanced Usage

### Remote Embedding Server

Start the embedding server for distributed setups:

```bash
# Start with GPU acceleration
python embedding_server.py --host 0.0.0.0 --port 5000

# Custom model and settings
python embedding_server.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --max-length 256 \
  --batch-size 64
```

### Multiple Repositories

Index multiple repositories to the same database:

```bash
# Index first repo
python index.py /path/to/logs1 --chroma-path ./shared_db

# Add second repo to same database  
python index.py /path/to/logs2 --chroma-path ./shared_db
```

### Ollama Integration

Ensure Ollama is running with required models:

```bash
# Install Ollama models
ollama pull dolphincoder:15b
ollama pull nomic-embed-text

# Start Ollama (usually runs as service)
ollama serve
```

## API Endpoints

The embedding server provides RESTful endpoints:

- `POST /embed` - Generate embeddings for text arrays
- `GET /health` - Health check with server status  
- `GET /info` - Detailed server and GPU information

Example request:

```bash
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["def hello_world():", "print(\"Hello, World!\")"]}'
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The system automatically falls back to CPU if GPU is unavailable
2. **Model Loading**: Ensure sufficient RAM/VRAM for embedding models
3. **Ollama Connection**: Verify Ollama is running and accessible at configured host
4. **ChromaDB Permissions**: Ensure write permissions for ChromaDB storage path

### Backward Compatibility

The system automatically detects and works with databases created by older versions:
- Checks multiple ChromaDB paths (`./chroma_ollama`, `./chroma_code_optimized`, etc.)
- Supports both "code" and "logs" collection names
- Gracefully handles missing metadata files

## Dependencies

- **chromadb**: Vector database for embeddings
- **sentence-transformers**: Local embedding generation
- **ollama**: LLM client for local inference
- **rich**: Enhanced terminal output and markdown rendering
- **flask**: Web server for embedding API
- **python-dotenv**: Environment configuration management

## File Structure

```
├── index.py              # Unified indexing script
├── ask.py                # Interactive query interface  
├── embedding_server.py   # Remote embedding server
├── requirements.txt      # Python dependencies
├── .env_example         # Environment configuration template
└── chroma_code/         # Default ChromaDB storage (created after indexing)
```

## License

This project is designed for local development and research use. Please ensure compliance with the terms of service for any external models or APIs used.

## Contributions

I welcome any assistance on this project, especially around trying new models for better performance and testing against ore logs than I have at my disposal!

Please just fork off of dev and then submit a PR