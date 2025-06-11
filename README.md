# Log File Analysis System

A powerful semantic search system for log files that enables natural language queries over your logs using vector embeddings and local LLM integration.

## Features

- **Multiple Embedding Strategies**: Choose between local SentenceTransformer, Ollama embeddings, or remote embedding server
- **Flexible Storage**: Uses ChromaDB for persistent vector storage with configurable paths
- **Local LLM Integration**: Generates AI responses using Ollama with customizable models
- **Interactive Query Interface**: Rich terminal interface with markdown rendering
- **GPU Acceleration**: Optional GPU support for faster embedding generation
- **Automatic File Detection**: Intelligently detects and indexes all text-based files by content analysis
- **Security-First Design**: Client-side trust_remote_code management with consent prompts and persistent tracking
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
cp .env.example .env
```

Edit `.env` to configure your setup:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=dolphincoder:15b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest

# Embedding Configuration  
EMBEDDING_SERVER=http://localhost:5000
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Storage Configuration
CHROMA_PATH=./chroma_db
DEFAULT_CHUNK_SIZE=2000

# Default Settings
USE_LOCAL_EMBEDDINGS=true
USE_LOCAL_OLLAMA=true
```

### 3. Index Your Files

Index a directory with automatic file detection:

```bash
python index.py /path/to/your/files
```

The system will:
- Automatically detect all text-based files by content analysis
- Skip binary files and common build/cache directories
- Prompt for trust_remote_code consent if needed for the embedding model

Or specify embedding type:

```bash
# Use local SentenceTransformer embeddings (default)
python index.py /path/to/files --local-embeddings

# Use Ollama embeddings
python index.py /path/to/files --ollama-embeddings

# Use remote embedding server
python index.py /path/to/files --remote-embeddings
```

Additional options:

```bash
# Custom model and chunk size
python index.py /path/to/logs --model custom-model --chunk-size 1500

# Custom ChromaDB path
python index.py /path/to/logs --chroma-path ./my_custom_db
```

### 4. Query Your Indexed Content

Start the interactive query interface:

```bash
python ask.py
```

The system will:
- Auto-detect the embedding type used during indexing
- Apply same trust_remote_code settings for consistency
- Generate responses using Ollama's local LLM
- Generate an output file ask-YYYY-Month-DD-HH-MM.md

Or specify a custom output file:

```bash
python ask.py my_queries.md
```

## Architecture

### Core Components

1. **Unified Indexer (`index.py`)**
   - Processes repositories with automatic file detection
   - Supports multiple embedding strategies via handler classes
   - Chunks content into configurable segments (configured via DEFAULT_CHUNK_SIZE)
   - Client-side trust_remote_code management
   - Stores embeddings in ChromaDB (collection: 'vectors') with metadata tracking

2. **Query Interface (`ask.py`)**  
   - Interactive CLI for natural language queries
   - Auto-detects embedding type and trust settings from metadata
   - Generates responses using Ollama's local LLM
   - Consistent security model with indexing phase
   - Saves all Q&A pairs with timestamps as markdown files

3. **Embedding Server (`embedding_server.py`)**
   - Optional remote embedding service with GPU support
   - Respects client-side trust_remote_code decisions
   - RESTful API with health checks and server info
   - Dynamic model loading with trust setting caching
   - Supports batch processing and multiple model variants

4. **Trust Manager (`trust_manager.py`)**
   - Centralized security management for trust_remote_code
   - Auto-detection of models requiring remote code execution
   - Interactive consent prompts with risk/benefit explanations
   - Persistent approval tracking in .env files
   - CLI tools for managing trust settings

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
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text:latest` |
| `EMBEDDING_SERVER` | Remote embedding server URL | `http://localhost:5000` |
| `EMBEDDING_MODEL` | Embedding model name | `nomic-ai/nomic-embed-text-v1.5` |
| `CHROMA_PATH` | ChromaDB storage path | `./chroma_db` |
| `DEFAULT_CHUNK_SIZE` | Default text chunk size | `2000` |
| `DEFAULT_OUTPUT_FILE_PREFIX` | Prefix for auto-generated output files | `ask` |
| `DEFAULT_TOP_K` | Default number of results to retrieve | `5` |
| `USE_LOCAL_EMBEDDINGS` | Default embedding strategy | `true` |
| `USE_LOCAL_OLLAMA` | Use local Ollama instance | `true` |
| `TRUST_REMOTE_CODE_*` | Model-specific trust settings | Auto-managed |

### Command Line Options

#### Indexing (`index.py`)

```bash
python index.py <repository> [options]

Options:
  --local-embeddings     Use local SentenceTransformer (default)
  --ollama-embeddings    Use Ollama embedding API  
  --remote-embeddings    Use remote embedding server
  --model MODEL          Override embedding model
  --chunk-size SIZE      Text chunk size (default from DEFAULT_CHUNK_SIZE env var)
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

## Security: Trust Remote Code Management

The system includes a comprehensive security framework for models that require `trust_remote_code=True`. This client-side security system:

- **Auto-detects** which models likely need remote code execution based on known patterns
- **Prompts for informed consent** with detailed security warnings
- **Persists decisions** in `.env` with model-specific hash tracking
- **Client-side control** - trust decisions made locally, not on remote servers
- **Cross-component consistency** - same security model for indexing, querying, and serving

### How It Works

1. **Detection**: System analyzes model names against known patterns
2. **User Consent**: Interactive prompts with clear risk/benefit explanations  
3. **Persistence**: Decisions saved locally with model identification hashes
4. **Communication**: Client sends trust settings to remote embedding servers

### Managing Trust Settings

```bash
# List all approved/denied models
python trust_manager.py --list

# Check if a specific model needs trust_remote_code
python trust_manager.py --check "nomic-ai/nomic-embed-text-v1.5"
```

### Security Flow

When you first use a model requiring remote code execution:

```
==============================================================
SECURITY WARNING: Remote Code Execution
==============================================================
Model: nomic-ai/nomic-embed-text-v1.5

This model may require 'trust_remote_code=True' which allows
the model to execute arbitrary code during loading.

RISKS:
- The model could execute malicious code
- Your system could be compromised
- Data could be stolen or corrupted

BENEFITS:
- Access to newer/specialized models
- Better embedding quality for some models

Your choice will be saved for this model.
==============================================================
Allow remote code execution for this model? [y/N]:
```

### Trust Settings Storage

Approval decisions are stored in your `.env` file:

```bash
# Example entries (automatically managed)
# TRUST_REMOTE_CODE_A1B2C3D4_MODEL=nomic-ai/nomic-embed-text-v1.5
TRUST_REMOTE_CODE_A1B2C3D4=true

# TRUST_REMOTE_CODE_E5F6G7H8_MODEL=sentence-transformers/all-MiniLM-L6-v2  
TRUST_REMOTE_CODE_E5F6G7H8=false
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

The system uses a single standardized configuration:
- Database path: `./chroma_db` (configurable via CHROMA_PATH)
- Collection name: `vectors`
- Metadata tracking for indexing configuration

## Dependencies

- **chromadb**: Vector database for embeddings
- **sentence-transformers**: Local embedding generation (optional, only needed for local embeddings)
- **ollama**: LLM client for local inference
- **rich**: Enhanced terminal output and markdown rendering
- **flask**: Web server for embedding API
- **python-dotenv**: Environment configuration management
- **tiktoken**: Token counting utilities
- **einops**: Tensor operations for advanced models
- **requests**: HTTP client for remote services

## File Structure

```
├── index.py              # Unified indexing script
├── ask.py                # Interactive query interface  
├── embedding_server.py   # Remote embedding server
├── trust_manager.py      # Security: trust_remote_code management
├── requirements.txt      # Python dependencies
├── .env_example         # Environment configuration template
└── chroma_db/           # Default ChromaDB storage (created after indexing)
```

## License

This project is designed for local development and research use. Please ensure compliance with the terms of service for any external models or APIs used.

## Contributions

I welcome any assistance on this project, especially around trying new models for better performance and testing against ore logs than I have at my disposal!

Please just fork off of dev and then submit a PR