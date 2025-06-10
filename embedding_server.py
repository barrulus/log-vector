#!/usr/bin/env python3
"""
Embedding Server with GPU Support

A Flask server that provides embeddings using SentenceTransformer models.
Supports GPU acceleration when available.

Usage:
    python embedding_server.py [options]

Options:
    --host HOST           Host to bind to (default: 0.0.0.0)
    --port PORT           Port to bind to (default: 5000)
    --model MODEL         SentenceTransformer model to use (default: nomic-ai/nomic-embed-text-v1.5)
    --max-length LENGTH   Maximum sequence length (default: 512)
    --batch-size SIZE     Batch size for encoding (default: 32)
"""

import argparse
import os
import logging
from typing import Any, Dict, Optional, cast
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for model
model: Optional[SentenceTransformer] = None
device: Optional[str] = None
args: Optional[argparse.Namespace] = None
model_cache: Dict[str, SentenceTransformer] = {}  # Cache models with different trust settings


def initialize_model() -> None:
    """Initialize the SentenceTransformer model"""
    global model, device
    
    if args is None:
        raise RuntimeError("Arguments not initialized")
    
    print(f"\nLoading SentenceTransformer model: {args.model}")
    
    # GPU Acceleration Diagnostics
    print("\nGPU Acceleration Diagnostics:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check for MPS (Apple Silicon GPU)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon) available: {mps_available}")
    
    if torch.cuda.is_available():
        if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') and torch.version.cuda:  # type: ignore[attr-defined]
            print(f"PyTorch CUDA version: {torch.version.cuda}")  # type: ignore[attr-defined]
        print(f"Number of CUDA GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA GPU {i}: {torch.cuda.get_device_name(i)}")
            props = cast(Any, torch.cuda.get_device_properties(i))  # type: ignore[misc]
            print(f"CUDA GPU {i} Memory: {props.total_memory / 1024**3:.1f} GB")
        device = 'cuda'
    elif mps_available:
        print("Using Apple Silicon GPU (MPS) for acceleration")
        device = 'mps'
    else:
        print("No GPU acceleration available. Using CPU.")
        print("Possible reasons:")
        print("- No NVIDIA GPU (for CUDA)")
        print("- No Apple Silicon chip (for MPS)")
        print("- PyTorch not compiled with GPU support")
        print("- Missing drivers or environment setup")
        device = 'cpu'
    
    print(f"\nUsing device: {device}")
    
    # Determine trust setting for default model
    if args.trust_remote_code:
        default_trust = True
        print(f"Using --trust-remote-code flag: trust_remote_code=True for {args.model}")
    else:
        from trust_manager import TrustManager
        trust_manager = TrustManager()
        default_trust = trust_manager.check_model_needs_trust(args.model)
        print(f"Auto-detected trust_remote_code={default_trust} for {args.model}")
    
    # Load default model with appropriate trust setting
    model = get_or_load_model(args.model, default_trust)
    
    print(f"Default model loaded: {args.model}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Note: Client requests can override model and trust settings")


def get_or_load_model(model_name: str, trust_remote_code: bool) -> SentenceTransformer:
    """Get or load a model with specific trust_remote_code setting"""
    global model_cache, device, args
    from sentence_transformers import SentenceTransformer
    
    if args is None:
        raise RuntimeError("Server not initialized")
    
    # Create cache key
    cache_key = f"{model_name}:trust={trust_remote_code}"
    
    if cache_key not in model_cache:
        print(f"Loading model {model_name} with trust_remote_code={trust_remote_code}")
        loaded_model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)
        loaded_model.max_seq_length = args.max_length
        model_cache[cache_key] = loaded_model
    
    return model_cache[cache_key]


@app.route('/embed', methods=['POST'])
def embed() -> Any:
    """Generate embeddings for provided texts"""
    try:
        if args is None:
            return jsonify({'error': 'Server not initialized'}), 500
            
        data = request.json
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        data = cast(Dict[str, Any], data)
            
        texts = data.get('texts', [])
        request_model = data.get('model', args.model)
        trust_remote_code = data.get('trust_remote_code', False)
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        # Get the appropriate model
        model_to_use = get_or_load_model(request_model, trust_remote_code)
        
        # Generate embeddings on GPU/CPU
        embeddings_result = cast(Any, model_to_use.encode(  # type: ignore[misc]
            texts, 
            batch_size=args.batch_size, 
            show_progress_bar=False,
            convert_to_numpy=True
        ))
        embeddings = embeddings_result.tolist()
        
        return jsonify({
            'embeddings': embeddings,
            'model': request_model,
            'trust_remote_code': trust_remote_code,
            'count': len(embeddings)
        })
    
    except Exception as e:
        # Log the full error details for debugging
        logging.error(f"Error in /embed endpoint: {str(e)}", exc_info=True)
        # Return generic error message to client
        return jsonify({'error': 'An internal server error occurred'}), 500


@app.route('/health', methods=['GET'])
def health() -> Any:
    """Health check endpoint"""
    if args is None:
        return jsonify({'error': 'Server not initialized'}), 500
        
    return jsonify({
        'status': 'healthy',
        'device': device,
        'model': args.model,
        'max_sequence_length': args.max_length
    })


@app.route('/info', methods=['GET'])
def info() -> Any:
    """Get server information"""
    if args is None:
        return jsonify({'error': 'Server not initialized'}), 500
        
    gpu_info: Dict[str, Dict[str, str]] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = cast(Any, torch.cuda.get_device_properties(i))  # type: ignore[misc]
            gpu_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memory_gb": f"{props.total_memory / 1024**3:.1f}"
            }
    
    return jsonify({
        'server': 'SentenceTransformer Embedding Server',
        'version': '1.0',
        'model': args.model,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') else None,  # type: ignore[attr-defined]
        'gpus': gpu_info,
        'max_sequence_length': args.max_length,
        'batch_size': args.batch_size
    })


def main() -> None:
    global args
    
    parser = argparse.ArgumentParser(
        description='Embedding server using SentenceTransformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=os.getenv('EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5'),
        help='SentenceTransformer model to use'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Force trust_remote_code=True for default model (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Initialize model
    initialize_model()
    
    # Run server
    print(f"\nStarting embedding server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"\nEndpoints:")
    print(f"  POST /embed  - Generate embeddings")
    print(f"  GET  /health - Health check")
    print(f"  GET  /info   - Server information")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()