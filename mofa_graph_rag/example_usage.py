#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightRAG Example Usage Script

This script demonstrates basic usage of LightRAG for document indexing and RAG queries.
It provides examples for working with different backends (Ollama, OpenAI, Azure OpenAI).
"""

import os
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import LightRAG core
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, set_verbose_debug

# Choose LLM and embedding implementations based on backend
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed

# Load environment variables from .env file
load_dotenv()


async def index_document(rag, file_path):
    """Index a document file using LightRAG"""
    print(f"Indexing document: {file_path}")
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Index the document
    doc_id = await rag.add_document(text, description=Path(file_path).name)
    print(f"Document indexed with ID: {doc_id}")
    return doc_id


async def run_query(rag, query, mode="hybrid", top_k=50):
    """Run a query using LightRAG with specified mode"""
    print(f"Running {mode} query: {query}")
    
    if mode == "local":
        result = await rag.query_local_kg(query, top_k=top_k)
    elif mode == "global":
        result = await rag.query_global_kg(query, top_k=top_k)
    elif mode == "naive":
        result = await rag.query_naive(query, top_k=top_k)
    elif mode == "mix":
        result = await rag.query_mix(query, top_k=top_k)
    else:  # Default to hybrid
        result = await rag.query_hybrid(query, top_k=top_k)
    
    print("\nLightRAG Response:")
    print("-" * 80)
    print(result.response)
    print("-" * 80)
    print(f"Processed in {result.process_time:.2f} seconds")
    
    # Print sources if available
    if hasattr(result, 'sources') and result.sources:
        print("\nSources:")
        for i, source in enumerate(result.sources, 1):
            print(f"{i}. {source.get('description', 'Unknown')} (score: {source.get('score', 'N/A'):.2f})")
    
    return result


async def stream_query(rag, query, mode="hybrid", top_k=50):
    """Run a streaming query using LightRAG with specified mode"""
    print(f"Running streaming {mode} query: {query}")
    
    if mode == "local":
        stream = rag.query_local_kg_stream(query, top_k=top_k)
    elif mode == "global":
        stream = rag.query_global_kg_stream(query, top_k=top_k)
    elif mode == "naive":
        stream = rag.query_naive_stream(query, top_k=top_k)
    elif mode == "mix":
        stream = rag.query_mix_stream(query, top_k=top_k)
    else:  # Default to hybrid
        stream = rag.query_hybrid_stream(query, top_k=top_k)
    
    print("\nLightRAG Streaming Response:")
    print("-" * 80)
    
    # For streaming output
    full_response = ""
    async for token in stream:
        print(token, end="", flush=True)
        full_response += token
    
    print("\n" + "-" * 80)
    return full_response


async def visualize_knowledge_graph(rag, output_path="kg_visualization.html"):
    """Generate a visualization of the knowledge graph"""
    from lightrag.tools.lightrag_visualizer import visualize_graph
    
    print(f"Generating knowledge graph visualization to {output_path}...")
    await visualize_graph(rag, output_path)
    print(f"Knowledge graph visualization saved to {output_path}")


async def setup_rag(backend="ollama", working_dir="./rag_storage"):
    """Set up and initialize a LightRAG instance with specified backend"""
    # Create working directory if it doesn't exist
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure based on backend
    if backend == "ollama":
        # Ollama configuration
        llm_model_func = ollama_model_complete
        llm_model_name = os.getenv("LLM_MODEL", "mistral:latest")
        llm_host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
        
        embedding_func = EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
                api_key=os.getenv("EMBEDDING_BINDING_API_KEY")
            )
        )
        
        llm_kwargs = {
            "host": llm_host,
            "timeout": int(os.getenv("TIMEOUT", "150")),
            "options": {"num_ctx": int(os.getenv("MAX_TOKENS", "8192"))},
            "api_key": os.getenv("LLM_BINDING_API_KEY")
        }
        
    elif backend == "openai":
        # OpenAI configuration
        llm_model_func = openai_complete_if_cache
        llm_model_name = os.getenv("LLM_MODEL", "gpt-4o")
        
        embedding_func = EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8191")),
            func=lambda texts: openai_embed(
                texts,
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                base_url=os.getenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )
        
        llm_kwargs = {
            "base_url": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "timeout": int(os.getenv("TIMEOUT", "150")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7"))
        }
        
    elif backend == "azure_openai":
        # Azure OpenAI configuration
        llm_model_func = azure_openai_complete_if_cache
        llm_model_name = os.getenv("LLM_MODEL", "your-model-deployment-name")
        
        embedding_func = EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8191")),
            func=lambda texts: azure_openai_embed(
                texts,
                model=os.getenv("EMBEDDING_MODEL", "your-embedding-model"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        )
        
        llm_kwargs = {
            "base_url": os.getenv("LLM_BINDING_HOST"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            "timeout": int(os.getenv("TIMEOUT", "150")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7"))
        }
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    # Initialize RAG system
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=llm_model_name,
        llm_model_max_async=int(os.getenv("MAX_ASYNC", "4")),
        llm_model_max_token_size=int(os.getenv("MAX_TOKENS", "8192")),
        chunk_token_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP_SIZE", "128")),
        llm_model_kwargs=llm_kwargs,
        embedding_func=embedding_func,
        enable_llm_cache_for_entity_extract=os.getenv("ENABLE_LLM_CACHE_FOR_EXTRACT", "true").lower() == "true",
        embedding_cache_config={
            "enabled": True,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        },
        auto_manage_storages_states=True,
        max_parallel_insert=int(os.getenv("MAX_PARALLEL_INSERT", "2"))
    )
    
    # Initialize storages
    await rag.initialize_storages()
    
    return rag


async def main():
    parser = argparse.ArgumentParser(description="LightRAG Example Usage")
    parser.add_argument("--backend", choices=["ollama", "openai", "azure_openai"], default="ollama",
                        help="Backend to use (ollama, openai, azure_openai)")
    parser.add_argument("--working-dir", default="./rag_storage", help="Working directory for LightRAG storage")
    parser.add_argument("--index", help="Path to document to index")
    parser.add_argument("--query", help="Query to run")
    parser.add_argument("--mode", choices=["local", "global", "hybrid", "naive", "mix"], default="hybrid",
                        help="Query mode to use")
    parser.add_argument("--stream", action="store_true", help="Use streaming response")
    parser.add_argument("--visualize", action="store_true", help="Generate knowledge graph visualization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debugging")
    args = parser.parse_args()
    
    # Set verbose debugging if requested
    if args.verbose:
        set_verbose_debug(True)
    
    # Initialize RAG
    print(f"Initializing LightRAG with {args.backend} backend...")
    rag = await setup_rag(backend=args.backend, working_dir=args.working_dir)
    
    try:
        # Index document if specified
        if args.index:
            await index_document(rag, args.index)
        
        # Run query if specified
        if args.query:
            if args.stream:
                await stream_query(rag, args.query, mode=args.mode)
            else:
                await run_query(rag, args.query, mode=args.mode)
        
        # Generate visualization if requested
        if args.visualize:
            await visualize_knowledge_graph(rag)
        
        # If no specific operation requested, show example usage
        if not any([args.index, args.query, args.visualize]):
            print("\nExample usage:")
            print("1. Index a document:    python example_usage.py --backend ollama --index path/to/document.txt")
            print("2. Run a query:         python example_usage.py --backend ollama --query \"What is LightRAG?\"")
            print("3. Stream a query:      python example_usage.py --backend ollama --query \"Tell me about RAG systems\" --stream")
            print("4. Visualize KG:        python example_usage.py --backend ollama --visualize")
            print("\nFor all options:       python example_usage.py --help")
    
    finally:
        # Clean up resources
        await rag.finalize_storages()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())