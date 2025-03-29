#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightRAG API Client Example

This script demonstrates how to interact with a running LightRAG API server
through HTTP requests. It provides examples for common operations like
document upload, indexing, and querying through the REST API.
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any, Tuple

# Load environment variables from .env file
load_dotenv()

# Default LightRAG server URL
DEFAULT_SERVER_URL = "http://localhost:9621"


class LightRagClient:
    """Client for interacting with the LightRAG API server"""
    
    def __init__(self, server_url: str = DEFAULT_SERVER_URL, api_key: Optional[str] = None):
        """
        Initialize the LightRAG API client
        
        Args:
            server_url: URL of the LightRAG server
            api_key: Optional API key for authentication
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.token = None
        
        # Set up headers
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def login(self, username: str = None, password: str = None) -> bool:
        """
        Login to the LightRAG server to get JWT token for authentication
        
        Returns:
            bool: True if login successful, False otherwise
        """
        # Check authentication status
        auth_status = self.get_auth_status()
        
        if not auth_status.get("auth_configured", False):
            # Authentication not configured, use guest token
            self.token = auth_status.get("access_token")
            print("Authentication not configured. Using guest access.")
            return True
        
        # If username/password not provided, try environment variables
        if not username:
            username = os.getenv("LIGHTRAG_USER")
        if not password:
            password = os.getenv("LIGHTRAG_PASSWORD")
        
        if not username or not password:
            print("Error: Username and password required for login")
            return False
        
        try:
            # Perform login request
            response = requests.post(
                f"{self.server_url}/login",
                data={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                print(f"Login successful for user {username}")
                return True
            else:
                print(f"Login failed: {response.text}")
                return False
        
        except Exception as e:
            print(f"Error during login: {str(e)}")
            return False
    
    def get_auth_status(self) -> Dict:
        """Get authentication status from the server"""
        try:
            response = requests.get(f"{self.server_url}/auth-status")
            return response.json()
        except Exception as e:
            print(f"Error getting auth status: {str(e)}")
            return {"auth_configured": False}
    
    def get_headers(self) -> Dict:
        """Get headers with authentication token if available"""
        headers = self.headers.copy()
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def check_health(self) -> Dict:
        """Check server health and configuration"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                headers=self.get_headers()
            )
            return response.json()
        except Exception as e:
            print(f"Error checking server health: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def query(self, query_text: str, mode: str = "hybrid", stream: bool = False) -> Dict:
        """
        Query the LightRAG server
        
        Args:
            query_text: The query text
            mode: Query mode (hybrid, local, global, naive, mix)
            stream: Whether to use streaming response
        
        Returns:
            Dict: Query response
        """
        endpoint = f"{self.server_url}/query/stream" if stream else f"{self.server_url}/query"
        payload = {
            "query": query_text,
            "mode": mode
        }
        
        try:
            if stream:
                # Handle streaming response
                with requests.post(
                    endpoint,
                    json=payload,
                    headers=self.get_headers(),
                    stream=True
                ) as response:
                    response.raise_for_status()
                    print("\nStreaming response:")
                    print("-" * 80)
                    full_response = ""
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            print(chunk, end="", flush=True)
                            full_response += chunk
                    print("\n" + "-" * 80)
                    return {"response": full_response}
            else:
                # Handle regular response
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers=self.get_headers()
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return {"error": str(e)}
    
    def add_text(self, text: str, description: str = None) -> Dict:
        """
        Add text content to the LightRAG server
        
        Args:
            text: Text content to add
            description: Optional description for the document
        
        Returns:
            Dict: Response from the server
        """
        payload = {
            "text": text,
            "description": description or "Uploaded text"
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/documents/text",
                json=payload,
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"Error adding text: {str(e)}")
            return {"error": str(e)}
    
    def upload_file(self, file_path: str, description: str = None) -> Dict:
        """
        Upload a file to the LightRAG server
        
        Args:
            file_path: Path to the file
            description: Optional description for the document
        
        Returns:
            Dict: Response from the server
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f)}
                data = {}
                if description:
                    data["description"] = description
                
                response = requests.post(
                    f"{self.server_url}/documents/file",
                    files=files,
                    data=data,
                    headers=self.get_headers()
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return {"error": str(e)}
    
    def upload_batch_files(self, file_paths: List[str]) -> Dict:
        """
        Upload multiple files to the LightRAG server
        
        Args:
            file_paths: List of file paths to upload
        
        Returns:
            Dict: Response from the server
        """
        files = []
        
        try:
            # Prepare files for upload
            file_handles = []
            for path in file_paths:
                path = Path(path)
                if not path.exists():
                    print(f"Warning: File not found: {path}")
                    continue
                
                f = open(path, "rb")
                file_handles.append(f)
                files.append(("files", (path.name, f)))
            
            if not files:
                return {"error": "No valid files to upload"}
            
            # Upload files
            response = requests.post(
                f"{self.server_url}/documents/batch",
                files=files,
                headers=self.get_headers()
            )
            
            # Close file handles
            for f in file_handles:
                f.close()
            
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            # Ensure all file handles are closed
            for f in file_handles:
                f.close()
            
            print(f"Error uploading batch files: {str(e)}")
            return {"error": str(e)}
    
    def scan_documents(self, timeout: int = 1800) -> Dict:
        """
        Trigger document scanning on the server
        
        Args:
            timeout: Request timeout in seconds
        
        Returns:
            Dict: Response from the server
        """
        try:
            response = requests.post(
                f"{self.server_url}/documents/scan",
                headers=self.get_headers(),
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"Error scanning documents: {str(e)}")
            return {"error": str(e)}
    
    def clear_documents(self) -> Dict:
        """
        Clear all documents from the LightRAG server
        
        Returns:
            Dict: Response from the server
        """
        try:
            response = requests.delete(
                f"{self.server_url}/documents",
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"Error clearing documents: {str(e)}")
            return {"error": str(e)}


def print_response(data: Dict, title: str = "Response"):
    """Pretty print response data"""
    print(f"\n{title}:")
    print("-" * 80)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="LightRAG API Client Example")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="LightRAG server URL")
    parser.add_argument("--key", help="API key for authentication")
    parser.add_argument("--user", help="Username for login")
    parser.add_argument("--password", help="Password for login")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Health check command
    subparsers.add_parser("health", help="Check server health")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the LightRAG server")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--mode", choices=["hybrid", "local", "global", "naive", "mix"], 
                             default="hybrid", help="Query mode")
    query_parser.add_argument("--stream", action="store_true", help="Use streaming response")
    
    # Upload text command
    text_parser = subparsers.add_parser("add-text", help="Add text to the LightRAG server")
    text_parser.add_argument("text", help="Text content")
    text_parser.add_argument("--description", help="Document description")
    
    # Upload file command
    file_parser = subparsers.add_parser("upload-file", help="Upload a file to the LightRAG server")
    file_parser.add_argument("file", help="Path to the file")
    file_parser.add_argument("--description", help="Document description")
    
    # Upload batch files command
    batch_parser = subparsers.add_parser("upload-batch", help="Upload multiple files to the LightRAG server")
    batch_parser.add_argument("files", nargs="+", help="Paths to the files")
    
    # Scan documents command
    scan_parser = subparsers.add_parser("scan", help="Trigger document scanning")
    scan_parser.add_argument("--timeout", type=int, default=1800, help="Request timeout in seconds")
    
    # Clear documents command
    subparsers.add_parser("clear", help="Clear all documents")
    
    args = parser.parse_args()
    
    # Create client
    client = LightRagClient(server_url=args.server, api_key=args.key)
    
    # Login if authentication is configured
    if args.user or args.password or os.getenv("LIGHTRAG_USER") or os.getenv("LIGHTRAG_PASSWORD"):
        if not client.login(args.user, args.password):
            print("Login failed. Exiting.")
            return 1
    
    # Execute command
    if args.command == "health" or not args.command:
        health_data = client.check_health()
        print_response(health_data, "Server Health")
        
        if health_data.get("status") != "healthy":
            print("Server is not healthy!")
            return 1
        
        print("Server is healthy and ready to use!")
        
        # Show configuration info
        config = health_data.get("configuration", {})
        print("\nServer Configuration:")
        print(f"LLM: {config.get('llm_binding')} ({config.get('llm_model')})")
        print(f"Embedding: {config.get('embedding_binding')} ({config.get('embedding_model')})")
        print(f"Working directory: {health_data.get('working_directory')}")
        print(f"Input directory: {health_data.get('input_directory')}")
        print(f"Auth mode: {health_data.get('auth_mode', 'unknown')}")
        print(f"API version: {health_data.get('api_version')}")
        print(f"Core version: {health_data.get('core_version')}")
    
    elif args.command == "query":
        result = client.query(args.text, mode=args.mode, stream=args.stream)
        if not args.stream:
            print_response(result, f"{args.mode.capitalize()} Query Result")
    
    elif args.command == "add-text":
        result = client.add_text(args.text, description=args.description)
        print_response(result, "Text Upload Result")
    
    elif args.command == "upload-file":
        result = client.upload_file(args.file, description=args.description)
        print_response(result, "File Upload Result")
    
    elif args.command == "upload-batch":
        result = client.upload_batch_files(args.files)
        print_response(result, "Batch Upload Result")
    
    elif args.command == "scan":
        print("Scanning documents... (this may take a while)")
        result = client.scan_documents(timeout=args.timeout)
        print_response(result, "Document Scan Result")
    
    elif args.command == "clear":
        # Ask for confirmation
        confirm = input("Are you sure you want to clear all documents? (y/N): ")
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return 0
        
        result = client.clear_documents()
        print_response(result, "Clear Documents Result")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())