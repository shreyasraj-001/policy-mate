#!/usr/bin/env python
"""
Uvicorn server script for running the Policy RAG API with multiple workers.
This script launches the FastAPI application using Uvicorn for Windows compatibility.

Usage:
    python run_server.py

Options:
    --workers N       Number of worker processes (default: 10)
    --port PORT       Port to bind to (default: 8000)
    --host HOST       Host address to bind to (default: 0.0.0.0)
    --log-level LEVEL Log level (default: info)
"""

import argparse
import sys
import os
import multiprocessing
import uvicorn
import signal
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor

def run_uvicorn_server(host, port, log_level, worker_id):
    """Run a single Uvicorn worker process"""
    # Adjust port for each worker to avoid conflicts
    worker_port = port + worker_id
    print(f"🚀 Starting worker {worker_id} on {host}:{worker_port}")
    
    try:
        # Run uvicorn server
        uvicorn.run(
            "main:app",
            host=host,
            port=worker_port,
            log_level=log_level
        )
    except KeyboardInterrupt:
        print(f"⛔ Worker {worker_id} stopped")
    except Exception as e:
        print(f"❌ Error in worker {worker_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run the Policy RAG API using Uvicorn workers")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of worker processes (default: 10)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Starting port to bind to (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--log-level", type=str, default="info",
                        help="Log level (default: info)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.workers} Uvicorn workers")
    print(f"🌐 Each worker will run on a separate port starting from {args.port}")
    print(f"💡 First worker: {args.host}:{args.port}")
    print(f"💡 Last worker: {args.host}:{args.port + args.workers - 1}")
    
    # Set up a load balancer using nginx or a similar tool to distribute requests across worker ports
    
    try:
        # Create a process pool to manage worker processes
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Start worker processes
            futures = []
            for i in range(args.workers):
                future = executor.submit(
                    run_uvicorn_server,
                    args.host,
                    args.port,
                    args.log_level,
                    i
                )
                futures.append(future)
            
            # Wait for all processes to complete
            for future in futures:
                future.result()
                
    except KeyboardInterrupt:
        print("\n⛔ Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    sys.exit(main())
