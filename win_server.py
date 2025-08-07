#!/usr/bin/env python
"""
Windows-compatible server script for running multiple instances of the Policy RAG API.
This script launches multiple FastAPI instances using uvicorn directly.

Usage:
    python win_server.py

Options:
    --workers N       Number of worker processes (default: 10)
    --base-port PORT  Base port number to start from (default: 8000)
    --host HOST       Host address to bind to (default: 0.0.0.0)
    --log-level LEVEL Log level (default: info)
"""

import argparse
import sys
import os
import subprocess
import time
import signal
import threading

def run_uvicorn_instance(host, port, log_level):
    """Run a single Uvicorn instance"""
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "main:app",
        f"--host={host}",
        f"--port={port}",
        f"--log-level={log_level}"
    ]
    
    print(f"🚀 Starting worker on {host}:{port}")
    
    # Start the process and return it
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

def log_reader(process, worker_id):
    """Read and print logs from worker process"""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"[Worker {worker_id}] {line.strip()}")

def main():
    parser = argparse.ArgumentParser(description="Run multiple instances of the Policy RAG API")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of worker processes (default: 10)")
    parser.add_argument("--base-port", type=int, default=8000,
                        help="Base port to start from (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--log-level", type=str, default="info",
                        help="Log level (default: info)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.workers} Uvicorn instances")
    print(f"🌐 Workers will run on ports {args.base_port} through {args.base_port + args.workers - 1}")
    print(f"💡 Configure a load balancer to distribute traffic across these ports")
    print(f"💡 Or use Windows Network Load Balancing (NLB) feature")
    
    # Start worker processes
    processes = []
    log_threads = []
    
    try:
        for i in range(args.workers):
            port = args.base_port + i
            process = run_uvicorn_instance(args.host, port, args.log_level)
            processes.append(process)
            
            # Create a thread to read and print logs
            log_thread = threading.Thread(
                target=log_reader,
                args=(process, i),
                daemon=True
            )
            log_thread.start()
            log_threads.append(log_thread)
            
            # Small delay to prevent port conflicts during startup
            time.sleep(0.5)
        
        print(f"✅ All {args.workers} workers started successfully")
        
        # Wait for all processes to complete
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⛔ Stopping all servers...")
        for process in processes:
            process.terminate()
        
        # Wait for processes to terminate
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✅ All servers stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        for process in processes:
            process.terminate()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
