#!/usr/bin/env python
"""
Run the KITE Trading System Web Interface

This script starts the FastAPI server and serves the web interface
for configuring and monitoring trading strategies.
"""

import os
import sys
import argparse
import webbrowser
import threading
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the KITE Trading System Web Interface")
    
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to run the server on (default: 127.0.0.1)")
    
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on (default: 8000)")
    
    parser.add_argument("--no-open-browser", action="store_true",
                        help="Don't automatically open the web interface in a browser")
    
    return parser.parse_args()

def open_browser(host, port):
    """Open the web interface in a browser."""
    # Wait for the server to start
    time.sleep(2)
    
    # Open the web interface
    url = f"http://{host}:{port}/app"
    webbrowser.open(url)
    
    print(f"Web interface opened at {url}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("frontend", exist_ok=True)
    
    # Import uvicorn here to avoid import errors if it's not installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Please install it with 'pip install uvicorn'.")
        sys.exit(1)
    
    # Import FastAPI here to avoid import errors if it's not installed
    try:
        import fastapi
    except ImportError:
        print("Error: fastapi is not installed. Please install it with 'pip install fastapi'.")
        sys.exit(1)
    
    # Check if the frontend files exist
    frontend_dir = Path("frontend")
    if not (frontend_dir / "index.html").exists():
        print("Warning: Frontend files not found. The web interface may not work correctly.")
    
    # Open the web interface in a browser
    if not args.no_open_browser:
        threading.Thread(target=open_browser, args=(args.host, args.port), daemon=True).start()
    
    # Start the server
    print(f"Starting KITE Trading System Web Interface at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the server
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
