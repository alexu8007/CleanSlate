#!/usr/bin/env python
"""
Script to launch the CleanSlate web UI.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get the absolute path to the cleanslate package
CLEANSLATE_DIR = Path(__file__).resolve().parent.parent.parent


def run_ui(port=8501, debug=False):
    """
    Run the CleanSlate web UI.
    
    Args:
        port: Port to run the web UI on.
        debug: Whether to run in debug mode.
    """
    # Add cleanslate to the Python path
    sys.path.insert(0, str(CLEANSLATE_DIR))
    
    # Get the path to the app.py file
    app_path = Path(__file__).resolve().parent / "app.py"
    
    # Run streamlit
    cmd = [
        "streamlit", "run", 
        str(app_path),
        "--server.port", str(port)
    ]
    
    if debug:
        cmd.append("--logger.level=debug")
        os.environ["CLEANSLATE_DEBUG"] = "1"
    
    print(f"Starting CleanSlate UI on port {port}...")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CleanSlate web UI.")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the web UI on.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    
    args = parser.parse_args()
    
    run_ui(port=args.port, debug=args.debug) 