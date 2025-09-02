#!/usr/bin/env python3
"""
Launch script for TrashNet Streamlit application.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Launch TrashNet Streamlit app')
    parser.add_argument('--port', type=int, default=8501, 
                       help='Port to run the app on')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to run the app on')
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode with auto-reload')
    
    args = parser.parse_args()
    
    # Get app path
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    if args.dev:
        cmd.extend(["--server.runOnSave", "true"])
    
    print(f"🚀 Starting TrashNet Streamlit app...")
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")


if __name__ == "__main__":
    main()