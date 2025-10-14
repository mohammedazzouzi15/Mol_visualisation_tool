#!/usr/bin/env python3
"""
Launch script for the Lightweight Molecular Visualization App.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        print("✓ Core dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False


def main():
    """Launch the Streamlit application."""
    print("🧬 Launching Lightweight Molecular Visualization App")
    print("=" * 60)
    
    # Check if we're in the right directory
    app_file = Path("app.py")
    if not app_file.exists():
        print("✗ app.py not found. Please run from the lite_viz_app directory.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Launch the app
    print("🚀 Starting Streamlit application...")
    print("📂 App will open in your default browser")
    print("🛑 Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"✗ Error launching app: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())