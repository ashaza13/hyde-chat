#!/usr/bin/env python3
"""
Script to run the Streamlit GUI for audit question processing.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    try:
        # Change to the directory containing the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nShutting down Streamlit app...")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 