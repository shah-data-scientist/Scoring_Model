"""
Start the Credit Scoring API Server.

Usage:
    poetry run python scripts/deployment/start_api.py
"""
import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    """Start FastAPI server."""
    print("=" * 80)
    print("STARTING CREDIT SCORING API SERVER")
    print("=" * 80)
    print()
    print("API Documentation will be available at:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc:      http://localhost:8000/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()

    # Start uvicorn
    cmd = [
        "poetry", "run", "uvicorn",
        "api.app:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
