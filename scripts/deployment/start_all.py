"""
Start All Services (MLflow UI + Dashboard + API).

This script launches all three services in separate processes.

Usage:
    poetry run python scripts/deployment/start_all.py
"""
import subprocess
import sys
import time
from pathlib import Path
import webbrowser

PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    """Start all services."""
    print("=" * 80)
    print("STARTING ALL CREDIT SCORING SERVICES")
    print("=" * 80)
    print()
    print("This will start:")
    print("  1. MLflow UI      → http://localhost:5000")
    print("  2. Dashboard      → http://localhost:8501")
    print("  3. API Server     → http://localhost:8000")
    print()
    print("Opening browsers in 5 seconds...")
    print("Press Ctrl+C to stop all services")
    print("=" * 80)
    print()

    processes = []

    try:
        # 1. Start MLflow UI
        print("[1/3] Starting MLflow UI...")
        mlflow_cmd = [
            "poetry", "run", "mlflow", "ui",
            "--backend-store-uri", "sqlite:///mlruns/mlflow.db",
            "--host", "0.0.0.0",
            "--port", "5000"
        ]
        mlflow_process = subprocess.Popen(
            mlflow_cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(("MLflow UI", mlflow_process))
        time.sleep(2)

        # 2. Start Dashboard
        print("[2/3] Starting Streamlit Dashboard...")
        dashboard_cmd = [
            "poetry", "run", "streamlit", "run",
            "scripts/deployment/dashboard.py",
            "--server.port", "8501"
        ]
        dashboard_process = subprocess.Popen(
            dashboard_cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(("Dashboard", dashboard_process))
        time.sleep(2)

        # 3. Start API
        print("[3/3] Starting API Server...")
        api_cmd = [
            "poetry", "run", "uvicorn",
            "api.app:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        api_process = subprocess.Popen(
            api_cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(("API Server", api_process))
        time.sleep(3)

        print()
        print("=" * 80)
        print("ALL SERVICES STARTED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Access services at:")
        print("  MLflow UI:  http://localhost:5000")
        print("  Dashboard:  http://localhost:8501")
        print("  API Docs:   http://localhost:8000/docs")
        print()
        print("Opening browsers...")
        print()

        # Open browsers
        time.sleep(2)
        webbrowser.open("http://localhost:5000")
        time.sleep(1)
        webbrowser.open("http://localhost:8501")
        time.sleep(1)
        webbrowser.open("http://localhost:8000/docs")

        print("Press Ctrl+C to stop all services")
        print("=" * 80)

        # Wait indefinitely
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping all services...")
        for name, process in processes:
            print(f"  Stopping {name}...")
            process.terminate()

        # Wait for processes to terminate
        time.sleep(2)
        for name, process in processes:
            if process.poll() is None:
                print(f"  Force killing {name}...")
                process.kill()

        print("\nAll services stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
