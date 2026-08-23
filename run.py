import subprocess
import sys
import time
import os
import signal
from pathlib import Path


def main():
    print("=" * 65)
    print("🚀 Launching Document-Based AI Question Answering System (RAG)")
    print("=" * 65)

    python_executable = sys.executable
    project_root = Path(__file__).resolve().parent

    # 1. Start FastAPI Backend
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        python_executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(project_root)
    )

    # Wait 2 seconds for backend to start up
    time.sleep(2)

    # 2. Start Streamlit Frontend
    print("[2/2] Starting Streamlit Frontend on http://localhost:8501 ...\n")
    frontend_cmd = [
        python_executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.headless", "false"
    ]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(project_root)
    )

    print("-" * 65)
    print("🎉 Both services are running!")
    print("👉 Frontend UI:      http://localhost:8501")
    print("👉 Backend API Docs: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to terminate both servers.")
    print("-" * 65)

    try:
        while True:
            time.sleep(1)
            # If any process dies, break
            if backend_proc.poll() is not None or frontend_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    finally:
        for proc in [backend_proc, frontend_proc]:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("✅ Servers stopped.")


if __name__ == "__main__":
    main()
