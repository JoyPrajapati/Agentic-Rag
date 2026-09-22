"""Launch both FastAPI and Streamlit from a single command.

Usage:
    python run.py
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def start_process(name, cmd, env=None):
    """Start a subprocess and return the Popen object."""
    print(f"🚀 Starting {name}...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, **(env or {})},
        shell=True,
    )
    return proc


def main():
    api_proc = None
    ui_proc = None

    try:
        # 1. Start FastAPI
        api_proc = start_process(
            "FastAPI",
            f'"{sys.executable}" -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000',
        )

        # Give FastAPI time to boot and load models
        print("⏳ Waiting 30 seconds for FastAPI to load models...")
        time.sleep(30)

        # 2. Start Streamlit
        ui_proc = start_process(
            "Streamlit",
            f'"{sys.executable}" -m streamlit run streamlit_app.py '
            f'--server.port 8501 --server.headless true',
        )

        print("\n" + "=" * 60)
        print("✅ Both services are running:")
        print("   API:      http://localhost:8000")
        print("   API docs: http://localhost:8000/docs")
        print("   UI:       http://localhost:8501")
        print("=" * 60)
        print("\nPress Ctrl+C to stop both.\n")

        # Wait for either to exit
        api_proc.wait()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        for name, proc in [("FastAPI", api_proc), ("Streamlit", ui_proc)]:
            if proc and proc.poll() is None:
                print(f"   Killing {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        # Clean up any lingering Python processes on Windows
        if sys.platform == "win32":
            subprocess.run(
                'taskkill /F /IM python.exe /FI "PID ne %d"' % os.getpid(),
                shell=True,
                capture_output=True,
            )

        print("✅ All services stopped.")


if __name__ == "__main__":
    main()