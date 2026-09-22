"""One-command setup for the Agentic RAG project.

Usage:
    python setup.py

What it does:
    1. Verifies Python version (3.10+)
    2. Creates a virtual environment (venv/)
    3. Installs dependencies from requirements.txt
    4. Creates .env from .env.example (if missing)
    5. Creates required folders
    6. Prompts you to place PDFs in data/raw/
    7. Optionally runs ingestion
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
REQUIRED_PYTHON = (3, 10)


def print_header(text):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def print_step(text):
    print(f"\n▶ {text}")


def check_python_version():
    print_step(f"Checking Python version (need {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+)")
    current = sys.version_info[:2]
    print(f"   Current: Python {current[0]}.{current[1]}")

    if current < REQUIRED_PYTHON:
        print(f"\n❌ ERROR: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ is required.")
        print(f"   You have Python {current[0]}.{current[1]}")
        print(f"   Download from: https://www.python.org/downloads/")
        sys.exit(1)

    print(f"   ✅ Python {current[0]}.{current[1]} is compatible")


def get_venv_python():
    """Return the path to the Python executable inside the venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def create_venv():
    print_step("Creating virtual environment (venv/)")

    if VENV_DIR.exists():
        print(f"   ⏭  venv/ already exists — skipping")
        return

    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        print(f"   ✅ Created venv/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Failed to create venv: {e}")
        print(f"   On Linux, you may need: sudo apt install python3-venv")
        sys.exit(1)


def install_dependencies():
    print_step("Installing dependencies (this may take 5-10 minutes)")

    venv_python = get_venv_python()

    # Upgrade pip first
    print("   Upgrading pip...")
    subprocess.check_call([
        str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"
    ])

    # Install from requirements.txt
    print("   Installing requirements.txt...")
    try:
        subprocess.check_call([
            str(venv_python), "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
        ])
        print("   ✅ All dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: pip install failed: {e}")
        print("   Try running manually to see full output:")
        print(f"   {venv_python} -m pip install -r requirements.txt")
        sys.exit(1)


def create_env_file():
    print_step("Setting up .env file")

    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"

    if env_path.exists():
        print("   ⏭  .env already exists — skipping")
        print("   (Delete it and re-run setup if you want to reset)")
        return

    if not example_path.exists():
        print("   ⚠  .env.example not found — creating blank .env")
        env_path.write_text("OPENROUTER_API_KEY=\nOPENAI_API_KEY=\n")
        return

    shutil.copy(example_path, env_path)
    print("   ✅ Created .env from .env.example")
    print()
    print("   ⚠️  IMPORTANT: Edit .env and add your OpenRouter API key:")
    print(f"      {env_path}")
    print("   Get a free key at: https://openrouter.ai/keys")


def create_folders():
    print_step("Creating required folders")
    folders = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        # Create .gitkeep so folder is tracked by git
        (folder / ".gitkeep").touch()
    print(f"   ✅ Created data/raw/ and data/processed/")


def check_pdfs():
    print_step("Checking for PDF documents")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    pdfs = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.PDF"))
    txts = list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.md"))

    total = len(pdfs) + len(txts)

    if total == 0:
        print("   ⚠  No documents found in data/raw/")
        print()
        print("   📄 You need to add some PDFs before ingestion.")
        print("      Options:")
        print("      1. Download SEC 10-K filings from:")
        print("         https://www.sec.gov/edgar/searchedgar/companysearch")
        print("      2. Place them in: data/raw/")
        print()
        return False

    print(f"   ✅ Found {total} document(s):")
    for p in pdfs + txts:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"      - {p.name} ({size_mb:.1f} MB)")

    return True


def run_ingestion():
    print_step("Running document ingestion")

    response = input("   Do you want to ingest documents now? (y/N): ").strip().lower()
    if response != "y":
        print("   ⏭  Skipped. Run 'python ingest.py' later.")
        return

    venv_python = get_venv_python()
    try:
        subprocess.check_call([str(venv_python), "ingest.py"])
        print("   ✅ Ingestion complete")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Ingestion failed: {e}")
        print("   Try running manually: python ingest.py")


def print_final_instructions():
    print_header("🎉 Setup Complete!")

    venv_activate = (
        "venv\\Scripts\\activate" if platform.system() == "Windows"
        else "source venv/bin/activate"
    )

    print(f"""
Next steps:

1. Edit .env and add your OpenRouter API key:
   {PROJECT_ROOT / '.env'}

2. Add PDFs to:
   {PROJECT_ROOT / 'data' / 'raw'}

3. Activate the virtual environment:
   {venv_activate}

4. Run ingestion (if you skipped it):
   python ingest.py

5. Start the app (API + UI):
   python run.py

6. Open in your browser:
   http://localhost:8501

Troubleshooting:
  - If you get "No module named 'src'", run from the project root.
  - If Qdrant/ChromaDB is locked, kill other Python processes:
      Windows: taskkill /F /IM python.exe
  - If API key error, verify .env is set correctly.

Full docs: see README.md
""")


def main():
    print_header("Agentic RAG — Setup")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Platform: {platform.system()} {platform.release()}")

    check_python_version()
    create_venv()
    install_dependencies()
    create_folders()
    create_env_file()
    has_docs = check_pdfs()

    if has_docs:
        run_ingestion()

    print_final_instructions()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠  Setup interrupted by user")
        sys.exit(1)