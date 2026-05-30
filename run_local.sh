#!/bin/bash
set -e

echo "======================================================================"
echo "    🚀 Bootstrapping Neuron AI v2.0 — Clinical Workstation MVP 🚀"
echo "======================================================================"
echo

# 1. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment 'venv'..."
    python3 -m venv venv
    echo "✓ Virtual environment created."
else
    echo "✓ Virtual environment 'venv' already exists."
fi

# 2. Activate Virtual Environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated."

# 3. Upgrade pip
echo "🆙 Upgrading pip..."
pip install --upgrade pip
echo "✓ Pip upgraded."

# 4. Install dependencies
echo "📥 Installing dependencies from requirements.txt..."
echo "Note: PyTorch, MONAI, torchxrayvision may take a few minutes on first install."
pip install -r requirements.txt
echo "✓ All dependencies installed successfully."

# 5. Ensure directories exist
mkdir -p temp_uploads static models

# 6. Start the local workstation FastAPI server
echo ""
echo "🖥️  Starting Neuron AI v2.0 on http://127.0.0.1:8000..."
echo "📋 Default login: admin / neuron2026"
echo "Press Ctrl+C to terminate the session."
echo
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
