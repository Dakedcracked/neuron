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
mkdir -p temp_uploads static models mock_s3_bucket

# 5.5 Pre-cache model weights dynamically
echo "⏳ Pre-checking and downloading model weights (this might take a few minutes on first run)..."
python models/download_weights.py
echo "✓ Model weights checked and ready."

# 6. Initialize local .env if it does not exist

if [ ! -f .env ]; then
    echo "📝 Creating local .env from .env.example..."
    cp .env.example .env
    
    # Generate random keys using active python
    export RAND_JWT_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    export RAND_SALT=$(python -c "import secrets; print(secrets.token_hex(16))")
    
    python -c "
import os
with open('.env', 'r') as f:
    content = f.read()
content = content.replace('NEURON_SECRET_KEY=', 'NEURON_SECRET_KEY=' + os.environ['RAND_JWT_KEY'])
content = content.replace('NEURON_CLINIC_SALT=', 'NEURON_CLINIC_SALT=' + os.environ['RAND_SALT'])
with open('.env', 'w') as f:
    f.write(content)
"
    echo "✓ Generated secure random NEURON_SECRET_KEY and NEURON_CLINIC_SALT in .env."
    # Clean up temporary export variables
    unset RAND_JWT_KEY
    unset RAND_SALT
fi


# 7. Load environment variables from .env
echo "🔌 Loading environment variables from .env..."
while IFS= read -r line || [ -n "$line" ]; do
    # Strip carriage returns and ignore comments / empty lines
    line=$(echo "$line" | tr -d '\r')
    if [[ ! "$line" =~ ^# ]] && [[ -n "$line" ]]; then
        export "$line"
    fi
done < .env

# 7. Start Celery worker in the background
echo "🐝 Starting Celery background worker..."
celery -A app.worker.celery_app worker --loglevel=info --pool=solo > celery.log 2>&1 &
CELERY_PID=$!

# 8. Start Neuron Inference Model Server in the background
echo "⚡ Starting Neuron AI Inference Server (Dynamic Batching)..."
uvicorn app.inference_server:app --host 127.0.0.1 --port 8001 > model_server.log 2>&1 &
MODEL_SERVER_PID=$!

# Function to kill background processes when uvicorn exits
cleanup() {
    echo "Stopping background processes..."
    kill $CELERY_PID 2>/dev/null || true
    kill $MODEL_SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# 8. Start the local workstation FastAPI server
echo ""
echo "🖥️  Starting Neuron AI v2.0 on http://127.0.0.1:8000..."
echo "📋 Default login: admin / neuron2026"
echo "Press Ctrl+C to terminate the session."
echo
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
