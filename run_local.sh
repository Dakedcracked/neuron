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

# 6. Set default environment variables
export DATABASE_URL=${DATABASE_URL:-"sqlite:///neuron_clinic.db"}
export REDIS_URL=${REDIS_URL:-"redis://localhost:6379/0"}
export MOCK_S3=${MOCK_S3:-"true"}

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
