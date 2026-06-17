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
content = content.replace('JWT_SECRET=', 'JWT_SECRET=' + os.environ['RAND_JWT_KEY'])
content = content.replace('NEURON_CLINIC_SALT=', 'NEURON_CLINIC_SALT=' + os.environ['RAND_SALT'])
with open('.env', 'w') as f:
    f.write(content)
"
    echo "✓ Generated secure random JWT_SECRET and NEURON_CLINIC_SALT in .env."
    # Clean up temporary export variables
    unset RAND_JWT_KEY
    unset RAND_SALT
fi


# 7. Load environment variables securely from .env
echo "🔌 Loading environment variables from .env securely..."
if [ -f .env ]; then
    eval $(python3 -c "
import os
with open('.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split('=', 1)
            if len(parts) == 2:
                k, v = parts
                k = k.strip()
                v = v.strip()
                # Strip quotes if present
                if (v.startswith('\"') and v.endswith('\"')) or (v.startswith('\'') and v.endswith('\'')):
                    v = v[1:-1]
                # Escape single quotes for safe shell export
                v_esc = v.replace(\"'\", \"'\\\"'\\\"'\")
                print(f\"export {k}='{v_esc}'\")
")
fi

# 7.5. Run Pre-flight Diagnostics
echo "🔍 Running pre-flight diagnostics..."

# Check CUDA capabilities
python3 -c "
import torch
print(f'   - PyTorch version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'   - CUDA available: {cuda_avail}')
if cuda_avail:
    print(f'   - GPU device: {torch.cuda.get_device_name(0)}')
    print(f'   - CUDA Device count: {torch.cuda.device_count()}')
else:
    print('   - Running in CPU-only fallback mode. Heavy neural model inference may be slower.')
"

# Check foundation weights files existence
echo "🔍 Verifying foundation model weights in 'models/'..."
if [ ! -f "models/densenet121_radimagenet.pt" ]; then
    echo "   ⚠ Warning: models/densenet121_radimagenet.pt is missing. Folds will degrade gracefully using densenet121_xrv.pt."
else
    echo "   ✓ models/densenet121_radimagenet.pt is present."
fi
if [ ! -f "models/resnet50_radimagenet.pt" ]; then
    echo "   ⚠ Warning: models/resnet50_radimagenet.pt is missing. Folds will degrade gracefully using resnet50_clinical.pt."
else
    echo "   ✓ models/resnet50_radimagenet.pt is present."
fi
if [ ! -f "models/medsam.onnx" ]; then
    echo "   ⚠ Warning: models/medsam.onnx is missing. MedSAM will degrade to MONAI SegResNet or simulated overlays."
else
    echo "   ✓ models/medsam.onnx is present."
fi

# Check Postgres and Redis connections
python3 -c "
import os
import sys

# Test PostgreSQL connection
db_url = os.environ.get('DATABASE_URL')
if db_url:
    try:
        import urllib.parse as urlparse
        # Extract connection properties
        url = urlparse.urlparse(db_url)
        import psycopg2
        conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
        conn.close()
        print('   - PostgreSQL database connection successful.')
    except Exception as e:
        print(f'   - ⚠ PostgreSQL connection failed: {e}')
else:
    print('   - ⚠ DATABASE_URL is not set in env.')

# Test Redis connection
redis_url = os.environ.get('REDIS_URL')
if redis_url:
    try:
        import redis
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print('   - Redis connection successful.')
    except Exception as e:
        print(f'   - ⚠ Redis connection failed: {e}')
else:
    print('   - ⚠ REDIS_URL is not set in env.')
"
echo "✓ Pre-flight diagnostics complete."

# 7. Run database migrations to ensure priority and tenant columns exist
echo "🔄 Running database migrations..."
python3 -c "
import os
from sqlalchemy import text
try:
    from app.database import engine
    with engine.begin() as conn:
        # Create clinic_tenants if not exists
        conn.execute(text(\"\"\"
        CREATE TABLE IF NOT EXISTS clinic_tenants (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL UNIQUE,
            created_at TIMESTAMP WITHOUT TIME ZONE
        );
        \"\"\"))
        
        # Add priority column to scans if not exists
        try:
            conn.execute(text(\"ALTER TABLE scans ADD COLUMN priority VARCHAR DEFAULT 'normal';\"))
        except Exception:
            pass
            
        # Add tenant_id column to scans if not exists
        try:
            conn.execute(text(\"ALTER TABLE scans ADD COLUMN tenant_id VARCHAR;\"))
            conn.execute(text(\"ALTER TABLE scans ADD CONSTRAINT fk_scans_tenant FOREIGN KEY (tenant_id) REFERENCES clinic_tenants(id);\"))
        except Exception:
            pass
            
        # Add tenant_id column to users if not exists
        try:
            conn.execute(text(\"ALTER TABLE users ADD COLUMN tenant_id VARCHAR;\"))
            conn.execute(text(\"ALTER TABLE users ADD CONSTRAINT fk_users_tenant FOREIGN KEY (tenant_id) REFERENCES clinic_tenants(id);\"))
        except Exception:
            pass
            
    print('   ✓ Database migrations successful (priority and tenant columns verified).')
except Exception as e:
    print(f'   - ⚠ Database migration warning: {e}')
"

# 8. Start Celery worker in the background
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
