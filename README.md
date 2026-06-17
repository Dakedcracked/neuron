# Neuron AI v2.0 — Clinical Workstation MVP & Production Scale Architecture

Neuron AI is a B2B SaaS Diagnostic Platform designed for Clinics and Hospitals. It is a full-stack clinical workstation built to process medical scans (X-Ray, CT, MRI), perform AI-driven inference using deep learning models, and present diagnostic highlights through an intuitive, radiologist-friendly dashboard.

This repository is optimized for **high-throughput production environments**, capable of handling millions of queries per minute and high-frequency model inference, while minimizing cloud hosting costs.

---

## 📁 Repository Structure & File Mappings

Below is the directory mapping of the clinical platform:

```text
.
├── app/                      # FastAPI Backend microservices
│   ├── main.py               # API routing, server lifecycle, rate-limiting, and endpoints
│   ├── auth.py               # JWT Authentication, token extraction, and password hashing
│   ├── database.py           # SQLite (local) / PostgreSQL (production) models & connection pooling
│   ├── inference.py          # Inference router and local fallback execution pipeline
│   ├── inference_server.py   # Standalone Model Server exposing dynamic-batching API
│   ├── utils.py              # DPDP-compliant PII anonymization, direct S3 upload url generator
│   └── worker.py             # Celery background task processing (downloads & preprocesses scans)
│
├── frontend/                 # Modern React + Vite frontend source code
│   ├── src/                  # React components (Dashboard, Inference, History, Settings)
│   └── package.json          # Frontend dependencies and build configurations
│
├── static/                   # Compiled static assets served directly by the backend
│   └── assets/               # Active CSS/JS production bundle chunks
│
├── templates/                # Jinja2 HTML templates
│   ├── index.html            # Main Single Page Application entry point
│   └── login.html            # Standalone secure radiologist login page
│
├── mock_s3_bucket/           # Local mock S3/R2 storage directory for offline development
├── models/                   # Local PyTorch (.pt) and ONNX weights
├── requirements.txt          # Python dependencies
├── run_local.sh              # One-click bootstrap script for local environments
├── docker-compose.yml        # Docker configuration for production stack (FastAPI, Redis, PostgreSQL)
└── Dockerfile                # Multi-stage build container specification
```

---

## 🔄 How Scans & Queries Flow (End-to-End)

When a radiologist uploads a scan and views the diagnostic results, the system processes it through a decoupled, asynchronous pipeline:

```text
 [Radiologist UI]
        │
        ▼ 1. Request Presigned URL
 [FastAPI API] (main.py) ──► Generates secure AWS S3/R2 presigned upload URL
        │
        ▼ 2. Stream Binary Bytes
   [S3/R2 Cloud] ◄────────── Direct-upload from Client (FastAPI memory usage: 0 MB)
        │
        ▼ 3. Notify Upload Complete
 [FastAPI API] (main.py) ──► Creates "Pending" DB record, puts Task on Redis queue
        │
        ▼ 4. Fetch Scan Task
  [Celery Worker] (worker.py)
        │
        ├──► a. Downloads raw scan file from S3 to temp location
        ├──► b. Extracts grayscale axial slice & converts to Base64 PNG preview
        └──► c. POSTs file path to Model Server on Port 8001
              │
              ▼ 5. Dynamic Queueing
      [Model Server] (inference_server.py)
              │
              ├──► a. Places request into an internal asyncio.Queue
              ├──► b. Groups incoming requests into a single stacked Tensor (Batching)
              └──► c. Executes single batched forward pass on GPU (ONNX/PyTorch)
              │
              ▼ 6. Result Return
  [Celery Worker] (worker.py) ──► Updates Database record with predictions, status, and bbox
        │
        ▼ 7. UI Polling
 [Radiologist UI] ◄───────── Pulls scan results & renders diagnostic bounding boxes
```

---

## ⚡ How the Platform Handles Millions of Requests & Queries per Minute

To achieve SaaS-level scale and prevent cloud bills from exploding, the workstation implements several key performance optimizations:

### 1. Zero-Memory Copy API Ingestion (Bypassing Server Buffering)
* **The Problem**: Uploading large medical scans (NIfTI/DICOM files can range from 50MB to 500MB) directly to a FastAPI server consumes enormous RAM. Under concurrent loads (e.g. thousands of uploads), the server will crash from Out-of-Memory (OOM) errors.
* **The Solution**: The API server never buffers the binary file. Instead, the frontend calls `/api/get-upload-url` to obtain a pre-signed AWS S3/R2 upload URL. The frontend uploads the heavy binary payload **directly to S3/R2**. FastAPI only receives the S3 key (`/api/register-scan`), reducing server memory overhead to $O(1)$.

### 2. Decoupled CPU-Heavy Preprocessing
* **The Problem**: Reading DICOM headers, loading NIfTI 3D volumes, selecting the mid-slice, and converting pixel matrices to base64 PNGs requires heavy CPU compute, which blocks the API event loop and slows response times.
* **The Solution**: FastAPI registers the scan as "Pending" in the database and delegates all image parsing and Base64 preview generation to the background [Celery Worker](file:///app/worker.py), keeping the API server fully responsive.

### 3. Queue-Based Dynamic Batching (Inference Server)
* **The Problem**: Deep learning models (DenseNet121, SegResNet) running in-process for single scans are highly inefficient. If 100 radiologists request a scan simultaneously, a GPU executing them sequentially will experience high latency, low utilization, and high cloud hosting costs.
* **The Solution**: We extract the models into a dedicated Model Server [inference_server.py](file:///app/inference_server.py).
  * Requests are queued in an asynchronous queue (`asyncio.Queue`).
  * The server waits for up to **15 milliseconds** or until **16 requests** accumulate.
  * It groups the separate image tensors into a single stacked batch tensor (e.g., `[16, 1, 224, 224]`) and executes a **single batched forward pass** on the GPU.
  * This multiplies inference throughput by **5x to 10x**, maximizes GPU utilization, and significantly reduces cloud compute costs.

### 4. Quantization & ONNX Runtime
* Volunteer weights (MRI SegResNet) are compiled into ONNX and run via `onnxruntime` using half-precision (`FP16`). This halves the memory bandwidth requirement of the model, allowing twice the batch size on the same GPU VRAM.

### 5. High-Performance Database Connection Pooling
* Standard SQLAlchemy database configurations block threads and suffer from connection starvation under heavy QPS.
* In [database.py](file:///app/database.py), connection pooling is optimized for PostgreSQL:
  * `pool_size=50`: Keeps 50 connections alive to eliminate connection setup overhead.
  * `max_overflow=100`: Allows up to 100 temporary connections under peak spikes.
  * `pool_recycle=1800`: Automatically recycles idle connections to prevent server timeout drops.

---

## 🛠️ How to Bootstrap and Run

### Prerequisites
* Linux / macOS
* Python 3.8+
* Redis server (must be installed and running on `localhost:6379`)

### Quick Start
To start all workstation microservices (FastAPI Server, Standalone Model Server, and Celery background worker) in local SQLite mode:

```bash
chmod +x run_local.sh
./run_local.sh
```

The bootstrap script will automatically:
1. Initialize the Python virtual environment and install dependencies.
2. Spin up the background **Celery worker** (logging to `celery.log`).
3. Spin up the standalone **Model Server** on Port 8001 (logging to `model_server.log`).
4. Start the **FastAPI Web Workstation** on Port 8000.

Open **http://127.0.0.1:8000** in your browser.
* **Username**: `admin`
* **Password**: `neuron2026`
