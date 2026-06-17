# Neuron AI v2.0 — Clinical Workstation MVP

Neuron AI is a Local B2B SaaS Diagnostic Platform designed for Clinics and Hospitals in India. It is a full-stack clinical workstation built to process medical scans (X-Ray, CT, MRI), perform AI-driven inference using deep learning models, and present diagnostic highlights through an intuitive, radiologist-friendly dashboard.

## 🚀 Features

- **End-to-End Inference Pipeline**:
  - **Chest X-Ray**: Uses `DenseNet121` via `torchxrayvision`.
  - **CT & MRI**: Uses 3D Segmentation models (`Swin UNETR`, `DynUNet`) via `MONAI`.
- **Privacy First (DPDP-Aligned)**: Automatically strips Patient PII and applies salted SHA-256 anonymization to medical files before processing.
- **Robust Authentication**: JWT-based authentication with bcrypt password hashing. Default admin credentials are provided out-of-the-box.
- **Clinical Dashboard**: Upload DICOM, NIfTI, or standard image formats and view diagnostic highlights (bounding boxes, predictions, confidence scores). Includes scan history, pagination, and overall clinical telemetry (metrics, modalities).
- **Scalable Architecture**: Device-aware inference (CPU/GPU detection), caching for model weights, and rate-limiting using `slowapi`.

## 📁 Repository Structure

```text
.
├── app/                  # FastAPI Backend logic
│   ├── main.py           # API routing, server lifecycle, configuration
│   ├── auth.py           # JWT Authentication & User management
│   ├── database.py       # SQLite database configuration & ORM models
│   ├── inference.py      # PyTorch model loading, execution, and transforms
│   ├── utils.py          # Preprocessing, PII anonymization, image format conversions
├── frontend/             # Modern React + Vite frontend setup (Work In Progress)
├── models/               # Downloaded PyTorch / MONAI model weights (*.pt files)
├── static/               # Native Frontend JS/CSS (app.js, app.css, responsive.css)
├── templates/            # Native Frontend HTML templates (index.html, login.html)
├── test_scans/           # Mock DICOM / NIfTI / PNG scans for testing
├── benchmark_sota.py     # Script to evaluate model latency, throughput, and memory
├── create_test_scans.py  # Utility to generate mock medical scans for testing
├── run_local.sh          # One-click bootstrap script for local environments
├── REPORT.md             # Detailed build report and architecture explanation
├── RESPONSIVE_DESIGN.md  # Notes on frontend responsive design and UI breakpoints
├── VALIDATION.md         # Status report on API and pipeline testing
└── requirements.txt      # Python dependencies
```

## 🛠️ How to Set Up and Run

This application is designed to be run locally in a secure clinical environment. 

### Prerequisites
- Linux / macOS (or WSL on Windows)
- Python 3.8+
- (Optional) A CUDA-capable GPU for accelerated inference.

### Quick Start
To bootstrap the entire environment (create a virtual environment, install dependencies, create necessary folders, and start the server), run the provided bash script:

```bash
chmod +x run_local.sh
./run_local.sh
```

The script will:
1. Create a Python virtual environment (`venv`).
2. Install all dependencies from `requirements.txt`.
3. Create required runtime directories (`temp_uploads/`, `models/`, `static/`).
4. Start the FastAPI server using `uvicorn`.

### Accessing the Portal
Once running, open your web browser and navigate to:
**http://127.0.0.1:8000**

**Default Login Credentials:**
- **Username:** `admin`
- **Password:** `neuron2026`

## 🧪 Testing and Benchmarking

- **Mock Data**: Use `python create_test_scans.py` to generate sample X-Ray, MRI, and CT files in the `test_scans/` directory for testing the upload pipeline.
- **Model Benchmark**: Use `python benchmark_sota.py` to test the performance (latency, throughput, peak VRAM) of the implemented models on your specific hardware.

## 📚 Further Reading

- **Architecture Details:** See `REPORT.md`
- **Responsive UI implementation:** See `RESPONSIVE_DESIGN.md`
- **Validation and Testing:** See `VALIDATION.md`
