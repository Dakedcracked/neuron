# Neuron AI Clinical Workstation — Build Report

## 1. Summary
This project is a **FastAPI-based radiologist workstation** for clinics that accepts medical scans, anonymizes PII, runs AI inference, and provides a full operational dashboard (upload, diagnostics, history, settings). It is a complete, working web application with authentication, database-backed analytics, and an end‑to‑end inference pipeline.

## 2. What was built

**Application surface**
- **Login + JWT auth** with default admin user (admin / neuron2026)
- **Dashboard** with telemetry (total scans, revenue, positive rate, modality distribution)
- **Upload + viewer** with contrast/brightness controls and bounding boxes
- **Scan history** with pagination and filters
- **Settings** for clinic name/station and account management

**Backend services**
- **FastAPI** application in `app/main.py`
- **SQLite** database models and aggregation in `app/database.py`
- **Auth** (JWT + bcrypt) in `app/auth.py`
- **Medical file preprocessing** (DICOM, NIfTI, PNG/JPG) in `app/utils.py`
- **Inference pipeline** in `app/inference.py`

**Model runtime**
- **X‑ray**: torchxrayvision DenseNet121, using local cached weights if available
- **CT/MRI**: MONAI 3D UNet / ResNet-50 (clinical weights required)
- **Fallback**: returns **Inconclusive** if no validated model is available

## 3. Architecture & data flow

1. **UI upload** → `/api/upload-scan`
2. **Preprocessing**:
   - DICOM: reads metadata and pixel array, anonymizes patient ID, extracts modality
   - NIfTI: reads volume, selects mid‑slice for preview, infers modality
   - PNG/JPG: grayscale conversion
3. **Inference**:
   - XRAY → DenseNet121 with pathology mapping
   - CT/MRI → 3D UNet → lesion probability → class mapping
4. **Persistence**:
   - Writes scan log and updates dashboard metrics
5. **Response**:
   - Returns prediction, confidence, costs, bbox, visualization PNG, model info

## 4. Security & compliance

- **PII anonymization** via salted SHA‑256 (DPDP‑aligned)
- **JWT sessions** (8h) with bcrypt password hashing
- **Rate limiting** on auth and upload endpoints

## 5. UX and visual design

The UI was refined to **look like a conventional clinical dashboard**, not an “AI demo”:
- Clean, light theme with conservative typography and subdued color palette
- Removed animated neural background on login
- Reduced gradients and “glow” effects
- Emphasis on readability, clarity, and clinical tone

## 6. Scalability posture

The application is **scalable by design** because it is stateless at the API layer and uses:
- **Device‑aware inference** (CPU/GPU) with PyTorch
- **Weight caching** for offline or cold‑start optimization
- **Modular inference** that can be separated into a worker service

To scale further in production:
- Run multiple API workers behind a reverse proxy
- Move inference to a GPU worker queue (Celery/RQ + Redis)
- Use a managed DB (Postgres) for concurrent clinic tenants

## 7. Known limitations

- CT/MRI inference is **disabled without clinical weights**
- X‑ray DenseNet uses pretrained weights but still requires clinical validation
- This system is for **decision support**, not autonomous diagnosis

## 8. Key files

- `app/main.py` — API routes, auth wiring, and server lifecycle
- `app/inference.py` — model loading, transforms, predictions
- `app/utils.py` — preprocessing, anonymization, base64 rendering
- `app/database.py` — ORM models and billing aggregation
- `static/app.css` — UI styling
- `templates/index.html`, `templates/login.html` — UI structure

---

If you want the report in a different format or additional diagrams (sequence/flow), say the word and I will extend this file.
