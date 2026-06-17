import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.database import init_db, get_db, create_pending_scan, get_dashboard_metrics, get_scan_history, update_clinic_setting, get_clinic_settings, SessionLocal as AuthSession, Scan
from app.utils import preprocess_medical_file, generate_presigned_upload_url
from app.inference import run_inference
from app.worker import process_scan
from app.auth import (
    User, seed_default_admin, get_current_user, get_admin_user,
    verify_password, hash_password, create_access_token,
)

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — init all tables (including User from auth.py)
    from app.database import engine, Base
    Base.metadata.create_all(bind=engine)
    init_db()
    db = AuthSession()
    try:
        seed_default_admin(db)
    finally:
        db.close()
    print("✓ Neuron AI Clinical Platform ready.")
    yield
    print("Neuron AI shutting down.")


# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neuron AI Assistant — Radiologist Portal",
    description="Local B2B SaaS Diagnostic Platform for Clinics and Hospitals in India",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS (localhost only) ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Templates & Static Files ──────────────────────────────────────────────────
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── HTML Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# ── Auth API ──────────────────────────────────────────────────────────────────

@app.post("/api/auth/login")
@limiter.limit("50/minute")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Validates credentials and returns JWT access token."""
    db = AuthSession()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.hashed_password) or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        token = create_access_token({"sub": user.username, "role": user.role})
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": user.username,
            "role": user.role,
        }
    finally:
        db.close()


@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "role": current_user.role}


@app.post("/api/auth/change-password")
async def change_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    db = AuthSession()
    try:
        user = db.query(User).filter(User.username == current_user.username).first()
        if not verify_password(old_password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect.")
        if len(new_password) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")
        user.hashed_password = hash_password(new_password)
        db.commit()
        return {"status": "success", "message": "Password updated successfully."}
    finally:
        db.close()


# ── Scaling Optimized S3/R2 Direct Uploads ────────────────────────────────────

@app.post("/api/get-upload-url")
async def get_upload_url(
    filename: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """
    Generates a presigned URL/POST payload for direct-to-S3 client uploading.
    This eliminates server-side buffering of large scan payloads.
    """
    url_data = generate_presigned_upload_url(filename)
    if not url_data:
        raise HTTPException(status_code=500, detail="Failed to generate upload URL.")
    return url_data


@app.post("/api/mock-upload")
async def mock_upload(file: UploadFile = File(...), key: str = Form(...)):
    """
    Simulates direct-to-S3 upload for local workstation testing.
    Saves the payload to the local mock S3 bucket.
    """
    try:
        mock_bucket = os.path.join(os.getcwd(), "mock_s3_bucket")
        os.makedirs(mock_bucket, exist_ok=True)
        local_path = os.path.join(mock_bucket, key)
        content = await file.read()
        with open(local_path, "wb") as f:
            f.write(content)
        return {"status": "success", "s3_url": f"s3://{os.environ.get('S3_BUCKET_NAME', 'neuron-clinical-scans')}/mock/{key}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/register-scan")
async def register_scan(
    s3_key: str = Form(...),
    modality: str = Form(None),
    patient_id: str = Form("ANON_ID"),
    patient_name: str = Form("ANON_NAME"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Registers a direct-to-S3 uploaded scan, parses the record metadata,
    and queues it for asynchronous Celery background processing.
    """
    from app.utils import anonymize_patient, S3_BUCKET
    patient_hash = anonymize_patient(patient_id, patient_name)
    s3_url = f"s3://{S3_BUCKET}/mock/{s3_key}" if os.environ.get("MOCK_S3", "true").lower() == "true" else f"https://{S3_BUCKET}.s3.{os.environ.get('S3_REGION', 'ap-south-1')}.amazonaws.com/{s3_key}"
    
    # Register scan in DB as pending
    logged = create_pending_scan(db=db, patient_hash=patient_hash, scan_type=modality or "XRAY", s3_url=s3_url)
    
    # Dispatch non-blocking Celery task
    process_scan.delay(scan_id=logged.id, s3_url=s3_url, modality=modality or "XRAY", patient_hash=patient_hash)
    
    return {
        "status": "pending",
        "scan_id": logged.id,
        "patient_hash": patient_hash,
        "modality": modality or "XRAY",
        "s3_url": s3_url,
        "message": "Scan registered and queued for background inference.",
    }


# ── Scan Upload API ───────────────────────────────────────────────────────────

@app.post("/api/upload-scan")
@limiter.limit("30/minute")
async def upload_scan(
    request: Request,
    file: UploadFile = File(...),
    modality: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Ingests scans, strips Patient PII under DPDP directives, performs deep learning
    inference, registers logs in SQLite database, and returns diagnostic highlights.
    """
    filename = file.filename
    ext = filename.lower()

    if not ext.endswith((".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Accepts: DICOM (.dcm), NIfTI (.nii, .nii.gz), PNG, JPG.",
        )

    try:
        content = await file.read()
        prepped = preprocess_medical_file(content, filename)

        patient_hash = prepped["patient_hash"]
        modality_from_file = prepped["modality"]
        modality_override = modality.upper() if modality else None
        if modality_override in {"XRAY", "CT", "MRI"} and not ext.endswith(".dcm"):
            modality = modality_override
        else:
            modality = modality_from_file
        img_base64 = prepped["img_base64"]
        metadata = prepped["metadata"]

        from app.utils import upload_to_s3
        import tempfile
        # 1. Permanently store the raw scan in S3/R2
        s3_url = upload_to_s3(content, filename)

        # 2. Register the scan in the DB as "Pending"
        logged = create_pending_scan(db=db, patient_hash=patient_hash, scan_type=modality, s3_url=s3_url)
        logged.img_base64 = img_base64
        db.commit()

        # 3. Dispatch the Celery task (non-blocking)
        process_scan.delay(scan_id=logged.id, s3_url=s3_url, modality=modality, patient_hash=patient_hash)

        return {
            "status": "pending",
            "scan_id": logged.id,
            "patient_hash": patient_hash,
            "modality": modality,
            "metadata": metadata,
            "img_base64": img_base64,
            "timestamp": logged.timestamp.strftime("%Y-%m-%d %H:%M:%S") if logged.timestamp else None,
            "message": "Scan uploaded and queued for background inference.",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠ Critical failure in upload-scan pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Clinical processor failure: {str(e)}")


# ── Dashboard Metrics API ─────────────────────────────────────────────────────

@app.get("/api/dashboard-metrics")
async def dashboard_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return get_dashboard_metrics(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query telemetry: {str(e)}")


# ── Scan History API ──────────────────────────────────────────────────────────

@app.get("/api/scans/{scan_id}")
async def get_scan_status(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    import json
    return {
        "id": scan.id,
        "status": scan.status,
        "patient_hash": scan.patient_hash,
        "scan_type": scan.scan_type,
        "pathology_detected": scan.pathology_detected,
        "confidence_score": scan.confidence_score,
        "predictions": json.loads(scan.predictions) if scan.predictions else None,
        "bbox": json.loads(scan.bbox) if scan.bbox else None,
        "img_base64": scan.img_base64,
        "inference_latency": scan.inference_latency,
        "timestamp": scan.timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.timestamp else None,
    }

@app.get("/api/scans")
async def scan_history(
    page: int = 1,
    size: int = 20,
    scan_type: str = None,
    pathology: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_scan_history(db, page=page, page_size=size, scan_type=scan_type, pathology=pathology)


# ── Settings API ──────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_clinic_settings(db)


@app.post("/api/settings")
async def save_settings(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    body = await request.json()
    allowed_keys = {"clinic_name", "station_id"}
    for key, value in body.items():
        if key in allowed_keys:
            update_clinic_setting(db, key, str(value))
    return {"status": "saved"}

