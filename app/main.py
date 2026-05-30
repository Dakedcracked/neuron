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

from app.database import init_db, get_db, log_scan, get_dashboard_metrics, get_scan_history, update_clinic_setting, get_clinic_settings, SessionLocal as AuthSession
from app.utils import preprocess_medical_file
from app.inference import run_inference
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
@limiter.limit("10/minute")
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
        if ext.endswith((".png", ".jpg", ".jpeg")) and modality_override in {"CT", "MRI"}:
            raise HTTPException(
                status_code=400,
                detail="CT/MRI uploads require DICOM or NIfTI files.",
            )
        if modality_override in {"XRAY", "CT", "MRI"} and not ext.endswith(".dcm"):
            modality = modality_override
        else:
            modality = modality_from_file
        img_base64 = prepped["img_base64"]
        metadata = prepped["metadata"]

        os.makedirs("temp_uploads", exist_ok=True)
        temp_path = os.path.join("temp_uploads", filename)
        with open(temp_path, "wb") as f:
            f.write(content)

        import time
        try:
            t_start = time.perf_counter()
            inference_out = run_inference(temp_path, modality, patient_hash)
            inference_latency = (time.perf_counter() - t_start) * 1000.0  # latency in ms
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        logged = log_scan(
            db=db,
            patient_hash=patient_hash,
            scan_type=modality,
            pathology_detected=inference_out["pathology_detected"],
            confidence_score=inference_out["confidence_score"],
            inference_latency=inference_latency,
            pytorch_executed=inference_out["pytorch_executed"],
        )

        return {
            "status": "success",
            "scan_id": logged.id,
            "patient_hash": patient_hash,
            "modality": modality,
            "metadata": metadata,
            "pathology_detected": inference_out["pathology_detected"],
            "confidence_score": inference_out["confidence_score"],
            "predictions": inference_out["predictions"],
            "bbox": inference_out["bbox"],
            "inference_latency": inference_latency,
            "img_base64": img_base64,
            "timestamp": logged.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_executed": inference_out["pytorch_executed"],
            "model_info": inference_out["model_info"],
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

