import os
import time
import json
import tempfile
import urllib.request
from celery import Celery
from app.inference import run_inference
from app.database import SessionLocal, update_scan_result, Scan

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("neuron_worker", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task(name="process_scan")
def process_scan(scan_id: str, s3_url: str, modality: str, patient_hash: str):
    print(f"⏳ [Celery] Starting background inference for Scan {scan_id}")
    
    # 1. Download file from S3 to an ephemeral tempfile
    fd, temp_path = tempfile.mkstemp()
    filename = s3_url.split("/")[-1]
    file_content = None
    with os.fdopen(fd, 'wb') as f:
        if s3_url.startswith("s3://") and "mock" in s3_url:
            print("⚠ Mock S3 URL detected. Reading from local mock_s3_bucket.")
            local_path = os.path.join(os.getcwd(), "mock_s3_bucket", filename)
            if not os.path.exists(local_path):
                print(f"Failed to find local mock file: {local_path}")
                os.remove(temp_path)
                return False
            with open(local_path, "rb") as mock_f:
                file_content = mock_f.read()
                f.write(file_content)
        else:
            try:
                req = urllib.request.Request(s3_url)
                with urllib.request.urlopen(req) as response:
                    file_content = response.read()
                    f.write(file_content)
            except Exception as e:
                print(f"⚠ [Celery] Failed to download scan from S3: {e}")
                os.remove(temp_path)
                return False

    # Extract img_base64 preview from the downloaded file
    from app.utils import preprocess_medical_file
    img_base64 = None
    try:
        if file_content:
            prepped = preprocess_medical_file(file_content, filename)
            img_base64 = prepped.get("img_base64")
    except Exception as e:
        print(f"⚠ [Celery] Preprocessing/base64 extraction failed: {e}")

    # 2. Execute Heavy ONNX Inference
    try:
        t_start = time.perf_counter()
        inference_out = run_inference(temp_path, modality, patient_hash)
        latency = (time.perf_counter() - t_start) * 1000.0
    except Exception as e:
        print(f"❌ [Celery] Inference crashed: {e}")
        db = SessionLocal()
        try:
            scan = db.query(Scan).filter(Scan.id == scan_id).first()
            if scan:
                scan.status = "failed"
                scan.pathology_detected = f"AI Error: {str(e)[:50]}"
                db.commit()
        finally:
            db.close()
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    # 3. Save Results to Database
    db = SessionLocal()
    try:
        update_scan_result(
            db=db,
            scan_id=scan_id,
            pathology=inference_out["pathology_detected"],
            confidence=inference_out["confidence_score"],
            latency=latency,
            pytorch_exec=inference_out["pytorch_executed"],
            img_base64=img_base64,
            predictions=json.dumps(inference_out["predictions"]) if inference_out["predictions"] else None,
            bbox=json.dumps(inference_out["bbox"]) if inference_out["bbox"] else None
        )
        print(f"✓ [Celery] Inference complete for Scan {scan_id}")
    except Exception as e:
        print(f"⚠ [Celery] Failed to save DB results: {e}")
    finally:
        db.close()
        
    return True
