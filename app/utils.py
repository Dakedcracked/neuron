import hashlib
import os
import io
import base64
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image
import torch

import boto3
from botocore.exceptions import NoCredentialsError

CLINIC_SALT = os.environ.get("NEURON_CLINIC_SALT", "neuron_secure_salt_2026")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "neuron-clinical-scans")
S3_REGION = os.environ.get("S3_REGION", "ap-south-1")

def upload_to_s3(file_content: bytes, filename: str) -> str:
    """
    Uploads the raw DICOM/NIfTI file to AWS S3 / Cloudflare R2.
    Returns the public/presigned URL.
    """
    # For MVP local development, we mock S3 unless credentials are explicitly provided
    if os.environ.get("MOCK_S3", "true").lower() == "true":
        mock_bucket = os.path.join(os.getcwd(), "mock_s3_bucket")
        os.makedirs(mock_bucket, exist_ok=True)
        local_path = os.path.join(mock_bucket, filename)
        with open(local_path, "wb") as f:
            f.write(file_content)
        return f"s3://{S3_BUCKET}/mock/{filename}"
        
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=S3_REGION
    )
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=filename, Body=file_content)
        return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
    except NoCredentialsError:
        print("⚠ AWS Credentials not available. Skipping S3 upload.")
        return f"s3://{S3_BUCKET}/failed_upload/{filename}"


def anonymize_patient(patient_id: str, patient_name: str) -> str:
    """
    Complies with India's Digital Personal Data Protection (DPDP) Act.
    Replaces direct patient identifiers with a secure salted SHA-256 cryptographic hash.
    """
    raw_str = f"{patient_id.strip()}_{patient_name.strip()}_{CLINIC_SALT}"
    hashed = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    return f"PX-{hashed[:20].upper()}"


def parse_dicom(file_content: bytes) -> dict:
    """
    Parses raw DICOM bytes using pydicom. Extracts headers and returns metadata
    and normalized pixel arrays.
    """
    dataset = pydicom.dcmread(io.BytesIO(file_content))

    patient_id = getattr(dataset, "PatientID", "ANON_ID")
    patient_name = str(getattr(dataset, "PatientName", "ANON_NAME"))
    patient_hash = anonymize_patient(patient_id, patient_name)

    modality_code = getattr(dataset, "Modality", "OT")
    if modality_code in ("CR", "DX", "PX", "RG"):
        modality = "XRAY"
    elif modality_code == "CT":
        modality = "CT"
    elif modality_code == "MR":
        modality = "MRI"
    else:
        desc = str(getattr(dataset, "StudyDescription", "")).upper()
        if "CT" in desc:
            modality = "CT"
        elif "MRI" in desc or "BRAIN" in desc or "MR" in desc:
            modality = "MRI"
        else:
            modality = "XRAY"

    pixel_array = dataset.pixel_array.astype(np.float32)

    if hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept"):
        slope = float(dataset.RescaleSlope)
        intercept = float(dataset.RescaleIntercept)
        pixel_array = pixel_array * slope + intercept

    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max - p_min > 0:
        normalized_pixels = (pixel_array - p_min) / (p_max - p_min)
    else:
        normalized_pixels = np.zeros_like(pixel_array)

    return {
        "patient_hash": patient_hash,
        "modality": modality,
        "normalized_pixels": normalized_pixels,
        "raw_pixels": pixel_array,
        "patient_age": getattr(dataset, "PatientAge", "N/A"),
        "patient_sex": getattr(dataset, "PatientSex", "N/A"),
        "study_description": getattr(dataset, "StudyDescription", "General Scan"),
    }


def convert_pixels_to_base64_png(pixel_array: np.ndarray) -> str:
    """
    Converts a normalized 2D numpy array [0, 1] to a base64 encoded PNG string.
    """
    scaled = (pixel_array * 255.0).astype(np.uint8)
    image = Image.fromarray(scaled).convert("L")

    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def preprocess_medical_file(file_content: bytes, filename: str) -> dict:
    """
    Accepts raw bytes and delegates processing based on extension.
    Returns:
        - patient_hash (DPDP anonymized string)
        - modality (XRAY, CT, or MRI)
        - tensor (PyTorch float tensor)
        - img_base64 (visualization PNG base64 representation)
        - metadata (demographics and study details)
    """
    name_lower = filename.lower()

    if name_lower.endswith(".dcm"):
        dicom_data = parse_dicom(file_content)
        tensor = torch.tensor(dicom_data["normalized_pixels"]).unsqueeze(0).unsqueeze(0)
        img_base64 = convert_pixels_to_base64_png(dicom_data["normalized_pixels"])

        return {
            "patient_hash": dicom_data["patient_hash"],
            "modality": dicom_data["modality"],
            "tensor": tensor,
            "img_base64": img_base64,
            "metadata": {
                "age": dicom_data["patient_age"],
                "sex": dicom_data["patient_sex"],
                "description": dicom_data["study_description"],
            },
        }

    elif name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
        # ✅ FIXED: Use system ephemeral tempfile instead of local directory
        temp_ext = ".nii.gz" if name_lower.endswith(".gz") else ".nii"
        import tempfile
        fd, temp_name = tempfile.mkstemp(suffix=temp_ext)
        with os.fdopen(fd, 'wb') as f:
            f.write(file_content)

        try:
            nii_img = nib.load(temp_name)
            data = nii_img.get_fdata().astype(np.float32)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

        shape = data.shape
        if len(shape) >= 3:
            mid_z = shape[2] // 2
            slice_data = data[:, :, mid_z]
            tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0)
        else:
            slice_data = data
            mid_z = 0
            tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0)

        d_min, d_max = slice_data.min(), slice_data.max()
        normalized_slice = (slice_data - d_min) / (d_max - d_min) if d_max - d_min > 0 else np.zeros_like(slice_data)

        modality = "MRI"
        if "ct" in name_lower or "abdominal" in name_lower:
            modality = "CT"
        elif "xray" in name_lower or "chest" in name_lower:
            modality = "XRAY"

        patient_hash = anonymize_patient("NII_ID", filename)
        img_base64 = convert_pixels_to_base64_png(normalized_slice)
        desc = f"NIfTI 3D Volume — Axial Slice {mid_z + 1}/{shape[2]}" if len(shape) >= 3 else "NIfTI 2D Volume"

        return {
            "patient_hash": patient_hash,
            "modality": modality,
            "tensor": tensor,
            "img_base64": img_base64,
            "metadata": {"age": "N/A", "sex": "N/A", "description": desc},
        }

    else:
        img = Image.open(io.BytesIO(file_content)).convert("L")
        img_np = np.array(img).astype(np.float32) / 255.0

        modality = "XRAY"
        if "mri" in name_lower or "brain" in name_lower:
            modality = "MRI"
        elif "ct" in name_lower or "abdominal" in name_lower:
            modality = "CT"

        patient_hash = anonymize_patient("IMG_ID", filename)
        tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
        img_base64 = convert_pixels_to_base64_png(img_np)

        return {
            "patient_hash": patient_hash,
            "modality": modality,
            "tensor": tensor,
            "img_base64": img_base64,
            "metadata": {"age": "N/A", "sex": "N/A", "description": "Standard Medical Image Ingest"},
        }
