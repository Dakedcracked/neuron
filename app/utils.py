import os
import hashlib
import io
import base64
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image
import torch
import boto3
from app.config import (
    NEURON_CLINIC_SALT,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET_NAME,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_PUBLIC_URL_PREFIX
)

CLINIC_SALT = NEURON_CLINIC_SALT
S3_BUCKET = S3_BUCKET_NAME


def get_s3_client():
    """Initializes and returns a boto3 S3 client configured for R2 object storage."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION
    )


def upload_to_s3(file_content: bytes, filename: str) -> str:
    """
    Streams raw scan bytes directly to Cloudflare R2 / AWS S3 using upload_fileobj.
    Bypasses local file caching completely to ensure cost-efficiency and performance.
    """
    s3_client = get_s3_client()
    fileobj = io.BytesIO(file_content)
    s3_client.upload_fileobj(fileobj, S3_BUCKET, filename)
    return f"{S3_PUBLIC_URL_PREFIX}/{filename}"


def generate_presigned_upload_url(filename: str, expiration=3600) -> dict:
    """
    Generates a secure presigned POST payload for direct browser-to-R2 uploads.
    """
    s3_client = get_s3_client()
    try:
        response = s3_client.generate_presigned_post(
            Bucket=S3_BUCKET,
            Key=filename,
            ExpiresIn=expiration
        )
        return response
    except Exception as e:
        print(f"❌ Error generating presigned post URL: {e}")
        return {}


def ensure_bucket_lifecycle_policy():
    """
    Applies a strict 30-day lifecycle expiration policy to the S3/R2 bucket
    to guarantee DPDP storage compliance and minimize storage overhead costs.
    """
    s3_client = get_s3_client()
    lifecycle_policy = {
        'Rules': [
            {
                'ID': 'DeleteScansAfter30Days',
                'Status': 'Enabled',
                'Filter': {'Prefix': ''},
                'Expiration': {'Days': 30}
            }
        ]
    }
    try:
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET,
            LifecycleConfiguration=lifecycle_policy
        )
        print("✓ Storage Bucket 30-day lifecycle tiered policy enforced successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not configure object lifecycle policy: {e}")


def anonymize_patient(patient_id: str, patient_name: str) -> str:
    """
    Complies with India's Digital Personal Data Protection (DPDP) Act.
    Replaces direct patient identifiers with a secure salted SHA-256 cryptographic hash.
    """
    raw_str = f"{patient_id.strip()}_{patient_name.strip()}_{CLINIC_SALT}"
    hashed = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    return f"PX-{hashed[:20].upper()}"


def apply_hu_window(hu_array: np.ndarray, wc: float, ww: float) -> np.ndarray:
    """Clips and normalizes raw Hounsfield Units to [0, 1] based on Window Center (WC) and Window Width (WW)."""
    v_min = wc - ww / 2.0
    v_max = wc + ww / 2.0
    clipped = np.clip(hu_array, v_min, v_max)
    if v_max - v_min > 0:
        return (clipped - v_min) / (v_max - v_min)
    else:
        return np.zeros_like(clipped)


def log_space_histogram_equalization(pixel_array: np.ndarray) -> np.ndarray:
    """Optimized log-space contrast histogram equalization for X-Ray imaging."""
    shifted = pixel_array - np.min(pixel_array)
    log_transformed = np.log1p(shifted)
    
    flat = log_transformed.flatten()
    f_min = flat.min()
    f_max = flat.max()
    if f_max - f_min > 0:
        flat_norm = (flat - f_min) / (f_max - f_min)
    else:
        flat_norm = np.zeros_like(flat)
    
    hist, bins = np.histogram(flat_norm, bins=256, range=(0, 1))
    cdf = hist.cumsum()
    if cdf[-1] > 0:
        cdf_normalized = cdf / cdf[-1]
    else:
        cdf_normalized = np.zeros_like(cdf)
    
    equalized_flat = np.interp(flat_norm, bins[:-1], cdf_normalized)
    return equalized_flat.reshape(pixel_array.shape)


def adaptive_z_score_normalization(pixel_array: np.ndarray) -> np.ndarray:
    """Adaptive Z-score signal intensity normalization for MRI scans."""
    threshold = np.mean(pixel_array) * 0.1
    foreground = pixel_array[pixel_array > threshold]
    if len(foreground) > 0:
        mean = np.mean(foreground)
        std = np.std(foreground)
    else:
        mean = np.mean(pixel_array)
        std = np.std(pixel_array)
        
    if std > 1e-6:
        z_scored = (pixel_array - mean) / std
    else:
        z_scored = pixel_array - mean
        
    clipped = np.clip(z_scored, -3.0, 3.0)
    return (clipped + 3.0) / 6.0


def build_multi_window_rgb(pixel_array: np.ndarray, modality: str) -> np.ndarray:
    """
    Converts a single-channel 2D or 3D slice array into an optimized 3-channel (RGB) composite array.
    Shape returned: (3, H, W).
    """
    if len(pixel_array.shape) == 3:
        if pixel_array.shape[2] < pixel_array.shape[0] and pixel_array.shape[2] < pixel_array.shape[1]:
            mid_z = pixel_array.shape[2] // 2
            slice_2d = pixel_array[:, :, mid_z]
        else:
            mid_z = pixel_array.shape[0] // 2
            slice_2d = pixel_array[mid_z]
    else:
        slice_2d = pixel_array

    if modality == "CT":
        # Channel R (Soft Tissue): WC = 40, WW = 400
        # Channel G (Lung/Air): WC = -600, WW = 1500
        # Channel B (Bone): WC = 500, WW = 2000
        r = apply_hu_window(slice_2d, 40.0, 400.0)
        g = apply_hu_window(slice_2d, -600.0, 1500.0)
        b = apply_hu_window(slice_2d, 500.0, 2000.0)
    elif modality == "XRAY":
        eq = log_space_histogram_equalization(slice_2d)
        r = eq
        g = eq
        b = eq
    elif modality == "MRI":
        z = adaptive_z_score_normalization(slice_2d)
        r = z
        g = z
        b = z
    else:
        p_min, p_max = slice_2d.min(), slice_2d.max()
        if p_max - p_min > 0:
            norm = (slice_2d - p_min) / (p_max - p_min)
        else:
            norm = np.zeros_like(slice_2d)
        r = norm
        g = norm
        b = norm

    return np.stack([r, g, b], axis=0)


def convert_pixels_to_base64_png(pixel_array: np.ndarray) -> str:
    """
    Converts a normalized 2D numpy array [0, 1] to a base64 encoded PNG string.
    Kept for legacy compatibility.
    """
    scaled = (pixel_array * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(scaled).convert("L")

    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def convert_rgb_to_base64_png(rgb_array: np.ndarray) -> str:
    """
    Converts a normalized 3-channel RGB numpy array [3, H, W] in range [0, 1]
    to a base64 encoded PNG string.
    """
    permuted = np.transpose(rgb_array, (1, 2, 0))
    scaled = (permuted * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(scaled).convert("RGB")

    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_dicom(file_content: bytes) -> dict:
    """
    Parses raw DICOM bytes using pydicom. Extracts headers and returns metadata
    and preserved raw Hounsfield Units (HU) inside multi-window RGB arrays.
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

    slope = 1.0
    intercept = 0.0
    if hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept"):
        slope = float(dataset.RescaleSlope)
        intercept = float(dataset.RescaleIntercept)
        pixel_array = pixel_array * slope + intercept

    # Generate multi-window 3-channel RGB composite
    rgb_composite = build_multi_window_rgb(pixel_array, modality)

    return {
        "patient_hash": patient_hash,
        "modality": modality,
        "normalized_pixels": rgb_composite,
        "raw_pixels": pixel_array,
        "patient_age": getattr(dataset, "PatientAge", "N/A"),
        "patient_sex": getattr(dataset, "PatientSex", "N/A"),
        "study_description": getattr(dataset, "StudyDescription", "General Scan"),
    }


def preprocess_medical_file(file_content: bytes, filename: str) -> dict:
    """
    Accepts raw bytes and delegates processing based on extension.
    Returns:
        - patient_hash (DPDP anonymized string)
        - modality (XRAY, CT, or MRI)
        - tensor (PyTorch float tensor [1, 3, H, W])
        - img_base64 (visualization PNG base64 representation)
        - metadata (demographics and study details)
    """
    name_lower = filename.lower()

    if name_lower.endswith(".dcm"):
        dicom_data = parse_dicom(file_content)
        rgb_arr = dicom_data["normalized_pixels"]
        tensor = torch.tensor(rgb_arr).unsqueeze(0)
        img_base64 = convert_rgb_to_base64_png(rgb_arr)

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

        modality = "MRI"
        if "ct" in name_lower or "abdominal" in name_lower:
            modality = "CT"
        elif "xray" in name_lower or "chest" in name_lower:
            modality = "XRAY"

        rgb_arr = build_multi_window_rgb(data, modality)
        tensor = torch.tensor(rgb_arr).unsqueeze(0)
        img_base64 = convert_rgb_to_base64_png(rgb_arr)
        patient_hash = anonymize_patient("NII_ID", filename)

        shape = data.shape
        mid_z = shape[2] // 2 if len(shape) >= 3 else 0
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
        img_np = np.array(img).astype(np.float32)

        modality = "XRAY"
        if "mri" in name_lower or "brain" in name_lower:
            modality = "MRI"
        elif "ct" in name_lower or "abdominal" in name_lower:
            modality = "CT"

        rgb_arr = build_multi_window_rgb(img_np, modality)
        tensor = torch.tensor(rgb_arr).unsqueeze(0)
        img_base64 = convert_rgb_to_base64_png(rgb_arr)
        patient_hash = anonymize_patient("IMG_ID", filename)

        return {
            "patient_hash": patient_hash,
            "modality": modality,
            "tensor": tensor,
            "img_base64": img_base64,
            "metadata": {"age": "N/A", "sex": "N/A", "description": "Standard Medical Image Ingest"},
        }


def preprocess_for_medsam(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocesses a 2D single-channel or 3-channel image array to form a 1024x1024 input tensor
    conforming to MedSAM's image encoder expectations.
    Normalizes with standard SAM mean=[123.675, 116.28, 103.53] and std=[58.395, 57.12, 57.375].
    Returns:
        np.ndarray: shape (1, 3, 1024, 1024)
    """
    if len(image_array.shape) == 3:
        if image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        if len(image_array.shape) == 3 and image_array.shape[2] > 3:
            mid_z = image_array.shape[2] // 2
            image_array = image_array[:, :, mid_z]
            
    img_min, img_max = image_array.min(), image_array.max()
    if img_max - img_min > 0:
        img_255 = (image_array - img_min) / (img_max - img_min) * 255.0
    else:
        img_255 = np.zeros_like(image_array)
        
    if len(img_255.shape) == 2:
        img_rgb = np.stack([img_255, img_255, img_255], axis=-1)
    elif len(img_255.shape) == 3 and img_255.shape[2] == 1:
        img_rgb = np.concatenate([img_255, img_255, img_255], axis=-1)
    else:
        img_rgb = img_255
        
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(img_rgb.astype(np.uint8))
    img_resized = img_pil.resize((1024, 1024), PILImage.Resampling.BILINEAR)
    img_resized_np = np.array(img_resized).astype(np.float32)
    
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    normalized = (img_resized_np - mean) / std
    
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
