# safety_rails.py — Pre-Inference Validation Module

import io
from PIL import Image
import pydicom
import nibabel as nib
import tempfile
import os

class SafetyRailException(Exception):
    """Custom exception raised when a medical scan fails pre-inference safety validations."""
    pass

def validate_dicom_file(content: bytes) -> dict:
    """
    Validates DICOM file structure, magic bytes, essential headers, and checks for corruption.
    """
    if len(content) < 132:
        raise SafetyRailException("File too small to be a valid DICOM scan.")
        
    # Check magic bytes "DICM" at offset 128
    magic = content[128:132]
    if magic != b"DICM":
        raise SafetyRailException("Missing DICOM magic bytes 'DICM' at offset 128.")

    try:
        # stop_before_pixels=True allows parsing header without reading large pixel array
        dicom = pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
        
        # Verify essential tags exist
        required_tags = [
            (0x0008, 0x0060),  # Modality
            (0x0028, 0x0010),  # Rows
            (0x0028, 0x0011),  # Columns
        ]
        
        for group, elem in required_tags:
            if (group, elem) not in dicom:
                tag_name = pydicom.datadict.keyword_for_tag((group, elem)) or f"({group:04x},{elem:04x})"
                raise SafetyRailException(f"Missing essential DICOM tag: {tag_name}")
                
        modality = str(dicom.Modality)
        rows = int(dicom.Rows)
        cols = int(dicom.Columns)
        
        if rows <= 0 or cols <= 0:
            raise SafetyRailException(f"Invalid DICOM spatial dimensions: {rows}x{cols}")
            
        # Perform quick corruption check: read pixel data structure metadata
        # Re-read with pixel data parsed but do not load pixel array into memory fully
        dicom_full = pydicom.dcmread(io.BytesIO(content))
        if not hasattr(dicom_full, "PixelData") or len(dicom_full.PixelData) == 0:
            raise SafetyRailException("DICOM scan does not contain any pixel data or pixel data is empty.")
            
    except SafetyRailException:
        raise
    except Exception as e:
        raise SafetyRailException(f"DICOM file corruption detected: {str(e)}")

    return {
        "format": "DICOM",
        "modality": modality,
        "dimensions": f"{cols}x{rows}",
        "metadata": {
            "manufacturer": str(getattr(dicom, "Manufacturer", "Unknown")),
            "pixel_spacing": str(getattr(dicom, "PixelSpacing", "Unknown")),
            "description": str(getattr(dicom, "StudyDescription", "DICOM Volume")),
        }
    }


def validate_nifti_file(content: bytes) -> dict:
    """
    Validates NIfTI volume integrity by loading the header structure through nibabel.
    """
    if len(content) < 348:
        raise SafetyRailException("File too small to be a valid NIfTI volume.")
        
    # NIfTI-1 header size is exactly 348 bytes.
    # The first 4 bytes should represent 348 (little/big-endian) or 540 for NIfTI-2.
    header_size_le = int.from_bytes(content[0:4], byteorder='little')
    header_size_be = int.from_bytes(content[0:4], byteorder='big')
    if header_size_le != 348 and header_size_be != 348 and header_size_le != 540 and header_size_be != 540:
        raise SafetyRailException("Invalid NIfTI header structure (mismatched header size magic).")

    fd, temp_path = tempfile.mkstemp(suffix=".nii")
    try:
        with os.fdopen(fd, 'wb') as temp_file:
            temp_file.write(content)
            
        # Attempt to load through nibabel to check for file structure truncation
        nii = nib.load(temp_path)
        shape = nii.shape
        if len(shape) < 2:
            raise SafetyRailException(f"Invalid NIfTI dimension count: {len(shape)}")
            
    except SafetyRailException:
        raise
    except Exception as e:
        raise SafetyRailException(f"NIfTI file corruption detected: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "format": "NIfTI",
        "modality": "Unknown",  # To be inferred from filename or in-process parsing
        "dimensions": "x".join(map(str, shape)),
        "metadata": {
            "description": "NIfTI Volumetric Scan",
            "pixel_spacing": "1.0mm x 1.0mm x 1.0mm"
        }
    }


def validate_standard_image(content: bytes) -> dict:
    """
    Validates standard 2D image file formats (PNG, JPG, JPEG) using Pillow.
    """
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # Verifies file integrity without loading actual pixels
        
        # Re-open for dimension checks (verify closes the file)
        img = Image.open(io.BytesIO(content))
        width, height = img.size
        
        if width <= 0 or height <= 0:
            raise SafetyRailException(f"Invalid standard image dimensions: {width}x{height}")
            
    except Exception as e:
        raise SafetyRailException(f"Image file corruption or unsupported standard format: {str(e)}")

    return {
        "format": "StandardImage",
        "modality": "Unknown",
        "dimensions": f"{width}x{height}",
        "metadata": {
            "description": "2D Radiograph Projection",
            "pixel_spacing": "Unknown"
        }
    }


def run_pre_inference_validation(content: bytes, filename: str) -> dict:
    """
    Executes pre-inference validation based on file extension.
    """
    ext = filename.lower()
    if ext.endswith(".dcm"):
        return validate_dicom_file(content)
    elif ext.endswith((".nii", ".nii.gz")):
        return validate_nifti_file(content)
    elif ext.endswith((".png", ".jpg", ".jpeg")):
        return validate_standard_image(content)
    else:
        raise SafetyRailException("Unsupported clinical imaging format.")
