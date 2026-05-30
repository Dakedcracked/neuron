import os
import numpy as np
from PIL import Image, ImageDraw
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import nibabel as nib

def create_mock_xray_png():
    # 256x256 image
    img = Image.new("L", (256, 256), color=20)
    draw = ImageDraw.Draw(img)
    # Draw lungs (darker regions)
    draw.ellipse([40, 50, 110, 200], fill=10)
    draw.ellipse([140, 50, 210, 200], fill=10)
    # Spine/mediastinum (brighter center)
    draw.rectangle([115, 40, 135, 220], fill=60)
    # Clavicles
    draw.line([30, 60, 115, 75], fill=80, width=5)
    draw.line([210, 60, 135, 75], fill=80, width=5)
    # Ribs
    for y in range(80, 200, 20):
        draw.line([40, y, 110, y + 5], fill=40, width=2)
        draw.line([210, y, 140, y + 5], fill=40, width=2)
        
    os.makedirs("test_scans", exist_ok=True)
    img.save("test_scans/chest_xray_sample.png")
    print("✓ Mock Chest X-Ray PNG created at test_scans/chest_xray_sample.png")

def create_mock_dicom():
    filename = "test_scans/brain_mri_sample.dcm"
    
    # Create file meta info
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4' # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7"
    file_meta.ImplementationClassUID = "1.2.3.4"
    
    # Create dataset
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Patient Demographic Info (direct PII to be anonymized by utils.py)
    ds.PatientID = "PT-94827-X"
    ds.PatientName = "Varun Sharma"
    ds.PatientBirthDate = "19880415"
    ds.PatientSex = "M"
    ds.PatientAge = "038Y"
    
    # Study info
    ds.Modality = "MR"
    ds.StudyDescription = "Brain MRI Contrast"
    
    # Image metrics
    ds.Rows = 128
    ds.Columns = 128
    ds.NumberOfFrames = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    
    # Create mock brain pixel array
    pixel_array = np.zeros((128, 128), dtype=np.uint16)
    y, x = np.ogrid[:128, :128]
    mask = (x - 64)**2 + (y - 64)**2 < 45**2
    pixel_array[mask] = 1000
    # Ventricles (darker inside)
    v1 = (x - 55)**2 + (y - 60)**2 < 12**2
    v2 = (x - 73)**2 + (y - 60)**2 < 12**2
    pixel_array[v1] = 200
    pixel_array[v2] = 200
    
    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    os.makedirs("test_scans", exist_ok=True)
    ds.save_as(filename)
    print(f"✓ Mock DICOM MRI created at {filename}")

def create_mock_nifti():
    filename = "test_scans/abdominal_ct_sample.nii"
    
    # Create 3D volume [64, 64, 32]
    data = np.zeros((64, 64, 32), dtype=np.float32)
    # Fill in a sphere/organ shape
    y, x, z = np.ogrid[:64, :64, :32]
    mask = (x - 32)**2 + (y - 32)**2 + (z - 16)**2 < 20**2
    data[mask] = 400.0
    
    # Liver lesion stub
    lesion = (x - 24)**2 + (y - 28)**2 + (z - 14)**2 < 6**2
    data[lesion] = 150.0
    
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)
    print(f"✓ Mock NIfTI CT created at {filename}")

if __name__ == "__main__":
    create_mock_xray_png()
    create_mock_dicom()
    create_mock_nifti()
