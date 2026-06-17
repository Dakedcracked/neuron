import os
import numpy as np
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    ConcatItemsd,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
)

def read_dicom_series(series_dir: str):
    """
    Reads a folder of 2D DICOM slices and stacks them into a continuous 3D SimpleITK image.
    Uses SimpleITK's ImageSeriesReader to guarantee correct Z-axis ordering.
    """
    if not os.path.exists(series_dir):
        raise ValueError(f"Directory not found: {series_dir}")
        
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
    
    if not dicom_names:
        raise ValueError(f"No valid DICOM slices found in {series_dir}")
        
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def process_clinical_mri(t1_dir: str, t1c_dir: str, t2_dir: str, flair_dir: str) -> np.ndarray:
    """
    Complete Pre-processing Pipeline for Clinical MRI:
    1. Loads the 4 independent DICOM series.
    2. Co-registers them to a common physical space (using T1 as the anchor).
    3. Concatenates them into a 4-channel structure.
    4. Resamples/Pads to a uniform 96x96x96 isotropic grid for ONNX inference.
    """
    print("⏳ [DICOM] Stacking 2D slices into 3D volumes...")
    images = {
        "t1": read_dicom_series(t1_dir),
        "t1c": read_dicom_series(t1c_dir),
        "t2": read_dicom_series(t2_dir),
        "flair": read_dicom_series(flair_dir)
    }
    
    # 2. Resample all modalities to the physical coordinate space of T1 (Basic Co-registration)
    # Note: For heavy patient movement, SimpleElastix (Affine/BSpline) would be used here.
    print("⏳ [DICOM] Co-registering T1c, T2, and FLAIR to T1 spatial anchor...")
    ref_image = images["t1"]
    resampled_arrays = {}
    
    for mod, img in images.items():
        if mod == "t1":
            resampled_arrays[mod] = sitk.GetArrayFromImage(img).astype(np.float32)
            continue
            
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        
        resampled_img = resampler.Execute(img)
        resampled_arrays[mod] = sitk.GetArrayFromImage(resampled_img).astype(np.float32)
        
    # Convert to dictionary format for MONAI Transforms (adding channel dimension)
    data = {
        "t1": np.expand_dims(resampled_arrays["t1"], axis=0),
        "t1c": np.expand_dims(resampled_arrays["t1c"], axis=0),
        "t2": np.expand_dims(resampled_arrays["t2"], axis=0),
        "flair": np.expand_dims(resampled_arrays["flair"], axis=0),
    }
    
    print("⏳ [DICOM] Applying MONAI Isotropic Resampling and Normalization...")
    # 3. MONAI Pipeline: Channel Concatenation, Normalization, and Cropping
    pipeline = Compose([
        # Combine the 4 separate 1-channel volumes into a single 4-channel volume
        ConcatItemsd(keys=["t1", "t1c", "t2", "flair"], name="image", dim=0),
        # Normalize intensities across the volume
        ScaleIntensityd(keys=["image"]),
        # Pad or crop the 3D volume to perfectly match the ONNX runtime input shape
        ResizeWithPadOrCropd(keys=["image"], spatial_size=[96, 96, 96]),
    ])
    
    transformed = pipeline(data)
    
    # Shape is now [4, 96, 96, 96]. We add the batch dimension [1, 4, 96, 96, 96]
    final_tensor = np.expand_dims(transformed["image"], axis=0)
    
    print(f"✓ [DICOM] Pre-processing complete. Final shape: {final_tensor.shape}")
    return final_tensor

