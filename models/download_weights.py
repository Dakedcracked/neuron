"""
download_weights.py — Pre-initialize model weight caches.

For v2.0, torchxrayvision auto-downloads weights on first inference call.
This script can be run optionally to pre-cache them.
"""
import os
import torch

def main():
    print("Pre-caching model weights for Neuron AI v2.0...")
    os.makedirs("models", exist_ok=True)

    # 1. torchxrayvision (X-Ray DenseNet121)
    try:
        import torchxrayvision as xrv
        print("⏳ Downloading torchxrayvision DenseNet121 weights...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        xrv_path = os.path.join("models", "densenet121_xrv.pt")
        if not os.path.exists(xrv_path):
            torch.save(model.state_dict(), xrv_path)
            print(f"✓ Cached torchxrayvision weights to {xrv_path}")
        print("✓ torchxrayvision weights cached successfully.")
    except Exception as e:
        print(f"⚠ torchxrayvision download skipped: {e}")

    # 2. ResNet-50 (ImageNet pretrained) via torchvision
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        print("⏳ Downloading ResNet-50 ImageNet weights...")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet_path = os.path.join("models", "resnet50_clinical.pt")
        if not os.path.exists(resnet_path):
            torch.save(model.state_dict(), resnet_path)
            print(f"✓ Cached ResNet-50 weights to {resnet_path}")
        print("✓ ResNet-50 'clinical' weights cached successfully.")
    except Exception as e:
        print(f"⚠ ResNet-50 download skipped: {e}")

    # 3. MONAI SegResNet (BraTS MRI Segmentation)
    try:
        import urllib.request
        import zipfile
        print("⏳ Downloading MONAI SegResNet (BraTS) weights...")
        
        # Download the bundle zip from MONAI model zoo
        bundle_url = "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/brats_mri_segmentation_v0.3.9.zip"
        zip_path = os.path.join("models", "brats_mri_segmentation.zip")
        extract_dir = os.path.join("models", "brats_mri_segmentation")
        model_path = os.path.join(extract_dir, "models", "model.pt")
        final_path = os.path.join("models", "segresnet_mri.pt")
        
        if not os.path.exists(final_path):
            if not os.path.exists(model_path):
                print("   Downloading zip file from MONAI Hub...")
                urllib.request.urlretrieve(bundle_url, zip_path)
                print("   Extracting zip file...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("models")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            
            # Move the model.pt to our desired location
            os.rename(model_path, final_path)
            print(f"✓ Cached MONAI SegResNet weights to {final_path}")
        else:
            print("✓ MONAI SegResNet 'clinical' weights cached successfully.")
    except Exception as e:
        print(f"⚠ MONAI SegResNet download skipped: {e}")

    print("\nModel pre-caching complete.")

if __name__ == "__main__":
    main()
