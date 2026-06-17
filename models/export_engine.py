"""
export_engine.py — Compiles PyTorch medical models to ONNX and TensorRT

This script:
1. Loads the downloaded MONAI SegResNet weights.
2. Strips any unused regularization branches (e.g., VAE head).
3. Exports the model to a lightweight ONNX format.
"""
import os
import torch

try:
    from monai.networks.nets import SegResNet
except ImportError:
    raise ImportError("MONAI is required. Please install it using: pip install monai")

def export_segresnet_to_onnx():
    print("⏳ Initializing SegResNet architecture for MRI...")
    # The BraTS bundle SegResNet uses 4 input channels and 3 output channels
    # We initialize it with use_vae=False so the VAE head is stripped during inference
    model = SegResNet(
        spatial_dims=3,
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1]
    )

    weight_path = os.path.join("models", "segresnet_mri.pt")
    if not os.path.exists(weight_path):
        print(f"⚠ Could not find weights at {weight_path}. Run download_weights.py first.")
        return

    print("⏳ Loading pre-trained weights (ignoring missing VAE keys)...")
    state_dict = torch.load(weight_path, map_location="cpu")
    
    # The bundle saves the model under 'state_dict' or as the raw dictionary
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Load with strict=False because the saved weights might contain VAE parameters
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Isotropic pooling for edge deployment: 96x96x96
    # Batch size 1, 4 channels (since BraTS expects T1, T1c, T2, FLAIR)
    dummy_input = torch.randn(1, 4, 96, 96, 96)
    
    onnx_path = os.path.join("models", "segresnet_mri.onnx")
    print(f"⏳ Exporting to ONNX format at {onnx_path}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    print(f"✓ ONNX export complete! Saved to {onnx_path}")
    print("  Next steps: Use TensorRT 'trtexec' to compile this ONNX to an FP16 engine:")
    print(f"  trtexec --onnx={onnx_path} --saveEngine=models/segresnet_mri.engine --fp16")

if __name__ == "__main__":
    export_segresnet_to_onnx()
