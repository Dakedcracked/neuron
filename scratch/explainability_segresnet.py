# explainability_segresnet.py — Reference implementation for MONAI 3D SegResNet Grad-CAM

import torch
import torch.nn as nn
import numpy as np

class GradCAM3D:
    """
    Grad-CAM implementation tailored for 3D Volumetric Medical Segmentation models.
    Unlike classification models, gradients are backpropagated from the sum of 
    activation scores within the segmented Region of Interest (ROI).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        # Register forward and backward hooks to capture intermediate states
        self._register_hooks()

    def _save_gradient(self, grad):
        self.gradients = grad

    def _forward_hook(self, module, input, output):
        # Captures activations of shape: [1, Channels, Depth, Height, Width]
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        # Captures gradients of shape: [1, Channels, Depth, Height, Width]
        self.gradients = grad_output[0]

    def _register_hooks(self):
        # Register forward hook
        h_f = self.target_layer.register_forward_hook(self._forward_hook)
        self.handlers.append(h_f)
        
        # Register backward hook (compatible with newer PyTorch versions)
        h_b = self.target_layer.register_full_backward_hook(self._backward_hook)
        self.handlers.append(h_b)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generates a 3D saliency map by backpropagating from the predicted mask volume.
        
        input_tensor: [1, Channels, Depth, Height, Width]
        class_idx: target segmentation class (e.g. 0: Tumor Core, 1: Whole Tumor)
        """
        self.model.zero_grad()
        
        # 1. Run forward pass
        outputs = self.model(input_tensor)  # shape: [1, NumClasses, D, H, W]
        
        # Apply sigmoid or softmax to get probability volume
        probs = torch.sigmoid(outputs) if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)
        
        # Extract target class probabilities
        target_probs = probs[0, class_idx]  # shape: [D, H, W]
        
        # 2. Determine region of interest (ROI) to backpropagate from
        # We sum all class probability scores where the probability > 0.5 (threshold)
        mask = (target_probs > 0.5).float()
        
        if mask.sum() == 0:
            # If no pixels are segmented, backprop from the maximum probability voxel
            score = target_probs.max()
        else:
            # Sum the probabilities in the predicted tumor mask region
            score = (target_probs * mask).sum()
            
        # 3. Backward pass to compute gradients with respect to target layer
        score.backward()
        
        # 4. Process activations and gradients
        # Shape: [C, D, H, W]
        activations = self.activations[0].detach().cpu()
        gradients = self.gradients[0].detach().cpu()
        
        # Compute channel importance weights via global average pooling across D, H, W
        weights = torch.mean(gradients, dim=(1, 2, 3))  # shape: [C]
        
        # 5. Compute weighted combination of activations
        cam3d = torch.zeros(activations.shape[1:], dtype=torch.float32)  # shape: [D, H, W]
        for i, w in enumerate(weights):
            cam3d += w * activations[i]
            
        # Apply ReLU to retain only positive features contributing to the segmentation
        cam3d = torch.clamp(cam3d, min=0)
        
        # Normalize heatmap to [0, 1] range
        cam_min, cam_max = cam3d.min(), cam3d.max()
        if cam_max > cam_min:
            cam3d = (cam3d - cam_min) / (cam_max - cam_min)
        else:
            cam3d = torch.zeros_like(cam3d)
            
        return cam3d.numpy()

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks in GPU VRAM."""
        for h in self.handlers:
            h.remove()
        self.handlers.clear()


# ── Usage Demonstration ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing 3D Segmentation Grad-CAM pipeline...")
    
    # 1. Mock a simple 3D CNN model representing SegResNet
    class MockSegResNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Initial feature extraction block
            self.stem = nn.Conv3d(4, 16, kernel_size=3, padding=1)
            # Up-sampling block before classification head
            self.final_conv_block = nn.Conv3d(16, 16, kernel_size=3, padding=1)
            # Segment class logits head
            self.class_head = nn.Conv3d(16, 3, kernel_size=1)  # 3 channels (e.g. TC, WT, ET)

        def forward(self, x):
            x = torch.relu(self.stem(x))
            x = torch.relu(self.final_conv_block(x))
            return self.class_head(x)

    model = MockSegResNet()
    model.eval()
    
    # Target layer is the final convolutional block of the decoder
    target_layer = model.final_conv_block
    
    # 2. Instantiate 3D Grad-CAM manager
    gcam3d = GradCAM3D(model, target_layer)
    
    # 3. Simulate a 3D multi-modal input MRI volume: [Batch, Channels, Depth, Height, Width]
    # Standard BraTS input has 4 channels: T1, T1c, T2, FLAIR
    mri_volume = torch.randn(1, 4, 96, 96, 96)
    
    try:
        # Generate heatmap for Tumor Core (class 0)
        heatmap_3d = gcam3d.generate_heatmap(mri_volume, class_idx=0)
        print(f"✓ 3D Saliency Heatmap generated successfully.")
        print(f"✓ Heatmap dimensions: {heatmap_3d.shape}")  # should be [96, 96, 96]
        print(f"✓ Heatmap min/max: {heatmap_3d.min():.4f} / {heatmap_3d.max():.4f}")
        
        # 4. Extract max activation slice for 2D UI overlay visualization
        # We project the 3D heatmap along the axial plane (Z-axis) to find the slice with largest activation
        axial_sums = np.sum(heatmap_3d, axis=(1, 2))  # sum across H, W
        best_slice_idx = np.argmax(axial_sums)
        print(f"✓ Recommended slice for Radiologist dashboard review: Slice {best_slice_idx} (axial plane)")
        
        heatmap_2d_slice = heatmap_3d[best_slice_idx]
        print(f"✓ Aligned 2D Heatmap Slice shape: {heatmap_2d_slice.shape}")
        
    finally:
        # Always remove hooks to prevent PyTorch leaks
        gcam3d.remove_hooks()
        print("✓ Hooks safely detached from the model.")
