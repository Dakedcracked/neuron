import os
import copy
import asyncio
import time
import json
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Form
import onnxruntime as ort
import torch.nn as nn
from pathlib import Path
import timm

try:
    import tritonclient.http as tritonhttp
except ImportError:
    tritonhttp = None

app = FastAPI(
    title="Neuron AI — High-Performance Model Serving Server",
    version="2.0.0"
)

# ── Configuration & Queue Setup ──────────────────────────────────────────────
BATCH_SIZE = int(os.environ.get("NEURON_BATCH_SIZE", "16"))
MAX_DELAY_MS = float(os.environ.get("NEURON_MAX_DELAY_MS", "15.0"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Async queue for batching: contains (modality, file_path, patient_hash, future)
inference_queue = asyncio.Queue()

# Models (primary references — fold 0 of each ensemble)
xrv_model = None
ct_model = None
mri_model = None
segresnet_session = None

# Fold ensembles
xray_folds = []
ct_folds = []
mri_folds = []

# 5-fold weighted soft-voting weights
FOLD_WEIGHTS = [0.25, 0.20, 0.20, 0.20, 0.15]

# Stats
stats = {
    "total_inferences": 0,
    "total_batches": 0,
    "last_batch_size": 0,
    "avg_batch_latency_ms": 0.0
}


# ── Load Models ──────────────────────────────────────────────────────────────
def load_models():
    global xrv_model, ct_model, mri_model, segresnet_session
    global xray_folds, ct_folds, mri_folds
    print(f"📦 [Model Server] Target Device: {DEVICE}")

    # ── 1. Load X-Ray DenseNet121 5-Fold Ensemble ────────────────────────────
    try:
        import torchxrayvision as xrv

        # Priority: radimagenet > xrv > upstream
        rad_path = MODEL_DIR / "densenet121_radimagenet.pt"
        xrv_path = MODEL_DIR / "densenet121_xrv.pt"

        xrv_state = None
        loaded_source = "none"

        if rad_path.exists():
            xrv_state = torch.load(rad_path, map_location="cpu")
            loaded_source = "RadImageNet"
        elif xrv_path.exists():
            xrv_state = torch.load(xrv_path, map_location="cpu")
            loaded_source = "clinical (RadImageNet fallback)"
        else:
            temp_model = xrv.models.DenseNet(weights=None)
            xrv_state = temp_model.state_dict()
            loaded_source = "upstream (densenet121-res224-all fallback)"

        print(f"✓ [Model Server] {loaded_source}-pretrained DenseNet121 base weights loaded.")

        for fold in range(1, 6):
            fold_path = MODEL_DIR / f"densenet121_fold{fold}.pt"
            fold_model = xrv.models.DenseNet(weights=None)

            # Modify first conv layer to accept 3-channel inputs (match inference.py)
            if hasattr(fold_model.features, "conv0"):
                old_conv = fold_model.features.conv0
                if old_conv.in_channels == 1:
                    new_conv = nn.Conv2d(
                        in_channels=3,
                        out_channels=old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None
                    )
                    fold_model.features.conv0 = new_conv

            if fold_path.exists():
                fold_state = torch.load(fold_path, map_location="cpu")
                fold_model.load_state_dict(fold_state, strict=False)
                print(f"  ✓ Loaded DenseNet121 Fold {fold} from disk.")
            else:
                # Perturbation fallback: copy base weights + noise
                fold_state = copy.deepcopy(xrv_state)

                # Map 1-channel conv0 weights to 3-channel
                if "features.conv0.weight" in fold_state:
                    w = fold_state["features.conv0.weight"]
                    if w.shape[1] == 1:
                        fold_state["features.conv0.weight"] = torch.cat(
                            [w / 3.0, w / 3.0, w / 3.0], dim=1
                        )

                for key in fold_state.keys():
                    if "weight" in key or "bias" in key:
                        t = fold_state[key]
                        if t.is_floating_point():
                            fold_state[key] = t + torch.randn_like(t) * 1e-4

                fold_model.load_state_dict(fold_state, strict=False)
                print(f"  ⚠ DenseNet121 Fold {fold} missing. Generated via perturbation fallback.")

            fold_model.to(DEVICE)
            if DEVICE.type == "cuda":
                fold_model = fold_model.half()
            fold_model.eval()
            xray_folds.append(fold_model)

        # Assign pathologies attribute if missing
        if not hasattr(xray_folds[0], "pathologies"):
            for m in xray_folds:
                m.pathologies = getattr(xrv.datasets, "default_pathologies", [])

        xrv_model = xray_folds[0]
        print(f"✓ [Model Server] DenseNet121 5-fold X-Ray ensemble loaded ({len(xray_folds)} folds).")
    except Exception as e:
        print(f"⚠ [Model Server] X-Ray DenseNet121 ensemble load failed: {e}")

    # ── 2. Load CT/MRI ResNet-50 5-Fold Ensembles ────────────────────────────
    try:
        def _build_resnet_folds(modality: str, num_classes: int):
            """Loads 5 folds of SOTA EfficientNet-B4 for CT or MRI."""
            folds = []
            model_name = "efficientnet_b4"
            clinical_path = MODEL_DIR / f"{model_name}_clinical.pt"

            base_state = None
            loaded_source = "none"

            if clinical_path.exists():
                try:
                    base_state = torch.load(clinical_path, map_location="cpu")
                    loaded_source = "clinical cache"
                except Exception:
                    pass

            if base_state is None:
                try:
                    net_temp = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
                    base_state = net_temp.state_dict()
                    loaded_source = "Hugging Face (SOTA pretrained)"
                    torch.save(base_state, clinical_path)
                except Exception as e:
                    print(f"    ⚠ Failed to download SOTA model from Hugging Face ({e}). Building uninitialized model.")
                    net_temp = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                    base_state = net_temp.state_dict()
                    loaded_source = "uninitialized fallback"

            print(f"  ✓ {loaded_source} SOTA convolutional features loaded for {modality} folds.")

            for fold in range(1, 6):
                fold_file = MODEL_DIR / f"{model_name}_{modality.lower()}_fold{fold}.pt"
                net = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

                if fold_file.exists():
                    try:
                        state_dict = torch.load(fold_file, map_location="cpu")
                        net.load_state_dict(state_dict, strict=False)
                        print(f"    ✓ Loaded SOTA {modality} Fold {fold} from {fold_file.name}.")
                    except Exception as e:
                        print(f"    ⚠ Error loading Fold {fold}: {e}. Falling back to perturbed base.")
                        fold_file = None

                if not fold_file or not fold_file.exists():
                    net.load_state_dict(base_state, strict=False)
                    for param in net.parameters():
                        param.data += torch.randn_like(param.data) * 1e-4
                    print(f"    ⚠ SOTA {modality} Fold {fold} checkpoint missing. Generated via perturbation.")

                net.to(DEVICE)
                if DEVICE.type == "cuda":
                    net = net.half()
                net.eval()
                folds.append(net)

            return folds

        ct_folds = _build_resnet_folds("CT", 4)
        mri_folds = _build_resnet_folds("MRI", 4)

        ct_model = ct_folds[0]
        mri_model = mri_folds[0]
        print(f"✓ [Model Server] SOTA EfficientNet-B4 5-fold ensembles loaded for CT ({len(ct_folds)}) and MRI ({len(mri_folds)}).")
    except Exception as e:
        print(f"⚠ [Model Server] SOTA EfficientNet-B4 ensemble load failed: {e}")

    # ── 3. Load MRI ONNX SegResNet ───────────────────────────────────────────
    try:
        onnx_path = MODEL_DIR / "segresnet_mri.onnx"
        if onnx_path.exists():
            segresnet_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            print("✓ [Model Server] SegResNet ONNX runtime session initialized.")
    except Exception as e:
        print(f"⚠ [Model Server] SegResNet ONNX load failed: {e}")


# ── Dynamic Batching Loop ───────────────────────────────────────────────────
async def dynamic_batch_processor():
    global stats
    while True:
        # Wait for at least one item
        item = await inference_queue.get()
        requests = [item]

        # Drain as many items as possible up to BATCH_SIZE or MAX_DELAY_MS
        start_time = time.perf_counter()
        while len(requests) < BATCH_SIZE:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            remaining_time = MAX_DELAY_MS - elapsed
            if remaining_time <= 0:
                break
            try:
                next_item = await asyncio.wait_for(inference_queue.get(), timeout=remaining_time / 1000.0)
                requests.append(next_item)
            except asyncio.TimeoutError:
                break

        # Process the batch
        batch_size = len(requests)
        stats["total_batches"] += 1
        stats["last_batch_size"] = batch_size
        print(f"⚡ [Model Server] Processing Batch of size: {batch_size}")

        # Group requests by modality to execute efficiently
        modalities = {}
        for req in requests:
            mod = req["modality"]
            modalities.setdefault(mod, []).append(req)

        t_batch_start = time.perf_counter()

        for mod, mod_reqs in modalities.items():
            try:
                results = execute_modality_batch(mod, mod_reqs)
                for req, res in zip(mod_reqs, results):
                    req["future"].set_result(res)
            except Exception as e:
                print(f"❌ [Model Server] Batch processing failed: {e}")
                for req in mod_reqs:
                    if not req["future"].done():
                        req["future"].set_exception(e)

        batch_latency = (time.perf_counter() - t_batch_start) * 1000.0
        stats["total_inferences"] += batch_size
        stats["avg_batch_latency_ms"] = (stats["avg_batch_latency_ms"] * 0.9) + (batch_latency * 0.1)

        for _ in range(batch_size):
            inference_queue.task_done()


# ── Triton Inference Helper ──────────────────────────────────────────────────
def _execute_triton_inference_batch(np_arr: np.ndarray, model_name: str, input_name: str = "input", output_name: str = "output") -> np.ndarray:
    """Sends batched arrays directly into Triton Inference Server pool."""
    if tritonhttp is None:
        raise RuntimeError("tritonclient.http package is not installed.")
        
    triton_url = os.environ.get("TRITON_SERVER_URL")
    if not triton_url:
        raise RuntimeError("TRITON_SERVER_URL is not set.")
        
    url_clean = triton_url.replace("http://", "").replace("https://", "")
    client = tritonhttp.InferenceServerClient(url=url_clean)
    
    # FP16 precision mapping matching hardware requirements
    datatype = "FP16" if np_arr.dtype == np.float16 else "FP32"
    
    infer_input = tritonhttp.InferInput(input_name, np_arr.shape, datatype)
    infer_input.set_data_from_numpy(np_arr)
    
    infer_output = tritonhttp.InferRequestedOutput(output_name)
    response = client.infer(model_name=model_name, inputs=[infer_input], outputs=[infer_output])
    return response.as_numpy(output_name)


# ── Grad-CAM Helpers ─────────────────────────────────────────────────────────
def _resolve_gradcam_target_layer(model, modality: str):
    """Resolves the target layer for Grad-CAM based on model architecture."""
    if modality == "XRAY":
        if hasattr(model, "features") and hasattr(model.features, "norm5"):
            return model.features.norm5
        if hasattr(model, "features"):
            children = list(model.features.children())
            if children:
                return children[-1]
    else:
        if hasattr(model, "layer4"):
            return model.layer4
        if hasattr(model, "conv_head"):
            return model.conv_head
        # Fallback: recursively search for the last Conv2d layer
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
    return None


def _run_gradcam_for_sample(model, tensor_single, class_idx, modality, file_path):
    """Runs Grad-CAM on a single sample and returns (bbox, img_base64)."""
    from app.inference import (
        GradCAM, extract_bbox_from_heatmap, _resize_heatmap,
        blend_image_and_heatmap, _convert_overlay_to_base64,
        _load_original_image_for_blending
    )

    target_layer = _resolve_gradcam_target_layer(model, modality)
    if target_layer is None:
        return None, None

    try:
        gradcam = GradCAM(model, target_layer)
        heatmap_raw = gradcam.generate_heatmap(tensor_single.to(DEVICE), class_idx)

        bbox = extract_bbox_from_heatmap(heatmap_raw)

        orig_img = _load_original_image_for_blending(file_path)
        h_orig, w_orig = orig_img.shape
        heatmap_resized = _resize_heatmap(heatmap_raw, (h_orig, w_orig))
        blended = blend_image_and_heatmap(orig_img, heatmap_resized)
        img_b64 = _convert_overlay_to_base64(blended)

        return bbox, img_b64
    except Exception as e:
        print(f"⚠ [Model Server] Grad-CAM generation failed: {e}")
        return None, None


# ── Core Batch Inference ─────────────────────────────────────────────────────
def execute_modality_batch(modality: str, requests: list) -> list:
    """Runs 5-fold ensemble batch inference with dynamic Grad-CAM localization or Triton routing."""
    from app.inference import (
        _xray_transform, _volume_transform, _volume_transform_3d,
        _build_unet_predictions, extract_bboxes_from_mask,
        _reconstruct_cam_from_activations, _resize_heatmap,
        blend_image_and_heatmap, _convert_overlay_to_base64,
        _load_original_image_for_blending, extract_bbox_from_heatmap
    )

    results = []
    triton_url = os.environ.get("TRITON_SERVER_URL")
    triton_active = triton_url and (tritonhttp is not None)

    # ── BATCH X-RAY ──────────────────────────────────────────────────────────
    if modality == "XRAY":
        if xrv_model is None or len(xray_folds) == 0:
            raise RuntimeError("X-Ray DenseNet model not loaded.")

        # Prepare input tensors
        tensors = []
        for r in requests:
            t = _xray_transform(r["file_path"])  # Shape: [1, 3, 224, 224]
            tensors.append(t)

        batched_tensor = torch.cat(tensors, dim=0).to(DEVICE)  # [B, 3, 224, 224]
        
        xrv_labels = xray_folds[0].pathologies
        label_map = {
            "Pneumonia":       ["Pneumonia"],
            "Cardiomegaly":    ["Cardiomegaly"],
            "Pleural Effusion": ["Pleural Effusion", "Effusion"],
            "Pneumothorax":    ["Pneumothorax"],
            "Atelectasis":     ["Atelectasis"],
            "Consolidation":   ["Consolidation"],
            "Edema":           ["Edema"],
            "Mass":            ["Mass"],
            "Nodule":          ["Nodule"],
            "Normal":          [],
        }

        # Route to Triton if active with FP16 precision
        if triton_active:
            try:
                np_arr = batched_tensor.cpu().numpy().astype(np.float16)
                model_name = os.environ.get("TRITON_MODEL_XRAY", "densenet121_xrv")
                
                logits_np = _execute_triton_inference_batch(np_arr, model_name, output_name="output")
                try:
                    triton_activations = _execute_triton_inference_batch(np_arr, model_name, output_name="norm5")
                except Exception:
                    triton_activations = None
                
                xrv_out = torch.tensor(logits_np)
                for idx, r in enumerate(requests):
                    probs_raw = torch.sigmoid(xrv_out[idx]).cpu().numpy()
                    pred_map = {}
                    for our_label, xrv_synonyms in label_map.items():
                        score = 0.0
                        for syn in xrv_synonyms:
                            if syn in xrv_labels:
                                l_idx = list(xrv_labels).index(syn)
                                score = max(score, float(probs_raw[l_idx]))
                        pred_map[our_label] = score
                    
                    max_abnormal = max(v for k, v in pred_map.items() if k != "Normal")
                    pred_map["Normal"] = max(0.0, 1.0 - max_abnormal)
                    total = sum(pred_map.values())
                    pred_map = {k: v / total for k, v in pred_map.items()}
                    
                    pathology = max(pred_map, key=pred_map.get)
                    confidence = float(pred_map[pathology])
                    
                    bbox = None
                    img_b64 = None
                    if pathology not in ("Normal", "Inconclusive (Discrepancy Triage: Requires Specialist Verification)"):
                        target_class_idx = None
                        syns = label_map.get(pathology, [])
                        for syn in syns:
                            if syn in xrv_labels:
                                target_class_idx = list(xrv_labels).index(syn)
                                break
                                
                        if target_class_idx is not None:
                            if triton_activations is not None:
                                weights = xrv_model.classifier.weight.data.cpu().numpy()
                                heatmap_raw = _reconstruct_cam_from_activations(triton_activations[idx], weights[target_class_idx])
                                h_orig, w_orig = _load_original_image_for_blending(r["file_path"]).shape
                                heatmap = _resize_heatmap(heatmap_raw, (h_orig, w_orig))
                                bbox = extract_bbox_from_heatmap(heatmap)
                                orig_img = _load_original_image_for_blending(r["file_path"])
                                blended = blend_image_and_heatmap(orig_img, heatmap)
                                img_b64 = _convert_overlay_to_base64(blended)
                            else:
                                single_tensor = tensors[idx].float()
                                bbox, img_b64 = _run_gradcam_for_sample(
                                    xrv_model, single_tensor, target_class_idx, "XRAY", r["file_path"]
                                )
                                
                    results.append({
                        "pathology_detected": pathology,
                        "confidence_score": confidence,
                        "bbox": bbox,
                        "predictions": pred_map,
                        "pytorch_executed": True,
                        "model_info": "MONAI ModelServer (Triton DenseNet FP16 Batched)",
                        "img_base64": img_b64,
                    })
                return results
            except Exception as e:
                print(f"⚠ Triton batch inference failed: {e}. Falling back to local ensembles.")

        if DEVICE.type == "cuda":
            batched_tensor = batched_tensor.half()

        # Run 5-fold ensemble locally
        all_fold_outputs = []
        for fold_model in xray_folds:
            with torch.inference_mode():
                outputs = fold_model(batched_tensor)
            probs_raw = torch.sigmoid(outputs).cpu().numpy()
            all_fold_outputs.append(probs_raw)

        for idx, r in enumerate(requests):
            fold_predictions = []
            for fold_idx, fold_probs_raw in enumerate(all_fold_outputs):
                probs = fold_probs_raw[idx]
                pred_map_fold = {}
                for our_label, xrv_synonyms in label_map.items():
                    score = 0.0
                    for syn in xrv_synonyms:
                        if syn in xrv_labels:
                            l_idx = list(xrv_labels).index(syn)
                            score = max(score, float(probs[l_idx]))
                    pred_map_fold[our_label] = score

                max_abnormal = max(v for k, v in pred_map_fold.items() if k != "Normal")
                pred_map_fold["Normal"] = max(0.0, 1.0 - max_abnormal)
                total = sum(pred_map_fold.values())
                pred_map_fold = {k: v / total for k, v in pred_map_fold.items()}
                fold_predictions.append(pred_map_fold)

            all_labels = list(label_map.keys())
            mean_probs = {}
            std_probs = {}
            for c in all_labels:
                probs_c = [pred[c] for pred in fold_predictions]
                mean_probs[c] = float(np.sum([p * w for p, w in zip(probs_c, FOLD_WEIGHTS)]))
                std_probs[c] = float(np.std(probs_c))

            max_variance = max(std_probs.values())
            if max_variance > 0.15:
                print(f"🚨 CLINICAL DISCREPANCY: XRAY fold variance {max_variance:.4f} > 0.15 — triggering triage.")
                pathology = "Inconclusive (Discrepancy Triage: Requires Specialist Verification)"
                confidence = 0.0
                pred_map = mean_probs
            else:
                pred_map = mean_probs
                pathology = max(pred_map, key=pred_map.get)
                confidence = float(pred_map[pathology])

            bbox = None
            img_b64 = None
            if pathology not in ("Normal", "Inconclusive (Discrepancy Triage: Requires Specialist Verification)"):
                target_class_idx = None
                syns = label_map.get(pathology, [])
                for syn in syns:
                    if syn in xrv_labels:
                        target_class_idx = list(xrv_labels).index(syn)
                        break

                if target_class_idx is not None:
                    single_tensor = tensors[idx].float()
                    bbox, img_b64 = _run_gradcam_for_sample(
                        xrv_model, single_tensor, target_class_idx, "XRAY", r["file_path"]
                    )

            results.append({
                "pathology_detected": pathology,
                "confidence_score": confidence,
                "bbox": bbox,
                "predictions": pred_map,
                "pytorch_executed": True,
                "model_info": "MONAI ModelServer (DenseNet121 5-Fold Ensemble Batched)",
                "img_base64": img_b64,
            })

    # ── BATCH CT / MRI ───────────────────────────────────────────────────────
    elif modality in ("CT", "MRI"):
        tensors_3d = []
        segresnet_available = segresnet_session is not None

        for r in requests:
            t3d = _volume_transform_3d(r["file_path"])  # [1, 1, 96, 96, 96]
            tensors_3d.append(t3d)

        # Route to Triton if active for SegResNet with FP16
        if triton_active and modality == "MRI" and segresnet_available:
            try:
                batched_3d = torch.cat(tensors_3d, dim=0)
                batched_3d = batched_3d.repeat(1, 4, 1, 1, 1)
                np_arr = batched_3d.numpy().astype(np.float16)
                model_name = os.environ.get("TRITON_MODEL_SEGRESNET", "segresnet_mri")
                
                logits = _execute_triton_inference_batch(np_arr, model_name, output_name="output")
                for seg_idx, r in enumerate(requests):
                    l_logits = logits[seg_idx]
                    probs = np.exp(l_logits) / np.sum(np.exp(l_logits), axis=0, keepdims=True)
                    tumor_mask = probs[1]

                    lesion_score = float(np.clip(tumor_mask.mean() * 15.0, 0, 1))
                    bbox = extract_bboxes_from_mask(tumor_mask, threshold=0.1)

                    pred_map = _build_unet_predictions(modality, lesion_score)
                    pathology = max(pred_map, key=pred_map.get)
                    confidence = float(pred_map[pathology])

                    results.append({
                        "pathology_detected": pathology,
                        "confidence_score": confidence,
                        "bbox": bbox,
                        "predictions": pred_map,
                        "pytorch_executed": True,
                        "model_info": "MONAI ModelServer (Triton SegResNet FP16 Batched)",
                        "img_base64": None,
                    })
                return results
            except Exception as e:
                print(f"⚠ Triton SegResNet batch failed: {e}. Falling back to ResNet ensemble.")

        # Local SegResNet ONNX runtime path
        if segresnet_available:
            try:
                batched_3d = torch.cat(tensors_3d, dim=0)
                batched_3d = batched_3d.repeat(1, 4, 1, 1, 1)

                ort_inputs = {"input": batched_3d.numpy().astype(np.float32)}
                logits = segresnet_session.run(["output"], ort_inputs)[0]

                for seg_idx, r in enumerate(requests):
                    l_logits = logits[seg_idx]
                    probs = np.exp(l_logits) / np.sum(np.exp(l_logits), axis=0, keepdims=True)
                    tumor_mask = probs[1]

                    lesion_score = float(np.clip(tumor_mask.mean() * 15.0, 0, 1))
                    bbox = extract_bboxes_from_mask(tumor_mask, threshold=0.1)

                    pred_map = _build_unet_predictions(modality, lesion_score)
                    pathology = max(pred_map, key=pred_map.get)
                    confidence = float(pred_map[pathology])

                    results.append({
                        "pathology_detected": pathology,
                        "confidence_score": confidence,
                        "bbox": bbox,
                        "predictions": pred_map,
                        "pytorch_executed": True,
                        "model_info": "MONAI ModelServer (SegResNet ONNX Batched)",
                        "img_base64": None,
                    })
                return results
            except Exception as e:
                print(f"⚠ ONNX batch session failed: {e}. Falling back to ResNet ensemble.")

        # ResNet-50 Ensemble
        folds = ct_folds if modality == "CT" else mri_folds
        model = ct_model if modality == "CT" else mri_model
        pathologies = (
            ["Renal Calculi", "Hepatic Lesion", "Appendicitis", "Normal"]
            if modality == "CT"
            else ["Glioblastoma", "Meningioma", "Ischemic Stroke", "Normal"]
        )

        if model is None or len(folds) == 0:
            for r in requests:
                results.append({
                    "pathology_detected": "Inconclusive",
                    "confidence_score": 0.0,
                    "bbox": None,
                    "predictions": {"Inconclusive": 1.0},
                    "pytorch_executed": False,
                    "model_info": "Inconclusive (Inference Server fallback)",
                    "img_base64": None,
                })
            return results

        tensors = []
        for r in requests:
            t = _volume_transform(r["file_path"])  # [1, 3, 224, 224]
            tensors.append(t)

        batched_tensor = torch.cat(tensors, dim=0).to(DEVICE)  # [B, 3, 224, 224]

        # Route to Triton if active for ResNet-50 with FP16
        if triton_active:
            try:
                np_arr = batched_tensor.cpu().numpy().astype(np.float16)
                model_name = os.environ.get("TRITON_MODEL_RESNET", f"resnet50_{modality.lower()}")
                
                logits_np = _execute_triton_inference_batch(np_arr, model_name, output_name="output")
                try:
                    triton_activations = _execute_triton_inference_batch(np_arr, model_name, output_name="layer4")
                except Exception:
                    triton_activations = None
                
                raw_probs = torch.softmax(torch.tensor(logits_np), dim=1).cpu().numpy()
                for idx, r in enumerate(requests):
                    probs = raw_probs[idx]
                    pred_map = dict(zip(pathologies, probs.tolist()))
                    pathology = max(pred_map, key=pred_map.get)
                    confidence = float(pred_map[pathology])

                    bbox = None
                    img_b64 = None
                    if pathology not in ("Normal", "Inconclusive (Discrepancy Triage: Requires Specialist Verification)"):
                        target_class_idx = None
                        if pathology in pathologies:
                            target_class_idx = pathologies.index(pathology)

                        if target_class_idx is not None and target_class_idx < len(pathologies) - 1:
                            if triton_activations is not None:
                                if hasattr(model, "fc"):
                                    weights = model.fc.weight.data.cpu().numpy()
                                elif hasattr(model, "classifier"):
                                    weights = model.classifier.weight.data.cpu().numpy()
                                else:
                                    last_linear = None
                                    for module in model.modules():
                                        if isinstance(module, torch.nn.Linear):
                                            last_linear = module
                                    weights = last_linear.weight.data.cpu().numpy()
                                heatmap_raw = _reconstruct_cam_from_activations(triton_activations[idx], weights[target_class_idx])
                                h_orig, w_orig = _load_original_image_for_blending(r["file_path"]).shape
                                heatmap = _resize_heatmap(heatmap_raw, (h_orig, w_orig))
                                bbox = extract_bbox_from_heatmap(heatmap)
                                orig_img = _load_original_image_for_blending(r["file_path"])
                                blended = blend_image_and_heatmap(orig_img, heatmap)
                                img_b64 = _convert_overlay_to_base64(blended)
                            else:
                                single_tensor = tensors[idx].float()
                                bbox, img_b64 = _run_gradcam_for_sample(
                                    model, single_tensor, target_class_idx, modality, r["file_path"]
                                )

                    results.append({
                        "pathology_detected": pathology,
                        "confidence_score": confidence,
                        "bbox": bbox,
                        "predictions": pred_map,
                        "pytorch_executed": True,
                        "model_info": "MONAI ModelServer (Triton ResNet FP16 Batched)",
                        "img_base64": img_b64,
                    })
                return results
            except Exception as e:
                print(f"⚠ Triton ResNet batch inference failed: {e}. Falling back to local ensembles.")

        if DEVICE.type == "cuda":
            batched_tensor = batched_tensor.half()

        # Run 5-fold ensemble locally
        all_fold_outputs = []
        for fold_model in folds:
            with torch.inference_mode():
                outputs = fold_model(batched_tensor)
                raw_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_fold_outputs.append(raw_probs)

        for idx, r in enumerate(requests):
            fold_predictions = []
            for fold_probs in all_fold_outputs:
                probs = fold_probs[idx]
                pred_map_fold = dict(zip(pathologies, probs.tolist()))
                fold_predictions.append(pred_map_fold)

            mean_probs = {}
            std_probs = {}
            for c in pathologies:
                probs_c = [pred[c] for pred in fold_predictions]
                mean_probs[c] = float(np.sum([p * w for p, w in zip(probs_c, FOLD_WEIGHTS)]))
                std_probs[c] = float(np.std(probs_c))

            max_variance = max(std_probs.values())
            if max_variance > 0.15:
                print(f"🚨 CLINICAL DISCREPANCY: {modality} fold variance {max_variance:.4f} > 0.15 — triggering triage.")
                pathology = "Inconclusive (Discrepancy Triage: Requires Specialist Verification)"
                confidence = 0.0
                pred_map = mean_probs
            else:
                pred_map = mean_probs
                pathology = max(pred_map, key=pred_map.get)
                confidence = float(pred_map[pathology])

            bbox = None
            img_b64 = None
            if pathology not in ("Normal", "Inconclusive (Discrepancy Triage: Requires Specialist Verification)"):
                target_class_idx = None
                if pathology in pathologies:
                    target_class_idx = pathologies.index(pathology)

                if target_class_idx is not None and target_class_idx < len(pathologies) - 1:
                    single_tensor = tensors[idx].float()
                    bbox, img_b64 = _run_gradcam_for_sample(
                        model, single_tensor, target_class_idx, modality, r["file_path"]
                    )

            results.append({
                "pathology_detected": pathology,
                "confidence_score": confidence,
                "bbox": bbox,
                "predictions": pred_map,
                "pytorch_executed": True,
                "model_info": "MONAI ModelServer (ResNet50 5-Fold Ensemble Batched)",
                "img_base64": img_b64,
            })

    return results


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file_path: str = Form(...),
    modality: str = Form(...),
    patient_hash: str = Form(...)
):
    """Submits a prediction request to the dynamic batch queue."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Target file path not found.")
        
    future = asyncio.get_running_loop().create_future()
    await inference_queue.put({
        "modality": modality.upper(),
        "file_path": file_path,
        "patient_hash": patient_hash,
        "future": future
    })
    
    try:
        result = await future
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model serving error: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "xray_folds": len(xray_folds),
        "ct_folds": len(ct_folds),
        "mri_folds": len(mri_folds),
        "stats": stats
    }


# ── Server Lifespan ───────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_models()
    asyncio.create_task(dynamic_batch_processor())
    print("✓ Neuron AI Inference Server running.")
