// ModelsView.tsx — Deep Learning Models Deep-Dive

import React from 'react';
import { Cpu, ShieldCheck, Flame, GitMerge, Settings, HelpCircle } from 'lucide-react';

export const ModelsView: React.FC = () => {
  const models = [
    {
      title: 'MONAI 3D U-Net',
      category: 'Volumetric Segmentation',
      desc: 'Standardized 3D encoder-decoder network engineered for voxel-level medical classification. Processes isotropic dimensions natively to trace bounding outlines and tissue parameters.',
      backbone: '3D ResNet/Custom Conv Block',
      params: '1.19M Parameters (Edge Config)',
      weights: '10.58 ms Load Time',
      provenance: 'Medical Segmentation Decathlon Winner, MONAI Core Hub',
      optimizations: [
        'Depth reductions from 5 down to 3 layers to prevent VRAM ceiling breaches.',
        'Deep supervision hooks stripped during deployment to reduce operational latency.',
        'Isotropic spatial pooling ($96^3$) reduces multi-slice compute overhead by 80%.'
      ]
    },
    {
      title: 'MONAI SegResNet',
      category: 'Advanced MRI Segmentation',
      desc: 'SOTA convolutional neural network incorporating encoder residual blocks and variational autoencoder (VAE) branches to reconstruct brain tumor boundaries.',
      backbone: 'ResNet block + VAE regularization',
      params: '24.7M Parameters',
      weights: 'Pretrained on BraTS 2023',
      provenance: 'Kaggle RSNA Brain Tumor Challenge / Harvard MGB Publications',
      optimizations: [
        'Channel compression (base filter width lowered from 32 to 16).',
        'Stripped variational autoencoder (VAE) regularization head at inference time.',
        'Compiled via TensorRT to run sub-100ms volumetric segmentations.'
      ]
    },
    {
      title: 'DenseNet-121 (Torchvision / XRV)',
      category: 'Chest Radiography Diagnostics',
      desc: 'Densely connected convolutional network that reuses features from previous layers. Pretrained on NIH, CheXpert, and MIMIC-CXR for multi-label pathology classification.',
      backbone: 'DenseNet-121 (Dense Blocks + Transition Layers)',
      params: '6.96M Parameters',
      weights: 'densenet121-res224-all (Pre-cached)',
      provenance: 'CheXNet (Stanford ML Group) / TorchXRayVision Library',
      optimizations: [
        'Pruned final fully-connected layers to target clinical pathology classes.',
        'Image intensity scaling mapped to [-1024, 1024] to avoid saturation in legacy CR X-rays.',
        'ONNX Runtime FP16 compilation reduces inference latency to <25ms on CPU.'
      ]
    },
    {
      title: 'ResNet-50 Backbone',
      category: 'Multi-Class CT/MRI Classifier',
      desc: 'Deep residual network serving as a generalist classifier backbone. Finetuned with a medical classification head to classify volumetric slices into pathological states.',
      backbone: 'ResNet-50 (Bottleneck Layers)',
      params: '23.5M Parameters',
      weights: 'resnet50_clinical.pt (On-Premise Cache)',
      provenance: 'ImageNet Pretrained / Fine-tuned on RSNA Stroke & Appendicitis Challenges',
      optimizations: [
        'Unified multi-slice 2.5D feature blending. Combines three adjacent slices into 3-channel input.',
        'Post-training INT8 quantization reduces memory footprint from 98MB to 24MB.',
        'Fused Batch Normalization into adjacent Conv layers during serialization.'
      ]
    }
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <h2 className="text-2xl font-extrabold tracking-tight text-white">Deep Learning Models Deep-Dive</h2>
        <p className="text-xs text-clinical-textMuted">Technical blueprint and optimization logs for active clinical models.</p>
      </div>

      {/* Model Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {models.map((m, idx) => (
          <div key={idx} className="bg-clinical-card border border-clinical-border rounded-xl p-6 flex flex-col justify-between shadow-lg relative overflow-hidden group hover:border-clinical-highlight/50 transition-all">
            {/* Background design glow */}
            <div className="absolute -top-16 -right-16 w-32 h-32 rounded-full bg-clinical-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl"></div>
            
            <div className="space-y-4">
              {/* Header */}
              <div className="flex justify-between items-start border-b border-clinical-border/50 pb-3">
                <div>
                  <span className="text-[9px] px-2 py-0.5 rounded-full bg-clinical-highlight/10 text-clinical-highlight border border-clinical-highlight/20 font-bold uppercase tracking-wider">
                    {m.category}
                  </span>
                  <h3 className="text-base font-extrabold text-white mt-1.5">{m.title}</h3>
                </div>
                <Cpu className="w-5 h-5 text-clinical-textMuted group-hover:text-clinical-highlight transition-colors" />
              </div>

              {/* Description */}
              <p className="text-xs text-clinical-textMuted leading-relaxed">{m.desc}</p>

              {/* Specifications */}
              <div className="grid grid-cols-2 gap-3 text-[10px] bg-clinical-bg/40 p-3 rounded-lg border border-clinical-border/40 font-mono">
                <div>
                  <span className="text-clinical-textMuted block font-sans">Core Backbone</span>
                  <span className="text-white font-bold">{m.backbone}</span>
                </div>
                <div>
                  <span className="text-clinical-textMuted block font-sans">VRAM/Capacity</span>
                  <span className="text-clinical-highlight font-bold">{m.params}</span>
                </div>
                <div className="col-span-2 border-t border-clinical-border/30 pt-2 mt-1">
                  <span className="text-clinical-textMuted block font-sans">SOTA Provenance</span>
                  <span className="text-white font-bold font-sans">{m.provenance}</span>
                </div>
              </div>

              {/* Optimization logs */}
              <div className="space-y-1.5">
                <span className="text-[9px] font-bold text-clinical-highlight uppercase tracking-wider block">Edge Optimization Log</span>
                <ul className="text-[10px] space-y-1.5 text-clinical-textMuted leading-relaxed">
                  {m.optimizations.map((opt, i) => (
                    <li key={i} className="flex gap-2 items-start">
                      <ShieldCheck className="w-3.5 h-3.5 text-clinical-success mt-0.5 flex-shrink-0" />
                      <span>{opt}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Edge Compilation Info Callout */}
      <div className="bg-clinical-card border border-clinical-border rounded-xl p-5 shadow-lg flex flex-col md:flex-row gap-5 items-start md:items-center">
        <div className="p-3 bg-clinical-accent/10 border border-clinical-accent/20 rounded-xl text-clinical-accent flex-shrink-0">
          <Flame className="w-6 h-6 text-clinical-accent animate-pulse" />
        </div>
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-white">Edge Compiled Quantization Pipeline</h4>
          <p className="text-xs text-clinical-textMuted mt-1 leading-relaxed">
            Local hospital deployment relies on ONNX and TensorRT runtimes. By compiling models to **FP16** and **INT8** calibration matrices, we achieve deterministic latencies, preventing thread blocking during concurrent diagnostic searches on low-powered local clinical servers.
          </p>
        </div>
      </div>
    </div>
  );
};
