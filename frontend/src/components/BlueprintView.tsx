// BlueprintView.tsx — Interactive Architecture Blueprint

import React, { useState } from 'react';
import { Folder, FileCode, CheckCircle, Cpu, ShieldAlert, Code } from 'lucide-react';

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  desc: string;
  responsibilities: string[];
  exports: string[];
  children?: FileNode[];
}

export const BlueprintView: React.FC = () => {
  const codebase: FileNode[] = [
    {
      name: 'frontend',
      path: '/frontend',
      type: 'folder',
      desc: 'Workstation client code built with React, TypeScript, and Tailwind CSS.',
      responsibilities: ['UI layouts', 'Client-side binary file parsing', 'Charts and telemetry widgets'],
      exports: [],
      children: [
        {
          name: 'src/components/layout/ClinicalLayout.tsx',
          path: '/frontend/src/components/layout/ClinicalLayout.tsx',
          type: 'file',
          desc: 'App Layout Engine. Responsible for viewport media breakpoints, sticky navigation sidebar, page drawer controllers, and styling containment.',
          responsibilities: [
            'Handles grid-12 layouts and sticky headers.',
            'Contains navigation menus and viewport responsiveness tags.',
            'Displays logged-in radiologist avatar and role.'
          ],
          exports: ['ClinicalLayout (React Component)'],
        },
        {
          name: 'src/services/modelApi.ts',
          path: '/frontend/src/services/modelApi.ts',
          type: 'file',
          desc: 'Model Service API Layer. Manages asynchronous endpoint dispatches, maps raw file payloads to Form FormData buffers, and parses API telemetry contracts.',
          responsibilities: [
            'Manages localStorage authorization JWT tokens.',
            'Sends multi-part scan files to /api/upload-scan.',
            'Queries dashboard metrics and historical clinics database logs.'
          ],
          exports: ['modelApi (object)', 'ScanResult (interface)', 'DashboardMetrics (interface)', 'ClinicSettings (interface)'],
        },
        {
          name: 'src/hooks/useMedicalParser.ts',
          path: '/frontend/src/hooks/useMedicalParser.ts',
          type: 'file',
          desc: 'Multi-Modality Parsing Hook. Asynchronously reads local file binary streams to isolate header metadata parameters from raw pixel matrices.',
          responsibilities: [
            'Scans offset byte 128 for DICOM magic tag "DICM".',
            'Scans offset byte 344 for NIfTI volume magic tags.',
            'Extracts manufacturer, dimensions, voxel spacing, and modality parameters.'
          ],
          exports: ['useMedicalParser (React Hook)', 'ParsedScan (interface)'],
        },
        {
          name: 'src/App.tsx',
          path: '/frontend/src/App.tsx',
          type: 'file',
          desc: 'Main Client Orchestrator. Directs state boundaries, controls page navigation transitions, and handles toast alerts.',
          responsibilities: [
            'Holds active page navigation state.',
            'Renders layout wrapper with custom views.',
            'Implements secure routing redirection if JWT tokens expire.'
          ],
          exports: ['App (React Component)'],
        }
      ]
    },
    {
      name: 'app',
      path: '/app',
      type: 'folder',
      desc: 'FastAPI Backend application handling file pre-processing, security compliance, database storage, and model execution.',
      responsibilities: ['FastAPI server routers', 'SQLite clinical logging', 'Pydicom & Nibabel image processing', 'PyTorch runtime inference'],
      exports: [],
      children: [
        {
          name: 'main.py',
          path: '/app/main.py',
          type: 'file',
          desc: 'FastAPI Server Entry Point. Registers route endpoints, configures CORS settings, mounts static static/ folders, and enforces SlowAPI rate-limiting protections.',
          responsibilities: [
            'Serves index.html templates.',
            'Implements /api/upload-scan binary endpoint.',
            'Implements User login and change-password endpoints.'
          ],
          exports: ['app (FastAPI instance)'],
        },
        {
          name: 'inference.py',
          path: '/app/inference.py',
          type: 'file',
          desc: 'Medical AI Inference Engine. Loads PyTorch weights, constructs model backbones (DenseNet121, UNet3d, ResNet50), executes inference, and generates bounding box highlights.',
          responsibilities: [
            'Loads TorchXRayVision DenseNet weights.',
            'Initializes MONAI 3D UNet structure.',
            'Maps predicted scores to modality-specific class structures.'
          ],
          exports: ['run_inference (function)', 'device (torch.device)'],
        },
        {
          name: 'database.py',
          path: '/app/database.py',
          type: 'file',
          desc: 'SQLite Database Controller. Implements SQL Alchemy schemas to track scan metrics, user accounts, and clinic-wide settings.',
          responsibilities: [
            'Logs patient hashes, modalities, and findings.',
            'Aggregates positive rates and modality distributions.',
            'Stores clinic variables like rates and station IDs.'
          ],
          exports: ['Base', 'ScanLog', 'ClinicSettings', 'log_scan', 'get_dashboard_metrics'],
        }
      ]
    }
  ];

  const [selectedFile, setSelectedFile] = useState<FileNode>(codebase[0].children![0]);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <h2 className="text-2xl font-extrabold tracking-tight text-white">Architecture & Source Base Map</h2>
        <p className="text-xs text-clinical-textMuted">Interactive blueprint mapping source modules to clinical deployment parameters.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        
        {/* Left Side: Folder & File Explorer */}
        <div className="lg:col-span-4 bg-clinical-card border border-clinical-border rounded-xl p-5 shadow-lg flex flex-col">
          <h3 className="text-xs font-bold uppercase tracking-wider text-white border-b border-clinical-border pb-3 mb-4">
            Directory Explorer
          </h3>

          <div className="flex-1 space-y-4 overflow-y-auto max-h-[450px]">
            {codebase.map((folder) => (
              <div key={folder.name} className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs font-bold text-white px-2">
                  <Folder className="w-4 h-4 text-clinical-highlight" />
                  {folder.name}/
                </div>
                <div className="pl-6 space-y-1">
                  {folder.children?.map((fileNode) => {
                    const isSelected = selectedFile.path === fileNode.path;
                    return (
                      <button
                        key={fileNode.path}
                        onClick={() => setSelectedFile(fileNode)}
                        className={`
                          w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-[11px] font-semibold tracking-wide transition-all
                          ${isSelected 
                            ? 'bg-clinical-accent/15 text-clinical-highlight border border-clinical-highlight/30 shadow-[0_2px_8px_rgba(6,182,212,0.05)]' 
                            : 'text-clinical-textMuted hover:bg-clinical-border/40 hover:text-clinical-text border border-transparent'}
                        `}
                      >
                        <FileCode className={`w-3.5 h-3.5 ${isSelected ? 'text-clinical-highlight' : 'text-clinical-textMuted'}`} />
                        {fileNode.name.replace('src/components/layout/', '').replace('src/hooks/', '').replace('src/services/', '').replace('src/', '')}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Side: Interactive Documentation Panel */}
        <div className="lg:col-span-8 bg-clinical-card border border-clinical-border rounded-xl p-6 shadow-lg flex flex-col justify-between">
          <div className="space-y-5">
            {/* Header file info */}
            <div className="border-b border-clinical-border/50 pb-4">
              <div className="text-[10px] text-clinical-textMuted font-mono uppercase tracking-widest">{selectedFile.path}</div>
              <h3 className="text-lg font-black text-white mt-1 flex items-center gap-2">
                <Code className="w-5 h-5 text-clinical-highlight" />
                {selectedFile.name.substring(selectedFile.name.lastIndexOf('/') + 1)}
              </h3>
            </div>

            {/* Description */}
            <div className="space-y-1.5">
              <span className="text-[9px] font-bold text-clinical-highlight uppercase tracking-wider block">Description</span>
              <p className="text-xs text-clinical-textMuted leading-relaxed">{selectedFile.desc}</p>
            </div>

            {/* Responsibilities list */}
            <div className="space-y-2 border-t border-clinical-border/40 pt-4">
              <span className="text-[9px] font-bold text-clinical-highlight uppercase tracking-wider block">Key System Responsibilities</span>
              <ul className="text-xs space-y-2 text-clinical-textMuted">
                {selectedFile.responsibilities.map((resp, i) => (
                  <li key={i} className="flex gap-2 items-start">
                    <CheckCircle className="w-4 h-4 text-clinical-success mt-0.5 flex-shrink-0" />
                    <span>{resp}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Exports */}
            {selectedFile.exports.length > 0 && (
              <div className="space-y-2 border-t border-clinical-border/40 pt-4">
                <span className="text-[9px] font-bold text-clinical-highlight uppercase tracking-wider block">Key Exports / Structures</span>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {selectedFile.exports.map((exp, i) => (
                    <span key={i} className="px-2 py-0.5 rounded bg-clinical-border/60 border border-clinical-border/80 text-[9px] font-mono text-white">
                      {exp}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="mt-6 pt-4 border-t border-clinical-border/40 text-[9.5px] text-clinical-textMuted leading-relaxed flex gap-2">
            <ShieldAlert className="w-5 h-5 text-clinical-highlight flex-shrink-0 mt-0.5" />
            <span>
              This file-by-file mapping is synchronized with the source-control repository. Modify this architecture map strictly under workstation coordinator supervision to maintain documentation integrity.
            </span>
          </div>
        </div>

      </div>
    </div>
  );
};
