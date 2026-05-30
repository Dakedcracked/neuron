// DashboardView.tsx — Clinical Telemetry Overview

import React, { useEffect, useState } from 'react';
import { Doughnut, Bar } from 'react-chartjs-2';
import { 
  Chart as ChartJS, 
  ArcElement, 
  Tooltip, 
  Legend, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title 
} from 'chart.js';
import { Activity, ShieldAlert, Zap, Hourglass } from 'lucide-react';
import { modelApi, DashboardMetrics } from '../services/modelApi';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

interface DashboardViewProps {
  onNavigate: (page: string) => void;
  onError: (msg: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate, onError }) => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const data = await modelApi.getDashboardMetrics();
        setMetrics(data);
      } catch (err: any) {
        onError(err.message || 'Failed to fetch clinical telemetry.');
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, [onError]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-3">
        <div className="w-8 h-8 border-2 border-clinical-accent border-t-transparent rounded-full animate-spin"></div>
        <div className="text-xs font-bold text-clinical-textMuted uppercase tracking-wider">Querying telemetry metrics...</div>
      </div>
    );
  }

  const m = metrics || {
    total_scans: 0,
    positive_rate: 0,
    modality_counts: { XRAY: 0, CT: 0, MRI: 0 },
    modality_latencies: { XRAY: 0, CT: 0, MRI: 0 },
    pathology_counts: {},
    recent_scans: []
  };

  const avg3DLatency = Math.round(
    ((m.modality_latencies.CT + m.modality_latencies.MRI) / 2) || 0
  );

  const doughnutData = {
    labels: ['Chest X-Ray', 'CT Scan', 'MRI Scan'],
    datasets: [{
      data: [m.modality_counts.XRAY, m.modality_counts.CT, m.modality_counts.MRI],
      backgroundColor: ['#38bdf8', '#6366f1', '#fbbf24'],
      borderColor: '#0f1424',
      borderWidth: 2,
    }],
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } }
      },
      title: {
        display: true,
        text: 'MODALITIES PROCESSED',
        color: '#e2e8f0',
        font: { family: 'Outfit', size: 11, weight: 'bold' as const }
      }
    }
  };

  const pathLabels = Object.keys(m.pathology_counts);
  const pathData = Object.values(m.pathology_counts);
  const barData = {
    labels: pathLabels.length ? pathLabels : ['No Scans'],
    datasets: [{
      data: pathData.length ? pathData : [0],
      backgroundColor: '#4f46e5',
      borderRadius: 2,
      barThickness: 20,
    }],
  };

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 9 } }
      },
      y: {
        grid: { color: '#1f2945' },
        ticks: { precision: 0, color: '#94a3b8', font: { family: 'Outfit', size: 9 } }
      }
    },
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'PATHOLOGY ACCURACY MATRIX',
        color: '#e2e8f0',
        font: { family: 'Outfit', size: 11, weight: 'bold' as const }
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <h2 className="text-xl font-bold tracking-tight text-white uppercase tracking-wider">Clinical Telemetry</h2>
        <p className="text-xs text-clinical-textMuted font-medium">Model verification performance, execution latencies, and accuracy checks on Indian patient datasets.</p>
      </div>

      {/* Telemetry Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Evaluation Scans', val: m.total_scans, sub: 'Log size in local database', icon: Zap, border: 'border-clinical-border/40 text-clinical-highlight' },
          { label: 'Abnormality Findings', val: `${m.positive_rate}%`, sub: 'Diagnostic discovery rate', icon: ShieldAlert, border: 'border-clinical-border/40 text-clinical-danger' },
          { label: 'Avg 2D X-Ray Latency', val: `${m.modality_latencies.XRAY ? m.modality_latencies.XRAY.toFixed(1) : '0.0'} ms`, sub: 'Classification benchmarks', icon: Activity, border: 'border-clinical-border/40 text-clinical-success' },
          { label: 'Avg 3D Volume Latency', val: `${avg3DLatency ? avg3DLatency : '0'} ms`, sub: 'MONAI 3D pipeline benchmarks', icon: Hourglass, border: 'border-clinical-border/40 text-clinical-warning' },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} className={`bg-clinical-card border ${stat.border} rounded-lg p-5 shadow-sm hover:border-clinical-highlight/30 transition-colors`}>
              <div className="flex justify-between items-start">
                <div className="space-y-1">
                  <div className="text-[10px] text-clinical-textMuted font-bold uppercase tracking-wider">{stat.label}</div>
                  <div className="text-xl font-black text-white">{stat.val}</div>
                  <div className="text-[9px] text-clinical-textMuted font-semibold">{stat.sub}</div>
                </div>
                <Icon className="w-4 h-4 opacity-40 text-clinical-textMuted" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-clinical-card border border-clinical-border/50 rounded-lg p-5 flex flex-col h-[280px]">
          <div className="flex-1 min-h-0">
            <Doughnut data={doughnutData} options={doughnutOptions} />
          </div>
        </div>

        <div className="lg:col-span-7 bg-clinical-card border border-clinical-border/50 rounded-lg p-5 flex flex-col h-[280px]">
          <div className="flex-1 min-h-0">
            <Bar data={barData} options={barOptions} />
          </div>
        </div>
      </div>

      {/* Recent Activity Table */}
      <div className="bg-clinical-card border border-clinical-border/50 rounded-lg p-5">
        <div className="flex items-center justify-between mb-4 border-b border-clinical-border/40 pb-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-white">Recent Telemetry Queries</h3>
          <button 
            onClick={() => onNavigate('history')}
            className="text-[10px] text-clinical-accent font-bold hover:underline uppercase tracking-wider"
          >
            Audit Log Trail →
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-clinical-border text-clinical-textMuted font-bold uppercase tracking-wider text-[10px]">
                <th className="pb-3">Logged Date</th>
                <th className="pb-3">Patient Hash</th>
                <th className="pb-3">Scanner Modality</th>
                <th className="pb-3">Finding Diagnosis</th>
                <th className="pb-3">Confidence</th>
                <th className="pb-3 text-right">Inference Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-clinical-border/20 font-medium">
              {m.recent_scans.length > 0 ? (
                m.recent_scans.map((scan) => {
                  const isNormal = scan.pathology_detected === 'Normal';
                  const isInconclusive = scan.pathology_detected === 'Inconclusive';
                  return (
                    <tr key={scan.id} className="hover:bg-clinical-border/10 transition-colors">
                      <td className="py-3 text-clinical-textMuted font-mono text-[10px]">{scan.timestamp}</td>
                      <td className="py-3 text-clinical-highlight font-mono font-bold tracking-tight text-[10px]">
                        {scan.patient_hash}
                      </td>
                      <td className="py-3">
                        <span className={`
                          px-2 py-0.5 rounded text-[10px] font-bold border
                          ${scan.scan_type === 'XRAY' ? 'bg-clinical-highlight/10 text-clinical-highlight border-clinical-highlight/20' : ''}
                          ${scan.scan_type === 'CT' ? 'bg-clinical-accent/10 text-clinical-accent border-clinical-accent/20' : ''}
                          ${scan.scan_type === 'MRI' ? 'bg-clinical-warning/10 text-clinical-warning border-clinical-warning/20' : ''}
                        `}>
                          {scan.scan_type}
                        </span>
                      </td>
                      <td className={`py-3 font-semibold ${isNormal ? 'text-clinical-success' : isInconclusive ? 'text-clinical-warning' : 'text-clinical-danger'}`}>
                        {scan.pathology_detected}
                      </td>
                      <td className="py-3 font-mono text-clinical-textMuted">
                        {(scan.confidence_score * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 text-right font-mono text-white font-bold">
                        {scan.inference_latency ? `${scan.inference_latency.toFixed(1)} ms` : '—'}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-clinical-textMuted">
                    No clinical model queries have been recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
