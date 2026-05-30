// HistoryView.tsx — Scan History Logs

import React, { useEffect, useState, useCallback } from 'react';
import { Search, ChevronLeft, ChevronRight, RefreshCw, Hourglass } from 'lucide-react';
import { modelApi, ScanRecord, ScanHistoryResponse } from '../services/modelApi';

interface HistoryViewProps {
  onError: (msg: string) => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({ onError }) => {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  
  // Filters
  const [modality, setModality] = useState<string>('');
  const [pathology, setPathology] = useState<string>('');
  
  // Pagination
  const [page, setPage] = useState<number>(1);
  const [pageSize] = useState<number>(15);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [totalRecords, setTotalRecords] = useState<number>(0);

  const fetchHistory = useCallback(async () => {
    try {
      setLoading(true);
      const data: ScanHistoryResponse = await modelApi.getScanHistory(page, pageSize, modality, pathology);
      setScans(data.scans);
      setTotalPages(data.total_pages || 1);
      setTotalRecords(data.total || 0);
    } catch (err: any) {
      onError(err.message || 'Failed to fetch scan logs.');
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, modality, pathology, onError]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleFilterChange = () => {
    setPage(1);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-clinical-border/40 pb-3">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold tracking-tight text-white uppercase tracking-wider">Audit Scan History</h2>
          <p className="text-xs text-clinical-textMuted font-medium font-sans">Complete database logs of processed clinical scanner queries.</p>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2">
          <select
            value={modality}
            onChange={(e) => { setModality(e.target.value); handleFilterChange(); }}
            className="bg-clinical-card border border-clinical-border/50 rounded text-xs font-semibold px-3 py-2 text-white outline-none focus:border-clinical-highlight"
          >
            <option value="">All Modalities</option>
            <option value="XRAY">Chest X-Ray</option>
            <option value="CT">CT Scan</option>
            <option value="MRI">MRI Scan</option>
          </select>
          <div className="relative">
            <input
              type="text"
              placeholder="Search pathology..."
              value={pathology}
              onChange={(e) => { setPathology(e.target.value); handleFilterChange(); }}
              className="bg-clinical-card border border-clinical-border/50 rounded text-xs font-semibold pl-8 pr-3 py-2 text-white outline-none focus:border-clinical-highlight w-44"
            />
            <Search className="w-3.5 h-3.5 text-clinical-textMuted absolute left-2.5 top-2.5" />
          </div>
          <button 
            onClick={fetchHistory}
            className="p-2 bg-clinical-border/80 hover:bg-clinical-border border border-clinical-border/50 rounded text-white transition-colors"
            title="Refresh database logs"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* History Card table */}
      <div className="bg-clinical-card border border-clinical-border/40 rounded-lg p-5 shadow-sm flex flex-col min-h-[400px] justify-between">
        <div className="overflow-x-auto flex-1">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-clinical-border text-clinical-textMuted font-bold uppercase tracking-wider text-[10px]">
                <th className="pb-3">Logged Date</th>
                <th className="pb-3">Patient SHA-256 Hash</th>
                <th className="pb-3">Scan Modality</th>
                <th className="pb-3">Pathology Finding</th>
                <th className="pb-3">Confidence</th>
                <th className="pb-3 text-right">Inference Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-clinical-border/20 font-medium">
              {loading ? (
                <tr>
                  <td colSpan={6} className="py-20 text-center">
                    <div className="w-6 h-6 border-2 border-clinical-accent border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                    <span className="text-clinical-textMuted text-[10px] font-bold uppercase">Accessing records...</span>
                  </td>
                </tr>
              ) : scans.length > 0 ? (
                scans.map((scan) => {
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
                      <td className="py-3">
                        <div className="flex items-center gap-1.5 font-semibold">
                          <span className={isNormal ? 'text-clinical-success' : isInconclusive ? 'text-clinical-warning' : 'text-clinical-danger'}>
                            {scan.pathology_detected}
                          </span>
                          {scan.pytorch_executed === 'true' && (
                            <span className="px-1.5 py-0.5 text-[8px] bg-clinical-highlight/10 border border-clinical-highlight/20 text-clinical-highlight rounded font-extrabold uppercase">
                              AI
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 font-mono text-clinical-textMuted">
                        {(scan.confidence_score * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 text-right font-mono text-white font-bold flex items-center justify-end gap-1">
                        <Hourglass className="w-3 h-3 opacity-40 text-clinical-textMuted" />
                        {scan.inference_latency ? `${scan.inference_latency.toFixed(1)} ms` : '—'}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-clinical-textMuted">
                    No matching scan audit records found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination footer */}
        {totalPages > 1 && (
          <div className="border-t border-clinical-border/20 pt-4 mt-4 flex items-center justify-between text-xs text-clinical-textMuted">
            <span>
              Showing Page **{page}** of **{totalPages}** ({totalRecords} records logged)
            </span>
            <div className="flex items-center gap-1">
              <button
                disabled={page <= 1 || loading}
                onClick={() => setPage(p => p - 1)}
                className="p-1.5 border border-clinical-border/50 hover:bg-clinical-border/30 rounded text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                disabled={page >= totalPages || loading}
                onClick={() => setPage(p => p + 1)}
                className="p-1.5 border border-clinical-border/50 hover:bg-clinical-border/30 rounded text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
