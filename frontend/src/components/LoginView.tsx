// LoginView.tsx — Secure Radiologist Login Portal

import React, { useState } from 'react';
import { ShieldCheck, Eye, EyeOff, Loader2 } from 'lucide-react';
import { modelApi, UserSession } from '../services/modelApi';

interface LoginViewProps {
  onLoginSuccess: (session: UserSession) => void;
  onError: (msg: string) => void;
}

export const LoginView: React.FC<LoginViewProps> = ({ onLoginSuccess, onError }) => {
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setErrorMsg('Username and password fields are required.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);

    try {
      const session = await modelApi.login(username, password);
      onLoginSuccess(session);
    } catch (err: any) {
      const msg = err.message || 'Access denied. Invalid credentials.';
      setErrorMsg(msg);
      onError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-clinical-bg text-clinical-text flex items-center justify-center p-4 font-sans select-none">
      <div className="w-full max-w-[420px] bg-clinical-card border border-clinical-border rounded-2xl p-8 shadow-[0_16px_40px_rgba(7,10,19,0.5)] relative overflow-hidden">
        
        {/* Glow effect */}
        <div className="absolute top-0 right-0 w-32 h-32 rounded-full bg-clinical-accent/10 blur-2xl"></div>
        
        {/* Top security tag */}
        <div className="absolute top-4 right-5 text-[8.5px] font-extrabold uppercase tracking-widest text-clinical-textMuted flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-clinical-success animate-ping"></span>
          Secure Node
        </div>

        {/* Logo and Titles */}
        <div className="flex items-center gap-3.5 mb-8">
          <div className="w-11 h-11 rounded-xl bg-clinical-accent flex items-center justify-center shadow-[0_6px_14px_rgba(99,102,241,0.35)]">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tight text-white">NEURON AI</h1>
            <p className="text-[9.5px] text-clinical-textMuted font-bold uppercase tracking-wider mt-0.5">
              Clinical Diagnostic Station
            </p>
          </div>
        </div>

        {/* Heading */}
        <div className="mb-6 space-y-1">
          <h2 className="text-lg font-bold text-white">Radiologist Sign In</h2>
          <p className="text-xs text-clinical-textMuted">Access localized hospital patient datasets and AI analysis</p>
        </div>

        {/* Login form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-clinical-textMuted uppercase tracking-wider block">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-clinical-bg border border-clinical-border rounded-lg text-xs font-semibold px-3.5 py-3 text-white outline-none focus:border-clinical-highlight placeholder-clinical-textMuted/40"
              placeholder="Enter clinic username"
              autoComplete="username"
              required
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-clinical-textMuted uppercase tracking-wider block">Password</label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-clinical-bg border border-clinical-border rounded-lg text-xs font-semibold px-3.5 py-3 pr-10 text-white outline-none focus:border-clinical-highlight placeholder-clinical-textMuted/40"
                placeholder="Enter workstation password"
                autoComplete="current-password"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(p => !p)}
                className="absolute right-3.5 top-3.5 text-clinical-textMuted hover:text-white transition-colors"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Validation error display */}
          {errorMsg && (
            <div className="p-3 bg-clinical-danger/10 border border-clinical-danger/25 text-clinical-danger text-xs rounded-lg animate-pulse">
              {errorMsg}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-clinical-accent hover:bg-clinical-accent/90 text-xs font-bold text-white rounded-lg transition-all shadow-[0_6px_16px_rgba(99,102,241,0.25)] flex items-center justify-center gap-2 uppercase tracking-wider"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Authorizing...
              </>
            ) : (
              'Access Workstation'
            )}
          </button>
        </form>

        {/* DPDP Disclaimer bottom footer */}
        <div className="mt-8 pt-5 border-t border-clinical-border/40 text-[9.5px] text-clinical-textMuted flex items-center gap-2 font-semibold">
          <ShieldCheck className="w-4 h-4 text-clinical-success flex-shrink-0" />
          <span>DPDP Act Compliant · Localized Data Nodes · JWT Sessions</span>
        </div>

      </div>
    </div>
  );
};
