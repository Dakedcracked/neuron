// App.tsx — Clinical Workstation Router

import React, { useState, useEffect } from 'react';
import { ClinicalLayout } from './components/layout/ClinicalLayout';
import { DashboardView } from './components/DashboardView';
import { InferenceView } from './components/InferenceView';
import { HistoryView } from './components/HistoryView';
import { ModelsView } from './components/ModelsView';
import { DatasetsView } from './components/DatasetsView';
import { BlueprintView } from './components/BlueprintView';
import { SettingsView } from './components/SettingsView';
import { LoginView } from './components/LoginView';
import { modelApi, UserSession } from './services/modelApi';

interface Toast {
  id: number;
  text: string;
  type: 'success' | 'error' | 'info';
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(modelApi.isAuthenticated());
  const [activePage, setActivePage] = useState<string>('dashboard');
  const [username, setUsername] = useState<string>(localStorage.getItem('neuron_username') || '');
  const [role, setRole] = useState<string>(localStorage.getItem('neuron_role') || '');
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Monitor paths to support hard links like /about and /models if served statically
  useEffect(() => {
    if (!isAuthenticated) return;
    const path = window.location.pathname.toLowerCase();
    if (path === '/about' || path === '/datasets') {
      setActivePage('datasets');
    } else if (path === '/models') {
      setActivePage('models');
    } else {
      setActivePage('dashboard');
    }
  }, [isAuthenticated]);

  // Toast notifier helper
  const showToast = (text: string, type: 'success' | 'error' | 'info' = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, text, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  };

  const handleLoginSuccess = (session: UserSession) => {
    setIsAuthenticated(true);
    setUsername(session.username);
    setRole(session.role);
    showToast(`Access granted. Welcome, ${session.username}!`, 'success');
  };

  const handleLogout = () => {
    modelApi.clearSession();
    setIsAuthenticated(false);
    setUsername('');
    setRole('');
    window.location.replace('/login');
  };

  // Render sub-views dynamically based on state router
  const renderContent = () => {
    switch (activePage) {
      case 'dashboard':
        return (
          <DashboardView 
            onNavigate={(page) => setActivePage(page)} 
            onError={(msg) => showToast(msg, 'error')} 
          />
        );
      case 'scan':
        return (
          <InferenceView 
            onSuccess={(msg) => showToast(msg, 'success')} 
            onError={(msg) => showToast(msg, 'error')} 
          />
        );
      case 'history':
        return (
          <HistoryView 
            onError={(msg) => showToast(msg, 'error')} 
          />
        );
      case 'models':
        return <ModelsView />;
      case 'datasets':
        return <DatasetsView />;
      case 'settings':
        return (
          <SettingsView 
            onSuccess={(msg) => showToast(msg, 'success')} 
            onError={(msg) => showToast(msg, 'error')} 
          />
        );
      default:
        return (
          <div className="text-center py-20 text-clinical-textMuted">
            Section '{activePage}' is currently offline.
          </div>
        );
    }
  };

  // If not authenticated, force the secure login card view
  if (!isAuthenticated) {
    return (
      <LoginView 
        onLoginSuccess={handleLoginSuccess} 
        onError={(msg) => showToast(msg, 'error')} 
      />
    );
  }

  return (
    <ClinicalLayout
      activePage={activePage}
      onNavigate={setActivePage}
      username={username}
      role={role}
      onLogout={handleLogout}
    >
      {/* Dynamic View Injection */}
      <div className="pb-16">{renderContent()}</div>

      {/* Floating Blueprint Access Icon */}
      <div className="fixed bottom-6 right-6 z-50">
        <button
          onClick={() => setActivePage('blueprint')}
          className={`
            flex items-center gap-2 px-4 py-3 rounded-full font-bold text-xs shadow-2xl transition-all border uppercase tracking-wider
            ${activePage === 'blueprint' 
              ? 'bg-clinical-highlight text-black border-clinical-highlight scale-105' 
              : 'bg-clinical-card text-clinical-highlight border-clinical-border hover:border-clinical-highlight hover:scale-105'}
          `}
        >
          <CodeIcon className="w-4 h-4" />
          {activePage === 'blueprint' ? 'Close Map' : 'Source Map'}
        </button>
      </div>

      {/* Blueprint View Injection override */}
      {activePage === 'blueprint' && (
        <div className="fixed inset-0 z-40 bg-clinical-bg/95 backdrop-blur-md p-6 overflow-y-auto">
          <div className="max-w-[1440px] mx-auto pt-16 relative">
            <button
              onClick={() => setActivePage('dashboard')}
              className="absolute top-4 right-0 px-4 py-2 border border-clinical-border rounded-lg text-xs font-semibold text-clinical-textMuted hover:text-white"
            >
              Exit Source Blueprint
            </button>
            <BlueprintView />
          </div>
        </div>
      )}

      {/* Unified absolute toast container */}
      <div className="fixed top-6 right-6 z-[9999] flex flex-col gap-2.5 max-w-sm w-full pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`
              p-4 rounded-xl shadow-2xl border text-xs font-semibold flex items-center gap-3 pointer-events-auto transform translate-x-0 transition-transform duration-300
              ${t.type === 'success' ? 'bg-[#0f2420] border-clinical-success/40 text-clinical-success' : ''}
              ${t.type === 'error' ? 'bg-[#2b161c] border-clinical-danger/40 text-clinical-danger' : ''}
              ${t.type === 'info' ? 'bg-clinical-card border-clinical-border text-clinical-highlight' : ''}
            `}
          >
            <span className="text-sm">
              {t.type === 'success' && '✓'}
              {t.type === 'error' && '✗'}
              {t.type === 'info' && 'ℹ'}
            </span>
            <span>{t.text}</span>
          </div>
        ))}
      </div>
    </ClinicalLayout>
  );
}

// Inline svg fallback for code blueprint button
const CodeIcon = ({ className }: { className?: string }) => (
  <svg 
    className={className} 
    fill="none" 
    viewBox="0 0 24 24" 
    stroke="currentColor" 
    strokeWidth="2.2"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
  </svg>
);

export default App;
