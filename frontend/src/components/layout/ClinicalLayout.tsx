// ClinicalLayout.tsx — App Layout Engine

import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Upload, 
  History, 
  Settings as SettingsIcon, 
  Info, 
  Cpu, 
  LogOut, 
  Menu, 
  X, 
  User as UserIcon, 
  ShieldCheck 
} from 'lucide-react';
import { modelApi } from '../../services/modelApi';

interface ClinicalLayoutProps {
  children: React.ReactNode;
  activePage: string;
  onNavigate: (page: string) => void;
  username: string;
  role: string;
  onLogout: () => void;
}

export const ClinicalLayout: React.FC<ClinicalLayoutProps> = ({
  children,
  activePage,
  onNavigate,
  username,
  role,
  onLogout,
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Close sidebar on navigation in mobile viewports
  const handleNavigate = (page: string) => {
    onNavigate(page);
    setSidebarOpen(false);
  };

  const navItems = [
    { id: 'dashboard', name: 'Clinical Telemetry', icon: Activity, section: 'Workspace' },
    { id: 'scan', name: 'Live Diagnostic Ingestion', icon: Upload, section: 'Workspace' },
    { id: 'history', name: 'Audit Scan History', icon: History, section: 'Workspace' },
    { id: 'models', name: 'Deep Learning Models', icon: Cpu, section: 'Model Analytics' },
    { id: 'datasets', name: 'Datasets & Localization', icon: Info, section: 'Model Analytics' },
    { id: 'settings', name: 'Workstation Settings', icon: SettingsIcon, section: 'Configuration' },
  ];

  // Group items by sections
  const sections = ['Workspace', 'Model Analytics', 'Configuration'];

  return (
    <div className="min-h-screen bg-clinical-bg text-clinical-text flex font-sans">
      {/* ── Desktop Sidebar ── */}
      <aside className={`
        fixed inset-y-0 left-0 z-50 w-72 bg-clinical-card border-r border-clinical-border flex flex-col justify-between
        transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:flex-shrink-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        {/* Sidebar Header */}
        <div>
          <div className="h-20 flex items-center px-6 border-b border-clinical-border gap-3">
            <div className="w-10 h-10 rounded-xl bg-clinical-accent flex items-center justify-center shadow-[0_4px_12px_rgba(99,102,241,0.3)]">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="font-extrabold text-base tracking-tight text-white flex items-center gap-1.5">
                NEURON AI
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-clinical-highlight/20 text-clinical-highlight font-bold border border-clinical-highlight/30">
                  V2.0
                </span>
              </div>
              <div className="text-[10px] text-clinical-textMuted font-medium uppercase tracking-wider">
                Clinical Workstation
              </div>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-4 space-y-6">
            {sections.map((section) => (
              <div key={section} className="space-y-1">
                <div className="text-[10px] text-clinical-textMuted font-bold uppercase tracking-widest px-3 mb-2">
                  {section}
                </div>
                <div className="space-y-1">
                  {navItems
                    .filter((item) => item.section === section)
                    .map((item) => {
                      const Icon = item.icon;
                      const isActive = activePage === item.id;
                      return (
                        <button
                          key={item.id}
                          onClick={() => handleNavigate(item.id)}
                          className={`
                            w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-semibold tracking-wide transition-all
                            ${isActive 
                              ? 'bg-clinical-accent text-white shadow-[0_4px_12px_rgba(99,102,241,0.2)]' 
                              : 'text-clinical-textMuted hover:bg-clinical-border/50 hover:text-clinical-text'}
                          `}
                        >
                          <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-clinical-textMuted'}`} />
                          {item.name}
                        </button>
                      );
                    })}
                </div>
              </div>
            ))}
          </nav>
        </div>

        {/* Sidebar Footer User Info */}
        <div className="p-4 border-t border-clinical-border flex items-center justify-between bg-clinical-bg/30">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-clinical-border flex items-center justify-center text-clinical-highlight border border-clinical-highlight/30 font-bold uppercase">
              {username ? username.charAt(0) : 'U'}
            </div>
            <div className="overflow-hidden">
              <div className="text-xs font-bold text-white truncate max-w-[120px]">
                {username}
              </div>
              <div className="text-[10px] text-clinical-success font-semibold tracking-wider uppercase">
                {role}
              </div>
            </div>
          </div>
          <button 
            onClick={onLogout}
            className="p-2 rounded-lg text-clinical-textMuted hover:bg-clinical-danger/10 hover:text-clinical-danger transition-colors"
            title="Logout workstation"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </aside>

      {/* ── Mobile Sidebar Drawer Backdrop ── */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Main Layout View ── */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen">
        {/* Mobile View Topbar */}
        <header className="h-16 border-b border-clinical-border bg-clinical-card px-4 flex items-center justify-between lg:hidden flex-shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg text-clinical-text hover:bg-clinical-border transition-colors"
          >
            <Menu className="w-6 h-6" />
          </button>
          <div className="font-extrabold text-sm tracking-widest text-white">NEURON AI</div>
          <div className="w-10 h-10 rounded-full bg-clinical-border flex items-center justify-center text-xs font-bold uppercase">
            {username ? username.charAt(0) : 'U'}
          </div>
        </header>

        {/* Content Body Container */}
        <main className="flex-1 overflow-y-auto p-4 lg:p-8 max-w-[1920px] mx-auto w-full">
          {children}
        </main>
      </div>
    </div>
  );
};
