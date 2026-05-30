// SettingsView.tsx — Workstation Settings

import React, { useState, useEffect } from 'react';
import { Settings, KeyRound, Terminal, ShieldAlert } from 'lucide-react';
import { modelApi } from '../services/modelApi';

interface SettingsViewProps {
  onSuccess: (msg: string) => void;
  onError: (msg: string) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({ onSuccess, onError }) => {
  const [clinicName, setClinicName] = useState<string>('');
  const [stationId, setStationId] = useState<string>('');
  
  // Password state
  const [oldPassword, setOldPassword] = useState<string>('');
  const [newPassword, setNewPassword] = useState<string>('');
  
  const [savingSettings, setSavingSettings] = useState<boolean>(false);
  const [savingPassword, setSavingPassword] = useState<boolean>(false);

  useEffect(() => {
    const loadSettings = async () => {
      try {
        const data = await modelApi.getSettings();
        setClinicName(data.clinic_name || '');
        setStationId(data.station_id || '');
      } catch (err: any) {
        onError('Failed to load clinic settings.');
      }
    };
    loadSettings();
  }, [onError]);

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!clinicName || !stationId) {
      onError('All configuration fields are required.');
      return;
    }

    setSavingSettings(true);
    try {
      await modelApi.saveSettings({ clinic_name: clinicName, station_id: stationId });
      onSuccess('Clinic settings updated successfully.');
    } catch (err: any) {
      onError(err.message || 'Failed to save settings.');
    } finally {
      setSavingSettings(false);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!oldPassword || !newPassword) {
      onError('All password fields are required.');
      return;
    }
    if (newPassword.length < 6) {
      onError('New password must be at least 6 characters.');
      return;
    }

    setSavingPassword(true);
    try {
      await modelApi.changePassword(oldPassword, newPassword);
      onSuccess('Password updated successfully.');
      setOldPassword('');
      setNewPassword('');
    } catch (err: any) {
      onError(err.message || 'Failed to change password.');
    } finally {
      setSavingPassword(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-1 border-b border-clinical-border/40 pb-3">
        <h2 className="text-xl font-bold tracking-tight text-white uppercase tracking-wider">Workstation Settings</h2>
        <p className="text-xs text-clinical-textMuted font-medium">Configure clinic descriptors and credentials.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
        {/* Clinic Info form */}
        <div className="bg-clinical-card border border-clinical-border/40 rounded-lg p-5 shadow-sm flex flex-col justify-between">
          <form onSubmit={handleSaveSettings} className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-white border-b border-clinical-border/40 pb-3 flex items-center gap-2">
              <Settings className="w-4 h-4 text-clinical-highlight" />
              Workstation Node Configuration
            </h3>

            <div className="space-y-3">
              <div className="space-y-1">
                <label className="text-[10px] font-bold text-clinical-textMuted uppercase tracking-wider block">Clinic Name</label>
                <input
                  type="text"
                  value={clinicName}
                  onChange={(e) => setClinicName(e.target.value)}
                  className="w-full bg-clinical-bg border border-clinical-border/50 rounded-lg text-xs font-semibold px-3.5 py-2.5 text-white outline-none focus:border-clinical-highlight"
                />
              </div>
              <div className="space-y-1">
                <label className="text-[10px] font-bold text-clinical-textMuted uppercase tracking-wider block">Station ID Node</label>
                <input
                  type="text"
                  value={stationId}
                  onChange={(e) => setStationId(e.target.value)}
                  className="w-full bg-clinical-bg border border-clinical-border/50 rounded-lg text-xs font-semibold px-3.5 py-2.5 text-white outline-none focus:border-clinical-highlight"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={savingSettings}
              className="px-4 py-2 bg-clinical-accent hover:bg-clinical-accent/90 text-xs font-bold text-white rounded transition-colors uppercase tracking-wider disabled:opacity-50"
            >
              {savingSettings ? 'Saving...' : 'Save Configuration'}
            </button>
          </form>

          {/* System metadata */}
          <div className="border-t border-clinical-border/40 pt-4 mt-6">
            <h4 className="text-[10px] font-bold uppercase text-white flex items-center gap-1.5 mb-2">
              <Terminal className="w-3.5 h-3.5 text-clinical-textMuted" />
              Deployment Core Specification
            </h4>
            <div className="text-[10px] text-clinical-textMuted space-y-1 font-mono leading-relaxed">
              <div>Clinic Host: <span className="text-white">ON-PREMISE LOCALHOST</span></div>
              <div>Compliance Directives: <span className="text-clinical-success font-semibold">DPDP SECURE</span></div>
              <div>Database Engine: <span className="text-white">SQLITE3 LOCAL v2</span></div>
            </div>
          </div>
        </div>

        {/* Change Password form */}
        <div className="bg-clinical-card border border-clinical-border/40 rounded-lg p-5 shadow-sm flex flex-col justify-between">
          <form onSubmit={handleChangePassword} className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-white border-b border-clinical-border/40 pb-3 flex items-center gap-2">
              <KeyRound className="w-4 h-4 text-clinical-warning" />
              Credentials Management
            </h3>

            <div className="space-y-3">
              <div className="space-y-1">
                <label className="text-[10px] font-bold text-clinical-textMuted uppercase tracking-wider block">Current Password</label>
                <input
                  type="password"
                  value={oldPassword}
                  onChange={(e) => setOldPassword(e.target.value)}
                  className="w-full bg-clinical-bg border border-clinical-border/50 rounded-lg text-xs font-semibold px-3.5 py-2.5 text-white outline-none focus:border-clinical-highlight"
                  placeholder="Enter current password"
                />
              </div>
              <div className="space-y-1">
                <label className="text-[10px] font-bold text-clinical-textMuted uppercase tracking-wider block">New Cryptographic Password</label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full bg-clinical-bg border border-clinical-border/50 rounded-lg text-xs font-semibold px-3.5 py-2.5 text-white outline-none focus:border-clinical-highlight"
                  placeholder="Minimum 6 characters"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={savingPassword}
              className="px-4 py-2 bg-clinical-accent hover:bg-clinical-accent/90 text-xs font-bold text-white rounded transition-colors uppercase tracking-wider disabled:opacity-50"
            >
              {savingPassword ? 'Updating...' : 'Update Password'}
            </button>
          </form>

          <div className="mt-6 p-3 bg-clinical-danger/10 border border-clinical-danger/20 rounded text-[9px] text-clinical-danger flex gap-2 font-bold leading-normal">
            <ShieldAlert className="w-4.5 h-4.5 text-clinical-danger flex-shrink-0 mt-0.5" />
            <span>
              WARNING: Sessions are locked to 8 hours. Modifying your password will terminate active JWT tokens.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
