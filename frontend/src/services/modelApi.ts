// modelApi.ts — Model Service API Layer

export interface UserSession {
  access_token: string;
  token_type: string;
  username: string;
  role: string;
}

export interface MedicalMetadata {
  patient_id?: string;
  age?: string;
  sex?: string;
  description?: string;
  manufacturer?: string;
  magnetic_field_strength?: string;
  kvp?: string;
  pixel_spacing?: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
}

export interface ScanResult {
  status: string;
  scan_id: number;
  patient_hash: string;
  modality: 'XRAY' | 'CT' | 'MRI';
  metadata: MedicalMetadata;
  pathology_detected: string;
  confidence_score: number;
  predictions: Record<string, number>;
  bbox: BoundingBox | null;
  img_base64: string;
  timestamp: string;
  pytorch_executed: boolean;
  model_info: string;
  inference_latency: number; // Measured by server in ms
  latency_ms?: number; // Total roundtrip time in ms
}

export interface ScanRecord {
  id: number;
  patient_hash: string;
  scan_type: 'XRAY' | 'CT' | 'MRI';
  pathology_detected: string;
  confidence_score: number;
  timestamp: string;
  pytorch_executed: string;
  inference_latency: number;
}

export interface ScanHistoryResponse {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  scans: ScanRecord[];
}

export interface DashboardMetrics {
  total_scans: number;
  positive_rate: number;
  modality_counts: {
    XRAY: number;
    CT: number;
    MRI: number;
  };
  modality_latencies: {
    XRAY: number;
    CT: number;
    MRI: number;
  };
  pathology_counts: Record<string, number>;
  recent_scans: ScanRecord[];
}

export interface ClinicSettings {
  clinic_name?: string;
  station_id?: string;
}

const TOKEN_KEY = 'neuron_token';
const USERNAME_KEY = 'neuron_username';
const ROLE_KEY = 'neuron_role';

export const modelApi = {
  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  },

  getHeaders(): HeadersInit {
    const token = this.getToken();
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  },

  setSession(session: UserSession) {
    localStorage.setItem(TOKEN_KEY, session.access_token);
    localStorage.setItem(USERNAME_KEY, session.username);
    localStorage.setItem(ROLE_KEY, session.role);
  },

  clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USERNAME_KEY);
    localStorage.removeItem(ROLE_KEY);
  },

  isAuthenticated(): boolean {
    return !!this.getToken();
  },

  async login(username: string, password: string): Promise<UserSession> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const res = await fetch('/api/auth/login', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Authentication failed.');
    }

    const data: UserSession = await res.json();
    this.setSession(data);
    return data;
  },

  async uploadScan(file: File, modality: 'XRAY' | 'CT' | 'MRI'): Promise<ScanResult> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modality', modality);

    const tStart = performance.now();
    const res = await fetch('/api/upload-scan', {
      method: 'POST',
      body: formData,
      headers: this.getHeaders(),
    });
    const tEnd = performance.now();

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Failed to process clinical scan.');
    }

    const data: ScanResult = await res.json();
    data.latency_ms = Math.round(tEnd - tStart);
    return data;
  },

  async getDashboardMetrics(): Promise<DashboardMetrics> {
    const res = await fetch('/api/dashboard-metrics', {
      headers: this.getHeaders(),
    });

    if (!res.ok) {
      if (res.status === 401) {
        this.clearSession();
      }
      throw new Error('Failed to retrieve telemetry metrics.');
    }

    return res.json();
  },

  async getScanHistory(
    page: number = 1,
    size: number = 20,
    scanType?: string,
    pathology?: string
  ): Promise<ScanHistoryResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      size: size.toString(),
    });
    if (scanType) params.append('scan_type', scanType);
    if (pathology) params.append('pathology', pathology);

    const res = await fetch(`/api/scans?${params}`, {
      headers: this.getHeaders(),
    });

    if (!res.ok) {
      throw new Error('Failed to retrieve scan records.');
    }

    return res.json();
  },

  async getSettings(): Promise<ClinicSettings> {
    const res = await fetch('/api/settings', {
      headers: this.getHeaders(),
    });

    if (!res.ok) {
      throw new Error('Failed to load settings.');
    }

    return res.json();
  },

  async saveSettings(settings: ClinicSettings): Promise<{ status: string }> {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: {
        ...this.getHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    });

    if (!res.ok) {
      throw new Error('Failed to update clinic configuration.');
    }

    return res.json();
  },

  async changePassword(oldPw: string, newPw: string): Promise<{ status: string; message: string }> {
    const formData = new FormData();
    formData.append('old_password', oldPw);
    formData.append('new_password', newPw);

    const res = await fetch('/api/auth/change-password', {
      method: 'POST',
      body: formData,
      headers: this.getHeaders(),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Password update failed.');
    }

    return data;
  },
};
