/* ── Neuron AI v2.0 — Application Logic ──────────────────────────────────── */

const TOKEN_KEY = 'neuron_token';
const USERNAME_KEY = 'neuron_username';
const ROLE_KEY = 'neuron_role';

let currentModality = 'XRAY';
let contrastValue = 100, brightnessValue = 100;
let modalityChart = null, pathologyChart = null;
let historyPage = 1, historyPageSize = 20;

// ── Auth Helpers ────────────────────────────────────────────────────────────
function getToken() { return localStorage.getItem(TOKEN_KEY); }
function authHeaders() { return { 'Authorization': `Bearer ${getToken()}` }; }
function logout() { localStorage.clear(); window.location.replace('/login'); }

function checkAuth() {
    if (!getToken()) { window.location.replace('/login'); return false; }
    const u = localStorage.getItem(USERNAME_KEY) || 'User';
    document.getElementById('sidebar-username').textContent = u;
    document.getElementById('sidebar-role').textContent = localStorage.getItem(ROLE_KEY) || 'radiologist';
    document.getElementById('sidebar-avatar').textContent = u.charAt(0).toUpperCase();
    return true;
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    sidebar.classList.toggle('open');
    backdrop.classList.toggle('visible');
    document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : 'auto';
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    sidebar.classList.remove('open');
    backdrop.classList.remove('visible');
    document.body.style.overflow = 'auto';
}

document.getElementById('sidebar-backdrop')?.addEventListener('click', closeSidebar);

// ── Toast System ────────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✓', error: '✗', info: 'ℹ' };
    toast.innerHTML = `<span style="font-size:15px;font-weight:700">${icons[type]||'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(40px)'; setTimeout(() => toast.remove(), 300); }, 4000);
}

// ── Navigation ──────────────────────────────────────────────────────────────
function navigateTo(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById(`page-${pageId}`).classList.add('active');
    document.querySelector(`[data-page="${pageId}"]`).classList.add('active');

    if (pageId === 'dashboard') fetchDashboardMetrics();
    if (pageId === 'history') { historyPage = 1; fetchScanHistory(); }
    if (pageId === 'settings') loadSettings();

    if (window.innerWidth <= 1024) closeSidebar();
}

// ── Modality Selection ──────────────────────────────────────────────────────
function setModality(mod) {
    currentModality = mod;
    document.querySelectorAll('.mod-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`mod-${mod}`).classList.add('active');
}

// ── Image Controls ──────────────────────────────────────────────────────────
function applyFilters() {
    const img = document.getElementById('viewer-image');
    if (img) img.style.filter = `contrast(${contrastValue}%) brightness(${brightnessValue}%)`;
}
function resetFilters() {
    contrastValue = brightnessValue = 100;
    document.getElementById('contrast-slider').value = 100;
    document.getElementById('brightness-slider').value = 100;
    document.getElementById('contrast-val').textContent = '100%';
    document.getElementById('brightness-val').textContent = '100%';
    applyFilters();
}

// ── File Upload ─────────────────────────────────────────────────────────────
function processUpload(file) {
    const status = document.getElementById('upload-status');
    const progress = document.getElementById('progress-fill');
    const statusText = document.getElementById('status-text');
    status.classList.add('visible');
    statusText.textContent = 'Anonymizing & processing scan...';
    progress.style.width = '35%';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('modality', currentModality);

    fetch('/api/upload-scan', { method: 'POST', body: formData, headers: authHeaders() })
        .then(res => {
            progress.style.width = '85%';
            if (!res.ok) return res.json().then(d => { throw new Error(d.detail || 'Upload failed'); });
            return res.json();
        })
        .then(data => {
            progress.style.width = '100%';
            setTimeout(() => status.classList.remove('visible'), 500);
            renderResult(data);
            fetchDashboardMetrics();
            showToast(`Scan processed: ${data.pathology_detected} (${(data.confidence_score*100).toFixed(1)}%)`, data.pathology_detected === 'Normal' ? 'success' : 'info');
        })
        .catch(err => {
            status.classList.remove('visible');
            showToast(err.message, 'error');
            if (err.message.includes('401') || err.message.includes('token')) logout();
        });
}

// ── Render Scan Result ──────────────────────────────────────────────────────
function renderResult(data) {
    // Viewer image
    document.getElementById('viewer-empty').classList.add('hidden');
    const container = document.getElementById('viewer-canvas');
    container.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.style.cssText = 'position:relative;display:inline-block;';
    const img = document.createElement('img');
    img.id = 'viewer-image';
    img.src = `data:image/png;base64,${data.img_base64}`;
    img.style.cssText = `max-width:100%;max-height:400px;object-fit:contain;display:block;filter:contrast(${contrastValue}%) brightness(${brightnessValue}%)`;
    wrap.appendChild(img);

    if (data.bbox) {
        const box = document.createElement('div');
        box.className = 'bbox-overlay';
        box.style.cssText = `left:${data.bbox.x}%;top:${data.bbox.y}%;width:${data.bbox.w}%;height:${data.bbox.h}%`;
        const lbl = document.createElement('span');
        lbl.className = 'bbox-label';
        lbl.textContent = `${data.bbox.label} (${(data.confidence_score*100).toFixed(1)}%)`;
        box.appendChild(lbl);
        wrap.appendChild(box);
    }
    container.appendChild(wrap);

    // Metadata
    document.getElementById('meta-card').classList.remove('hidden');
    document.getElementById('meta-hash').textContent = data.patient_hash;
    document.getElementById('meta-modality').textContent = data.modality;
    document.getElementById('meta-demo').textContent = `${data.metadata.age} / ${data.metadata.sex}`;
    document.getElementById('meta-study').textContent = data.metadata.description;

    // Diagnosis
    document.getElementById('diag-placeholder').classList.add('hidden');
    document.getElementById('diag-results').classList.remove('hidden');
    const findingEl = document.getElementById('primary-finding');
    const confEl = document.getElementById('primary-conf');
    const badge = document.getElementById('diag-badge');
    findingEl.textContent = data.pathology_detected;
    confEl.textContent = `${(data.confidence_score*100).toFixed(1)}%`;

    if (data.pathology_detected === 'Normal') {
        findingEl.className = 'text-emerald'; confEl.className = 'text-emerald font-mono';
        badge.className = 'diag-badge normal'; badge.textContent = 'Normal';
    } else if (data.pathology_detected === 'Inconclusive') {
        findingEl.className = 'text-amber'; confEl.className = 'text-amber font-mono';
        badge.className = 'diag-badge inconclusive'; badge.textContent = 'Inconclusive';
    } else {
        findingEl.className = 'text-rose'; confEl.className = 'text-rose font-mono';
        badge.className = 'diag-badge abnormal'; badge.textContent = data.pathology_detected;
    }
    badge.classList.remove('hidden');

    // Model info
    const modelEl = document.getElementById('model-info');
    if (data.model_info) { modelEl.textContent = data.model_info; modelEl.classList.remove('hidden'); }

    // Prediction bars
    const barsEl = document.getElementById('pred-bars');
    barsEl.innerHTML = '';
    Object.entries(data.predictions).forEach(([name, val]) => {
        const pct = (val * 100).toFixed(1);
        const isTarget = name === data.pathology_detected;
        const color = isTarget
            ? (name === 'Normal' ? 'var(--emerald)' : name === 'Inconclusive' ? 'var(--amber)' : 'var(--rose)')
            : 'var(--text-dim)';
        barsEl.innerHTML += `<div class="pred-bar-wrap"><div class="pred-row"><span style="color:${isTarget?'white':'var(--text-muted)'}">${name}</span><span class="font-mono" style="font-size:10px;color:${isTarget?'var(--cyan)':'var(--text-dim)'}">${pct}%</span></div><div class="pred-bar"><div class="pred-fill" style="width:${pct}%;background:${color}"></div></div></div>`;
    });
}

// ── Dashboard Metrics ───────────────────────────────────────────────────────
async function fetchDashboardMetrics() {
    try {
        const res = await fetch('/api/dashboard-metrics', { headers: authHeaders() });
        if (res.status === 401) { logout(); return; }
        if (!res.ok) throw new Error('Failed to load metrics');
        const m = await res.json();

        // Stats
        document.getElementById('stat-total-scans').textContent = m.total_scans;
        document.getElementById('stat-positive').textContent = `${m.positive_rate}%`;
        const abnormalCount = Object.entries(m.pathology_counts || {}).reduce((sum, [key, val]) => {
            if (key === 'Normal' || key === 'Inconclusive') return sum;
            return sum + (val || 0);
        }, 0);
        document.getElementById('stat-abnormal').textContent = abnormalCount;
        const modEntries = Object.entries(m.modality_counts || {});
        const topMod = modEntries.sort((a, b) => (b[1] || 0) - (a[1] || 0))[0];
        document.getElementById('stat-top-modality').textContent = topMod && topMod[1] ? `${topMod[0]} (${topMod[1]})` : '—';

        // Recent logs table
        const tbody = document.getElementById('recent-logs');
        if (m.recent_scans.length) {
            tbody.innerHTML = m.recent_scans.map(s => {
                const modCls = s.scan_type.toLowerCase();
                const pCls = s.pathology_detected === 'Normal' ? 'text-emerald' : s.pathology_detected === 'Inconclusive' ? 'text-amber' : 'text-rose';
                return `<tr><td class="font-mono text-dim">${s.timestamp}</td><td class="font-mono text-cyan" style="font-size:10px">${s.patient_hash}</td><td><span class="mod-tag ${modCls}">${s.scan_type}</span></td><td class="${pCls}" style="font-weight:600">${s.pathology_detected}</td><td class="font-mono text-dim">${(s.confidence_score*100).toFixed(0)}%</td></tr>`;
            }).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:30px;color:var(--text-dim)">No scans logged yet</td></tr>';
        }

        // Charts
        updateCharts(m);

    } catch (err) { console.error(err); }
}

function updateCharts(m) {
    const mc = m.modality_counts;
    if (modalityChart) modalityChart.destroy();
    modalityChart = new Chart(document.getElementById('chart-modality'), {
        type: 'doughnut',
        data: { labels: ['X-Ray','CT','MRI'], datasets: [{ data: [mc.XRAY,mc.CT,mc.MRI], backgroundColor: ['#06b6d4','#6366f1','#f59e0b'], borderColor: '#0f172a', borderWidth: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } } }, title: { display: true, text: 'Modality Distribution', color: '#e2e8f0', font: { family: 'Outfit', size: 11 } } } }
    });
    const pc = m.pathology_counts;
    const pLabels = Object.keys(pc), pData = Object.values(pc);
    if (pathologyChart) pathologyChart.destroy();
    pathologyChart = new Chart(document.getElementById('chart-pathology'), {
        type: 'bar',
        data: { labels: pLabels.length ? pLabels : ['Normal'], datasets: [{ data: pData.length ? pData : [0], backgroundColor: '#10b981', borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 9 } } }, y: { grid: { color: '#1e293b' }, ticks: { precision: 0, color: '#94a3b8', font: { family: 'Outfit', size: 9 } } } }, plugins: { legend: { display: false }, title: { display: true, text: 'Pathology Breakdown', color: '#e2e8f0', font: { family: 'Outfit', size: 11 } } } }
    });
}


// ── Scan History ────────────────────────────────────────────────────────────
async function fetchScanHistory() {
    const type = document.getElementById('filter-modality')?.value || '';
    const pathology = document.getElementById('filter-pathology')?.value || '';
    try {
        const params = new URLSearchParams({ page: historyPage, size: historyPageSize });
        if (type) params.append('scan_type', type);
        if (pathology) params.append('pathology', pathology);
        const res = await fetch(`/api/scans?${params}`, { headers: authHeaders() });
        if (res.status === 401) { logout(); return; }
        const data = await res.json();

        const tbody = document.getElementById('history-body');
        if (data.scans.length) {
            tbody.innerHTML = data.scans.map(s => {
                const modCls = s.scan_type.toLowerCase();
                const pCls = s.pathology_detected === 'Normal' ? 'text-emerald' : s.pathology_detected === 'Inconclusive' ? 'text-amber' : 'text-rose';
                const ptTag = s.pytorch_executed === 'true' ? '<span class="model-tag" style="margin-left:4px;font-size:8px">AI</span>' : '';
                return `<tr><td class="font-mono text-dim">${s.timestamp}</td><td class="font-mono text-cyan" style="font-size:10px">${s.patient_hash}</td><td><span class="mod-tag ${modCls}">${s.scan_type}</span></td><td class="${pCls}" style="font-weight:600">${s.pathology_detected}${ptTag}</td><td class="font-mono text-dim">${(s.confidence_score*100).toFixed(0)}%</td></tr>`;
            }).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:30px;color:var(--text-dim)">No scan records found</td></tr>';
        }

        // Pagination
        const pagEl = document.getElementById('history-pagination');
        pagEl.innerHTML = '';
        const prevBtn = `<button class="page-btn" ${data.page<=1?'disabled':''} onclick="historyPage--;fetchScanHistory()">← Prev</button>`;
        const info = `<span class="page-info">Page ${data.page} of ${data.total_pages} (${data.total} records)</span>`;
        const nextBtn = `<button class="page-btn" ${data.page>=data.total_pages?'disabled':''} onclick="historyPage++;fetchScanHistory()">Next →</button>`;
        pagEl.innerHTML = prevBtn + info + nextBtn;

    } catch (err) { console.error(err); }
}

// ── Settings ────────────────────────────────────────────────────────────────
async function loadSettings() {
    try {
        const res = await fetch('/api/settings', { headers: authHeaders() });
        if (res.status === 401) { logout(); return; }
        const s = await res.json();
        document.getElementById('set-clinic-name').value = s.clinic_name || '';
        document.getElementById('set-station-id').value = s.station_id || '';
    } catch (err) { showToast('Failed to load settings', 'error'); }
}

async function saveSettings() {
    const body = {
        clinic_name: document.getElementById('set-clinic-name').value,
        station_id: document.getElementById('set-station-id').value,
    };
    try {
        const res = await fetch('/api/settings', { method: 'POST', headers: { ...authHeaders(), 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!res.ok) throw new Error('Save failed');
        showToast('Settings saved successfully', 'success');
    } catch (err) { showToast(err.message, 'error'); }
}

async function changePassword() {
    const oldPw = document.getElementById('set-old-password').value;
    const newPw = document.getElementById('set-new-password').value;
    if (!oldPw || !newPw) { showToast('Both fields required', 'error'); return; }
    const form = new FormData(); form.append('old_password', oldPw); form.append('new_password', newPw);
    try {
        const res = await fetch('/api/auth/change-password', { method: 'POST', body: form, headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed');
        showToast('Password changed successfully', 'success');
        document.getElementById('set-old-password').value = '';
        document.getElementById('set-new-password').value = '';
    } catch (err) { showToast(err.message, 'error'); }
}

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    const path = window.location.pathname.toLowerCase();
    const pageFromPath = path === '/about' ? 'about' : path === '/models' ? 'models' : 'dashboard';
    navigateTo(pageFromPath);

    // Image controls
    const cs = document.getElementById('contrast-slider');
    const bs = document.getElementById('brightness-slider');
    cs.addEventListener('input', e => { contrastValue = e.target.value; document.getElementById('contrast-val').textContent = contrastValue+'%'; applyFilters(); });
    bs.addEventListener('input', e => { brightnessValue = e.target.value; document.getElementById('brightness-val').textContent = brightnessValue+'%'; applyFilters(); });

    // Dropzone
    const dz = document.getElementById('dropzone');
    const fi = document.getElementById('file-input');
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag-over'); if (e.dataTransfer.files.length) processUpload(e.dataTransfer.files[0]); });
    dz.addEventListener('click', () => fi.click());
    fi.addEventListener('change', e => { if (e.target.files.length) processUpload(e.target.files[0]); });
});
