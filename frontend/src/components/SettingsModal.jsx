import React, { useState } from 'react';
import { getApiBaseUrl, setApiBaseUrl, checkBackendHealth } from '../services/api';

export default function SettingsModal({ onClose, onBackendStatusUpdate }) {
  const [apiUrl, setApiUrl] = useState(getApiBaseUrl());
  const [testResult, setTestResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    setApiBaseUrl(apiUrl);
    const result = await checkBackendHealth();
    setTestResult(result);
    setTesting(false);
    if (onBackendStatusUpdate) onBackendStatusUpdate(result);
  };

  const handleSave = () => {
    setApiBaseUrl(apiUrl);
    setSaved(true);
    setTimeout(() => { setSaved(false); onClose(); }, 1200);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box glass-card fade-in" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-brand">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-light)" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            <span>System Settings</span>
          </div>
          <button className="icon-btn" onClick={onClose}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="modal-body">
          <h3>Backend API Configuration</h3>
          <p className="modal-desc">Configure the backend server endpoint. The frontend will use this URL for authentication and health record operations.</p>

          <div className="form-group" style={{ marginTop: '20px' }}>
            <label className="form-label"><span>Backend API Base URL</span></label>
            <input
              type="url"
              className="form-input"
              value={apiUrl}
              onChange={(e) => { setApiUrl(e.target.value); setSaved(false); setTestResult(null); }}
              placeholder="http://localhost:5000/api"
            />
          </div>

          <div className="settings-actions">
            <button className="btn btn-secondary" onClick={handleTest} disabled={testing}>
              {testing ? <><span className="spinner"></span>Testing...</> : '🔌 Test Connection'}
            </button>
          </div>

          {testResult && (
            <div className={`test-result ${testResult.online ? 'online' : 'offline'} fade-in`}>
              <span className="tr-dot"></span>
              <span>{testResult.online ? '✅' : '❌'} {testResult.message}</span>
            </div>
          )}

          <div className="setting-info-box">
            <p>
              <strong>Note:</strong> The AI risk prediction engine runs entirely in the browser
              and does not require a backend connection. The backend (Node.js at <code>localhost:5000</code>)
              is only needed for user authentication and syncing health records.
            </p>
          </div>

          <div className="settings-footer-actions">
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave} disabled={saved}>
              {saved ? '✅ Saved!' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
