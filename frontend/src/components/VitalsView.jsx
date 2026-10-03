import React, { useState } from 'react';
import { vitalsService } from '../services/api';

const VITALS_LABELS = {
  bpSys: 'Systolic BP (mmHg)',
  bpDia: 'Diastolic BP (mmHg)',
  heartRate: 'Heart Rate (bpm)',
  glucose: 'Fasting Glucose (mg/dL)',
  spo2: 'SpO₂ (%)',
  weight: 'Weight (kg)',
  bmi: 'BMI (kg/m²)'
};

const RANGES = {
  bpSys: { low: 90, normal: [100, 129], high: 160 },
  bpDia: { low: 60, normal: [65, 84], high: 100 },
  heartRate: { low: 50, normal: [60, 100], high: 110 },
  glucose: { low: 70, normal: [70, 99], high: 126 },
  spo2: { low: 94, normal: [95, 100], high: 100 },
  bmi: { low: 18.5, normal: [18.5, 24.9], high: 30 }
};

function getVitalStatus(key, value) {
  const r = RANGES[key];
  if (!r) return { label: 'Recorded', cls: 'low' };
  if (value < r.low) return { label: 'Below Normal', cls: 'mod' };
  if (key === 'spo2') {
    return value >= r.low ? { label: 'Healthy', cls: 'low' } : { label: 'Low Oxygen', cls: 'critical' };
  }
  if (value > r.high) return { label: 'Above Normal', cls: 'high' };
  if (value >= r.normal[0] && value <= r.normal[1]) return { label: 'Normal', cls: 'low' };
  return { label: 'Borderline', cls: 'mod' };
}

export default function VitalsView() {
  const [vitals, setVitals] = useState(() => vitalsService.getVitals());
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    bpSys: 120, bpDia: 78, heartRate: 72, glucose: 95, spo2: 98, weight: 68.0, bmi: 22.2
  });
  const [formMsg, setFormMsg] = useState('');

  const latest = vitals.length > 0 ? vitals[vitals.length - 1] : null;

  const handleLogVital = (e) => {
    e.preventDefault();
    const saved = vitalsService.addVital(formData);
    setVitals([...vitals, saved]);
    setFormMsg('✅ Vitals logged successfully!');
    setTimeout(() => { setFormMsg(''); setShowForm(false); }, 1800);
  };

  const metrics = ['bpSys', 'bpDia', 'heartRate', 'glucose', 'spo2', 'bmi'];

  return (
    <div className="vitals-view fade-in">
      {/* Header Row */}
      <div className="section-title-row">
        <div>
          <h2>Biometric Vitals & Telemetry</h2>
          <p className="section-subtitle">7-day rolling biometric trend — {vitals.length} readings recorded</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Log New Reading
        </button>
      </div>

      {/* Log Vitals Form */}
      {showForm && (
        <form className="log-vitals-card glass-card fade-in" onSubmit={handleLogVital}>
          <div className="form-column-header">
            <h4>New Vitals Entry</h4>
            <span className="badge badge-low">Manual Input</span>
          </div>
          <div className="vitals-form-grid">
            {Object.keys(formData).map(key => (
              <div className="form-group" key={key}>
                <label className="form-label">
                  <span>{VITALS_LABELS[key] || key}</span>
                  <span className="unit">{formData[key]}</span>
                </label>
                <input
                  type="number"
                  step="0.1"
                  className="form-input"
                  value={formData[key]}
                  onChange={(e) => setFormData({ ...formData, [key]: parseFloat(e.target.value) })}
                  required
                />
              </div>
            ))}
          </div>
          {formMsg && <div className="success-msg">{formMsg}</div>}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary">Save Vitals Reading</button>
            <button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </form>
      )}

      {/* Current Readings */}
      {latest && (
        <>
          <div className="subsection-label">📊 Latest Biometric Snapshot — {latest.timestamp}</div>
          <div className="vital-status-grid">
            {metrics.map(key => {
              const status = getVitalStatus(key, latest[key]);
              return (
                <div key={key} className={`vital-status-card glass-card border-${status.cls}`}>
                  <div className="vs-label">{VITALS_LABELS[key]}</div>
                  <div className="vs-value">{latest[key]}</div>
                  <span className={`badge badge-${status.cls}`}>{status.label}</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Historical Trend Table */}
      <div className="subsection-label mt-16">📅 Historical Vitals Log</div>
      <div className="vitals-table-card glass-card">
        <div className="table-scroll">
          <table className="vitals-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>BP (mmHg)</th>
                <th>HR (bpm)</th>
                <th>Glucose (mg/dL)</th>
                <th>SpO₂ (%)</th>
                <th>Weight (kg)</th>
                <th>BMI</th>
              </tr>
            </thead>
            <tbody>
              {[...vitals].reverse().map((v) => {
                const bpStatus = getVitalStatus('bpSys', v.bpSys);
                return (
                  <tr key={v.id} className="table-row">
                    <td className="td-time">{v.timestamp}</td>
                    <td>
                      <span className={`td-val ${bpStatus.cls}`}>
                        {v.bpSys}/{v.bpDia}
                      </span>
                    </td>
                    <td>{v.heartRate}</td>
                    <td>
                      <span className={`td-val ${getVitalStatus('glucose', v.glucose).cls}`}>
                        {v.glucose}
                      </span>
                    </td>
                    <td>{v.spo2}</td>
                    <td>{v.weight}</td>
                    <td>{v.bmi}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Trend Summary Cards */}
      <div className="subsection-label mt-16">📈 7-Day Trend Analysis</div>
      <div className="trend-cards-grid">
        {['bpSys', 'glucose', 'heartRate'].map(key => {
          const values = vitals.map(v => v[key]);
          const avg = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
          const min = Math.min(...values);
          const max = Math.max(...values);
          const status = getVitalStatus(key, parseFloat(avg));
          return (
            <div key={key} className="trend-mini-card glass-card">
              <div className="trend-title">{VITALS_LABELS[key]}</div>
              <div className="trend-avg">{avg}</div>
              <div className="trend-minmax">Min: {min} &nbsp;|&nbsp; Max: {max}</div>
              <div className="trend-sparkline">
                {values.map((v, i) => (
                  <div
                    key={i}
                    className={`spark-bar ${getVitalStatus(key, v).cls}`}
                    style={{ height: `${Math.max(15, ((v - min) / (max - min + 0.01)) * 50 + 10)}px` }}
                    title={`${v}`}
                  ></div>
                ))}
              </div>
              <span className={`badge badge-${status.cls}`}>{status.label} Avg</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
