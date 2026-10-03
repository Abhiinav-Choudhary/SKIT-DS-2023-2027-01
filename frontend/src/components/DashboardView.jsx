import React from 'react';

export default function DashboardView({
  records,
  vitals,
  onNavigateTab,
  onSelectRecord,
  currentUser
}) {
  const latestVital = vitals.length > 0 ? vitals[vitals.length - 1] : null;

  return (
    <div className="dashboard-view fade-in">
      {/* Top Welcome / Hero Banner */}
      <div className="hero-banner glass-card">
        <div className="hero-text">
          <div className="hero-badge">
            <span className="sparkle-icon">✨</span> Clinical Intelligence Active
          </div>
          <h2>Welcome back, {currentUser?.name || 'Abhinav'}</h2>
          <p>
            Real-time biometric telemetry and AI risk models are synchronized.
            Last full clinical analysis completed today at 08:20 AM.
          </p>
        </div>
        <div className="hero-actions">
          <button
            className="btn btn-primary"
            onClick={() => onNavigateTab('predictor')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Run New AI Risk Assessment
          </button>
        </div>
      </div>

      {/* Top 4 Key Score Cards */}
      <div className="metric-cards-grid">
        <div className="metric-card glass-card">
          <div className="metric-header">
            <span className="metric-title">Cardiovascular 10-Yr Risk</span>
            <span className="metric-icon-wrap cardio">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20.42 4.58a5.4 5.4 0 0 0-7.65 0l-.77.78-.77-.78a5.4 5.4 0 0 0-7.65 7.65l.77.78L12 20.67l7.65-7.66.77-.78a5.4 5.4 0 0 0 0-7.65z" />
              </svg>
            </span>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">12.4%</div>
            <span className="badge badge-low">Low Risk</span>
          </div>
          <div className="metric-footer">
            <span className="footer-trend pos">↓ 1.2%</span> from last month's baseline
          </div>
        </div>

        <div className="metric-card glass-card">
          <div className="metric-header">
            <span className="metric-title">Type-2 Diabetes Risk</span>
            <span className="metric-icon-wrap diabetes">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="m19 11-4-7-4 7" />
                <path d="M9 11h10" />
                <path d="m14 11-2 10" />
                <circle cx="12" cy="18" r="3" />
              </svg>
            </span>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">28.5%</div>
            <span className="badge badge-mod">Moderate</span>
          </div>
          <div className="metric-footer">
            <span className="footer-trend warn">↑ 2.1%</span> elevated fasting glucose
          </div>
        </div>

        <div className="metric-card glass-card">
          <div className="metric-header">
            <span className="metric-title">Metabolic Wellness Index</span>
            <span className="metric-icon-wrap wellness">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v20" />
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
              </svg>
            </span>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">86<span className="val-sub">/100</span></div>
            <span className="badge badge-low">Optimal</span>
          </div>
          <div className="metric-footer">
            <span className="footer-trend pos">Top 15%</span> for demographic cohort
          </div>
        </div>

        <div className="metric-card glass-card">
          <div className="metric-header">
            <span className="metric-title">Hypertension & Stroke</span>
            <span className="metric-icon-wrap stroke">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            </span>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">18.2%</div>
            <span className="badge badge-low">Standard</span>
          </div>
          <div className="metric-footer">
            <span className="footer-trend pos">Stable</span> resting pulse pressure
          </div>
        </div>
      </div>

      {/* Live Vitals Ribbon */}
      <div className="section-title-row">
        <h3>Current Biometric Telemetry</h3>
        <button
          className="btn-link"
          onClick={() => onNavigateTab('vitals')}
        >
          View Full Trends & History →
        </button>
      </div>

      <div className="vitals-ribbon-grid">
        <div className="vital-ribbon-card glass-card">
          <div className="v-label">Blood Pressure</div>
          <div className="v-val">
            {latestVital ? `${latestVital.bpSys}/${latestVital.bpDia}` : '119/77'}
            <span className="v-unit">mmHg</span>
          </div>
          <span className="badge badge-low">Normotensive</span>
        </div>

        <div className="vital-ribbon-card glass-card">
          <div className="v-label">Resting Heart Rate</div>
          <div className="v-val">
            {latestVital ? latestVital.heartRate : 70}
            <span className="v-unit">bpm</span>
          </div>
          <span className="badge badge-low">Normal Sinus</span>
        </div>

        <div className="vital-ribbon-card glass-card">
          <div className="v-label">Fasting Glucose</div>
          <div className="v-val">
            {latestVital ? latestVital.glucose : 95}
            <span className="v-unit">mg/dL</span>
          </div>
          <span className="badge badge-low">Euglycemia</span>
        </div>

        <div className="vital-ribbon-card glass-card">
          <div className="v-label">Blood Oxygen (SpO2)</div>
          <div className="v-val">
            {latestVital ? latestVital.spo2 : 99}
            <span className="v-unit">%</span>
          </div>
          <span className="badge badge-low">Healthy Saturation</span>
        </div>

        <div className="vital-ribbon-card glass-card">
          <div className="v-label">Body Mass Index (BMI)</div>
          <div className="v-val">
            {latestVital ? latestVital.bmi : 22.2}
            <span className="v-unit">kg/m²</span>
          </div>
          <span className="badge badge-low">Normal Weight</span>
        </div>
      </div>

      {/* Two Column Grid: Risk Distribution + Recent Assessments */}
      <div className="dashboard-cols-grid">
        {/* Risk Distribution Breakdown */}
        <div className="col-card glass-card">
          <div className="col-header">
            <h4>Multi-Organ Risk Distribution</h4>
            <span className="model-chip">AI Model Ensemble</span>
          </div>

          <div className="distribution-bars">
            <div className="dist-item">
              <div className="dist-meta">
                <span className="dist-name">Cardiovascular Disease (CVD)</span>
                <span className="dist-pct low">12.4% (Low)</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill low" style={{ width: '12.4%' }}></div>
              </div>
            </div>

            <div className="dist-item">
              <div className="dist-meta">
                <span className="dist-name">Type 2 Diabetes Mellitus</span>
                <span className="dist-pct mod">28.5% (Moderate)</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill mod" style={{ width: '28.5%' }}></div>
              </div>
            </div>

            <div className="dist-item">
              <div className="dist-meta">
                <span className="dist-name">Hypertension & Arterial Stiffness</span>
                <span className="dist-pct low">18.2% (Low)</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill low" style={{ width: '18.2%' }}></div>
              </div>
            </div>

            <div className="dist-item">
              <div className="dist-meta">
                <span className="dist-name">Cerebrovascular Stroke Risk</span>
                <span className="dist-pct low">7.8% (Minimal)</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill low" style={{ width: '7.8%' }}></div>
              </div>
            </div>
          </div>

          <div className="clinical-advisory-note">
            <div className="note-title">⚡ Machine Learning Insight</div>
            <p>
              Primary elevation factor is postprandial glucose fluctuation. 
              Maintaining cardiovascular fitness keeps 10-year stroke and myocardial infarction risk below 15%.
            </p>
          </div>
        </div>

        {/* Recent Assessment Feed */}
        <div className="col-card glass-card">
          <div className="col-header">
            <h4>Recent AI Diagnostic Reports</h4>
            <button className="btn-link" onClick={() => onNavigateTab('records')}>
              All Records ({records.length})
            </button>
          </div>

          <div className="records-feed">
            {records.slice(0, 3).map((rec) => (
              <div
                key={rec.id}
                className="record-feed-item"
                onClick={() => {
                  onSelectRecord(rec);
                  onNavigateTab('records');
                }}
              >
                <div className="feed-item-left">
                  <div className="feed-disease">{rec.disease}</div>
                  <div className="feed-model">{rec.model} • {rec.date}</div>
                  <div className="feed-summary">{rec.summary}</div>
                </div>
                <div className="feed-item-right">
                  <div className="feed-score">{rec.riskScore}%</div>
                  <span className={`badge badge-${rec.riskLevel === 'Low' ? 'low' : rec.riskLevel === 'Moderate' ? 'mod' : 'high'}`}>
                    {rec.riskLevel}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <button
            className="btn btn-secondary w-full"
            style={{ marginTop: '16px' }}
            onClick={() => onNavigateTab('predictor')}
          >
            Launch Interactive Risk Model
          </button>
        </div>
      </div>
    </div>
  );
}
