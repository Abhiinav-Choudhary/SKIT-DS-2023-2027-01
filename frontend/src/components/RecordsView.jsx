import React, { useState } from 'react';

export default function RecordsView({ records, onDeleteRecord, onNavigatePredictor }) {
  const [search, setSearch] = useState('');
  const [filterLevel, setFilterLevel] = useState('All');
  const [expanded, setExpanded] = useState(null);

  const filtered = records.filter(r => {
    const matchSearch = r.disease.toLowerCase().includes(search.toLowerCase()) ||
      r.summary.toLowerCase().includes(search.toLowerCase());
    const matchLevel = filterLevel === 'All' || r.riskLevel === filterLevel;
    return matchSearch && matchLevel;
  });

  const getBadgeClass = (level) => {
    if (!level) return 'badge-low';
    const l = level.toLowerCase();
    if (l.includes('critical') || l.includes('high')) return 'badge-critical';
    if (l.includes('elevated')) return 'badge-high';
    if (l.includes('moderate')) return 'badge-mod';
    return 'badge-low';
  };

  return (
    <div className="records-view fade-in">
      {/* Header */}
      <div className="section-title-row">
        <div>
          <h2>Patient Health Records</h2>
          <p className="section-subtitle">{records.length} AI diagnostic assessments on file</p>
        </div>
        <button className="btn btn-primary" onClick={onNavigatePredictor}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Assessment
        </button>
      </div>

      {/* Search & Filter */}
      <div className="records-filter-bar glass-card">
        <div className="search-wrap">
          <svg className="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            className="form-input search-input"
            placeholder="Search diagnoses, summaries..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="filter-chips">
          {['All', 'Low', 'Moderate', 'Elevated', 'Critical / High'].map(level => (
            <button
              key={level}
              className={`filter-chip ${filterLevel === level ? 'active' : ''}`}
              onClick={() => setFilterLevel(level)}
            >
              {level}
            </button>
          ))}
        </div>
      </div>

      {/* Records Summary Bar */}
      <div className="records-stats-bar">
        <div className="stat-item">
          <span className="stat-num">{records.length}</span>
          <span className="stat-desc">Total Assessments</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <span className="stat-num low">{records.filter(r => r.riskLevel === 'Low').length}</span>
          <span className="stat-desc">Low Risk</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <span className="stat-num mod">{records.filter(r => r.riskLevel === 'Moderate').length}</span>
          <span className="stat-desc">Moderate</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <span className="stat-num high">{records.filter(r => r.riskLevel === 'Elevated' || r.riskLevel?.includes('High')).length}</span>
          <span className="stat-desc">Elevated / High</span>
        </div>
      </div>

      {/* Records List */}
      {filtered.length === 0 ? (
        <div className="empty-state glass-card">
          <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
          </svg>
          <h4>No matching records found</h4>
          <p>Try adjusting filters or run a new AI risk assessment.</p>
          <button className="btn btn-primary" onClick={onNavigatePredictor}>Run New Assessment</button>
        </div>
      ) : (
        <div className="records-list">
          {filtered.map((rec) => {
            const isOpen = expanded === rec.id;
            return (
              <div key={rec.id} className={`record-card glass-card ${isOpen ? 'expanded' : ''}`}>
                <div className="record-card-header" onClick={() => setExpanded(isOpen ? null : rec.id)}>
                  <div className="rec-left">
                    <div className="rec-disease-name">{rec.disease}</div>
                    <div className="rec-meta">
                      <span className="rec-model">{rec.model}</span>
                      <span className="rec-dot">•</span>
                      <span className="rec-date">{rec.date}</span>
                    </div>
                  </div>
                  <div className="rec-right">
                    <div className="rec-score">{rec.riskScore}%</div>
                    <span className={`badge ${getBadgeClass(rec.riskLevel)}`}>{rec.riskLevel}</span>
                    <svg
                      className={`expand-chevron ${isOpen ? 'open' : ''}`}
                      width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                    >
                      <polyline points="6 9 12 15 18 9" />
                    </svg>
                  </div>
                </div>

                {isOpen && (
                  <div className="record-detail fade-in">
                    <div className="detail-divider" />

                    {/* Summary */}
                    <div className="detail-summary">{rec.summary}</div>

                    {/* Parameters */}
                    {rec.parameters && (
                      <div className="params-grid">
                        {Object.entries(rec.parameters).map(([k, v]) => (
                          <div key={k} className="param-chip">
                            <span className="param-key">{k}</span>
                            <span className="param-val">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="detail-actions">
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => {
                          onDeleteRecord(rec.id);
                          setExpanded(null);
                        }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6l-1 14H6L5 6" />
                          <path d="M10 11v6M14 11v6" />
                          <path d="M9 6V4h6v2" />
                        </svg>
                        Delete Record
                      </button>
                      <button className="btn btn-secondary btn-sm" onClick={() => window.print()}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="6 9 6 2 18 2 18 9" />
                          <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2" />
                          <rect x="6" y="14" width="12" height="8" />
                        </svg>
                        Export PDF
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
