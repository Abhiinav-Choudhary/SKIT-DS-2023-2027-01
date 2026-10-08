import React from 'react';

export default function Sidebar({ activeTab, setActiveTab, currentUser }) {
  const navItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="7" height="9" rx="1" />
          <rect x="14" y="3" width="7" height="5" rx="1" />
          <rect x="14" y="12" width="7" height="9" rx="1" />
          <rect x="3" y="16" width="7" height="5" rx="1" />
        </svg>
      )
    },
    {
      id: 'predictor',
      label: 'AI Risk Predictor',
      badge: 'ML Engine',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2a10 10 0 0 1 10 10c0 5.5-4.5 10-10 10S2 17.5 2 12A10 10 0 0 1 12 2z" />
          <path d="M12 6v6l4 2" />
        </svg>
      )
    },
    {
      id: 'vitals',
      label: 'Vitals & Telemetry',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
      )
    },
    {
      id: 'records',
      label: 'Health Records',
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
      )
    }
  ];

  return (
    <aside className="app-sidebar">
      <div className="sidebar-nav">
        <div className="nav-group-title">CLINICAL SUITE</div>
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
            onClick={() => setActiveTab(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
            {item.badge && <span className="nav-tag">{item.badge}</span>}
          </button>
        ))}
      </div>

      <div className="sidebar-patient-card glass-card">
        <div className="patient-card-header">
          <span className="pulse-dot"></span>
          <span className="patient-card-title">ACTIVE SUBJECT</span>
        </div>
        <div className="patient-meta">
          <div className="p-name">{currentUser?.name || 'Abhinav Chaudhary'}</div>
          <div className="p-details">Age: 46 • Blood: B+ • ID: #SKIT-01</div>
        </div>
        <div className="patient-status-row">
          <div className="stat-pill">
            <span className="stat-pill-label">Risk Profile</span>
            <span className="stat-pill-val low">Low (12.4%)</span>
          </div>
          <div className="stat-pill">
            <span className="stat-pill-label">BMI</span>
            <span className="stat-pill-val">22.4</span>
          </div>
        </div>
      </div>

      <div className="sidebar-footer">
        <div className="footer-dept">SKIT Dept. of Data Science</div>
        <div className="footer-batch">Batch 2023–2027 • Project 01</div>
      </div>
    </aside>
  );
}
