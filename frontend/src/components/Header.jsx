import React, { useState } from 'react'

export default function Header({
  backendStatus,
  onCheckBackend,
  currentUser,
  onOpenAuth,
  onLogout,
  onOpenSettings,
  notifications = [],
  darkMode,
  onToggleDark
}) {
  const [showNotifMenu, setShowNotifMenu] = useState(false)

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-logo-glow">
          <svg
            className="pulse-svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
          </svg>
        </div>

        <div className="brand-text">
          <div className="brand-name">
            PulseRisk <span className="ai-tag">AI</span>
          </div>

          <div className="brand-subtitle">
            Health & Risk Prediction System
          </div>
        </div>
      </div>

      <div className="header-actions">

        {/* Backend Connection Status */}
        <button
          className={`backend-pill ${backendStatus.online ? 'online' : 'demo'
            }`}
          onClick={onCheckBackend}
          title={backendStatus.message + ' (Click to re-ping)'}
        >
          <span className="status-dot"></span>

          <span className="status-label">
            {backendStatus.online
              ? 'Backend Live'
              : 'Demo Mode (Offline)'}
          </span>
        </button>

        {/* Dark Mode Toggle */}
        <button
          className="icon-btn theme-toggle"
          onClick={onToggleDark}
          title={
            darkMode
              ? 'Switch to Light Mode'
              : 'Switch to Dark Mode'
          }
          aria-label={
            darkMode
              ? 'Switch to Light Mode'
              : 'Switch to Dark Mode'
          }
        >
          {darkMode ? (
            /* Sun Icon */
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2" />
              <path d="M12 20v2" />
              <path d="m4.93 4.93 1.41 1.41" />
              <path d="m17.66 17.66 1.41 1.41" />
              <path d="M2 12h2" />
              <path d="M20 12h2" />
              <path d="m6.34 17.66-1.41 1.41" />
              <path d="m19.07 4.93-1.41 1.41" />
            </svg>
          ) : (
            /* Moon Icon */
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>

        {/* Notifications */}
        <div className="notif-wrapper">
          <button
            className="icon-btn notif-btn"
            onClick={() => setShowNotifMenu(!showNotifMenu)}
            title="Clinical Alerts"
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
              <path d="M13.73 21a2 2 0 0 1-3.46 0" />
            </svg>

            {notifications.length > 0 && (
              <span className="notif-badge">
                {notifications.length}
              </span>
            )}
          </button>

          {showNotifMenu && (
            <div className="notif-dropdown glass-card">
              <div className="notif-header">
                <h4>Clinical Alerts</h4>

                <span className="notif-count">
                  {notifications.length} Unread
                </span>
              </div>

              <div className="notif-list">
                {notifications.length > 0 ? (
                  notifications.map((n, i) => (
                    <div
                      key={i}
                      className={`notif-item ${n.severity || 'info'
                        }`}
                    >
                      <div className="notif-time">
                        {n.time}
                      </div>

                      <div className="notif-msg">
                        {n.message}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="notif-empty">
                    No new clinical alerts
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Settings */}
        <button
          className="icon-btn"
          onClick={onOpenSettings}
          title="Backend API Settings"
          aria-label="Backend API Settings"
        >
          <svg
            width="19"
            height="19"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="12" r="3" />

            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>

        {/* User */}
        {currentUser ? (
          <div className="user-profile-menu">
            <div
              className="user-avatar"
              title={currentUser.email}
            >
              {currentUser.name
                ? currentUser.name.charAt(0).toUpperCase()
                : 'U'}
            </div>

            <div className="user-info-text">
              <div className="user-name">
                {currentUser.name || 'User'}
              </div>

              <div className="user-role">
                Patient / Researcher
              </div>
            </div>

            <button
              className="btn-logout"
              onClick={onLogout}
              title="Sign Out"
              aria-label="Sign Out"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line
                  x1="21"
                  y1="12"
                  x2="9"
                  y2="12"
                />
              </svg>
            </button>
          </div>
        ) : (
          <button
            className="btn btn-primary btn-sm"
            onClick={onOpenAuth}
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
              <polyline points="10 17 15 12 10 7" />
              <line
                x1="15"
                y1="12"
                x2="3"
                y2="12"
              />
            </svg>

            Sign In / Register
          </button>
        )}
      </div>
    </header>
  )
}