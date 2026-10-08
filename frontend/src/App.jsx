import { useState, useEffect } from 'react'
import './index.css'
import './App.css'

import Header from './components/Header'
import Sidebar from './components/Sidebar'
import DashboardView from './components/DashboardView'
import PredictorView from './components/PredictorView'
import VitalsView from './components/VitalsView'
import RecordsView from './components/RecordsView'
import AuthModal from './components/AuthModal'
import SettingsModal from './components/SettingsModal'

import {
  authService,
  recordService,
  checkBackendHealth
} from './services/api'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')

  const [currentUser, setCurrentUser] = useState(() =>
    authService.getCurrentUser()
  )

  const [records, setRecords] = useState(() =>
    recordService.getRecords()
  )

  const [showAuth, setShowAuth] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  const [backendStatus, setBackendStatus] = useState({
    online: false,
    message: 'Checking...'
  })

  const [selectedRecord, setSelectedRecord] = useState(null)

  // =========================
  // DARK MODE
  // =========================

  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('theme')

    if (savedTheme === 'dark') {
      return true
    }

    if (savedTheme === 'light') {
      return false
    }

    return window.matchMedia(
      '(prefers-color-scheme: dark)'
    ).matches
  })

  // Apply theme to entire application
  useEffect(() => {
    const root = document.documentElement

    if (darkMode) {
      root.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      root.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [darkMode])

  // Toggle dark mode
  const handleToggleDark = () => {
    setDarkMode((previous) => !previous)
  }

  // =========================
  // NOTIFICATIONS
  // =========================

  const notifications = [
    {
      time: '08:20 AM',
      message:
        'Fasting glucose slightly elevated (98 mg/dL). Consider dietary review.',
      severity: 'mod'
    },
    {
      time: 'Yesterday',
      message:
        'New AI cardiovascular risk assessment completed — Risk: 12.4% (Low).',
      severity: 'info'
    },
    {
      time: '3 days ago',
      message:
        'Blood pressure trend stable over the past 7 days.',
      severity: 'low'
    }
  ]

  // =========================
  // BACKEND HEALTH
  // =========================

  useEffect(() => {
    checkBackendHealth().then(setBackendStatus)
  }, [])

  // =========================
  // AUTO LOGIN
  // =========================

  useEffect(() => {
    const user = authService.getCurrentUser()

    if (user) {
      setCurrentUser(user)
    }
  }, [])

  // =========================
  // LOGOUT
  // =========================

  const handleLogout = () => {
    authService.logout()
    setCurrentUser(null)
  }

  // =========================
  // SAVE RECORD
  // =========================

  const handleSaveRecord = (record) => {
    const saved = recordService.addRecord(record)

    setRecords(recordService.getRecords())

    return saved
  }

  // =========================
  // DELETE RECORD
  // =========================

  const handleDeleteRecord = (id) => {
    recordService.deleteRecord(id)
    setRecords(recordService.getRecords())

    if (selectedRecord?.id === id) {
      setSelectedRecord(null)
    }
  }

  // =========================
  // CHECK BACKEND AGAIN
  // =========================

  const handleBackendCheck = async () => {
    setBackendStatus({
      online: false,
      message: 'Re-checking...'
    })

    const result = await checkBackendHealth()

    setBackendStatus(result)
  }

  // =========================
  // UI
  // =========================

  return (
    <div className="app-container">

      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentUser={currentUser}
      />

      <div className="main-content">

        {/* Header */}
        <Header
          backendStatus={backendStatus}
          onCheckBackend={handleBackendCheck}
          currentUser={currentUser}
          onOpenAuth={() => setShowAuth(true)}
          onLogout={handleLogout}
          onOpenSettings={() => setShowSettings(true)}
          notifications={notifications}
          darkMode={darkMode}
          onToggleDark={handleToggleDark}
        />

        {/* Main Page Content */}
        <div className="page-body">

          {activeTab === 'dashboard' && (
            <DashboardView
              records={records}
              vitals={[]}
              onNavigateTab={setActiveTab}
              onSelectRecord={setSelectedRecord}
              currentUser={currentUser}
            />
          )}

          {activeTab === 'predictor' && (
            <PredictorView
              onSaveRecord={handleSaveRecord}
            />
          )}

          {activeTab === 'vitals' && (
            <VitalsView />
          )}

          {activeTab === 'records' && (
            <RecordsView
              records={records}
              onDeleteRecord={handleDeleteRecord}
              onNavigatePredictor={() =>
                setActiveTab('predictor')
              }
            />
          )}

        </div>
      </div>

      {/* Authentication Modal */}
      {showAuth && (
        <AuthModal
          onClose={() => setShowAuth(false)}
          onSuccess={(user) => {
            setCurrentUser(user)
            setShowAuth(false)
          }}
        />
      )}

      {/* Settings Modal */}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onBackendStatusUpdate={setBackendStatus}
        />
      )}

    </div>
  )
}

export default App