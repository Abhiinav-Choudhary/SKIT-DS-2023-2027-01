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
import { authService, recordService, checkBackendHealth } from './services/api'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [currentUser, setCurrentUser] = useState(() => authService.getCurrentUser())
  const [records, setRecords] = useState(() => recordService.getRecords())
  const [showAuth, setShowAuth] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [backendStatus, setBackendStatus] = useState({ online: false, message: 'Checking...' })
  const [selectedRecord, setSelectedRecord] = useState(null)

  // Notifications based on vitals/records
  const notifications = [
    {
      time: '08:20 AM',
      message: 'Fasting glucose slightly elevated (98 mg/dL). Consider dietary review.',
      severity: 'mod'
    },
    {
      time: 'Yesterday',
      message: 'New AI cardiovascular risk assessment completed — Risk: 12.4% (Low).',
      severity: 'info'
    },
    {
      time: '3 days ago',
      message: 'Blood pressure trend stable over the past 7 days.',
      severity: 'low'
    }
  ]

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth().then(setBackendStatus)
  }, [])

  // Auto-login from stored token on mount
  useEffect(() => {
    const user = authService.getCurrentUser()
    if (user) setCurrentUser(user)
  }, [])

  const handleLogout = () => {
    authService.logout()
    setCurrentUser(null)
  }

  const handleSaveRecord = (record) => {
    const saved = recordService.addRecord(record)
    setRecords(recordService.getRecords())
    return saved
  }

  const handleDeleteRecord = (id) => {
    recordService.deleteRecord(id)
    setRecords(recordService.getRecords())
    if (selectedRecord?.id === id) setSelectedRecord(null)
  }

  const handleBackendCheck = async () => {
    setBackendStatus({ online: false, message: 'Re-checking...' })
    const result = await checkBackendHealth()
    setBackendStatus(result)
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentUser={currentUser}
      />

      {/* Main Area */}
      <div className="main-content">
        <Header
          backendStatus={backendStatus}
          onCheckBackend={handleBackendCheck}
          currentUser={currentUser}
          onOpenAuth={() => setShowAuth(true)}
          onLogout={handleLogout}
          onOpenSettings={() => setShowSettings(true)}
          notifications={notifications}
        />

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
            <PredictorView onSaveRecord={handleSaveRecord} />
          )}

          {activeTab === 'vitals' && (
            <VitalsView />
          )}

          {activeTab === 'records' && (
            <RecordsView
              records={records}
              onDeleteRecord={handleDeleteRecord}
              onNavigatePredictor={() => setActiveTab('predictor')}
            />
          )}
        </div>
      </div>

      {/* Auth Modal */}
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
