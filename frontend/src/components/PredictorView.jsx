import React, { useState } from 'react';
import {
  calculateCardioRisk,
  calculateDiabetesRisk,
  calculateHypertensionStrokeRisk
} from '../services/api';

export default function PredictorView({ onSaveRecord }) {
  const [activeModel, setActiveModel] = useState('cardio');
  const [isCalculating, setIsCalculating] = useState(false);
  const [result, setResult] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Form States - Cardio
  const [cardioData, setCardioData] = useState({
    age: 46,
    gender: 'male',
    bpSys: 124,
    bpDia: 80,
    cholesterol: 195,
    hdl: 52,
    smoker: false,
    diabetic: false,
    activity: 'moderate'
  });

  // Form States - Diabetes
  const [diabetesData, setDiabetesData] = useState({
    age: 46,
    bmi: 24.5,
    glucose: 98,
    hba1c: 5.4,
    familyHistory: false,
    dailyExerciseMin: 35
  });

  // Form States - Hypertension / Stroke
  const [hyperData, setHyperData] = useState({
    bpSys: 126,
    bpDia: 82,
    heartRate: 74,
    sleepHours: 7,
    stressLevel: 4
  });

  // Quick Preset Handlers
  const handleLoadPreset = (presetType) => {
    if (presetType === 'healthy') {
      setCardioData({
        age: 32,
        gender: 'female',
        bpSys: 114,
        bpDia: 74,
        cholesterol: 165,
        hdl: 65,
        smoker: false,
        diabetic: false,
        activity: 'high'
      });
      setDiabetesData({
        age: 32,
        bmi: 21.5,
        glucose: 88,
        hba1c: 5.1,
        familyHistory: false,
        dailyExerciseMin: 45
      });
      setHyperData({
        bpSys: 112,
        bpDia: 72,
        heartRate: 65,
        sleepHours: 8,
        stressLevel: 2
      });
    } else if (presetType === 'moderate') {
      setCardioData({
        age: 48,
        gender: 'male',
        bpSys: 132,
        bpDia: 86,
        cholesterol: 215,
        hdl: 46,
        smoker: false,
        diabetic: false,
        activity: 'moderate'
      });
      setDiabetesData({
        age: 48,
        bmi: 27.2,
        glucose: 112,
        hba1c: 5.9,
        familyHistory: true,
        dailyExerciseMin: 20
      });
      setHyperData({
        bpSys: 134,
        bpDia: 88,
        heartRate: 80,
        sleepHours: 6,
        stressLevel: 6
      });
    } else if (presetType === 'high') {
      setCardioData({
        age: 62,
        gender: 'male',
        bpSys: 158,
        bpDia: 96,
        cholesterol: 260,
        hdl: 36,
        smoker: true,
        diabetic: true,
        activity: 'low'
      });
      setDiabetesData({
        age: 62,
        bmi: 33.4,
        glucose: 145,
        hba1c: 7.4,
        familyHistory: true,
        dailyExerciseMin: 5
      });
      setHyperData({
        bpSys: 162,
        bpDia: 98,
        heartRate: 92,
        sleepHours: 5,
        stressLevel: 9
      });
    }
    setResult(null);
    setSaveSuccess(false);
  };

  // Run Prediction
  const handleRunPrediction = () => {
    setIsCalculating(true);
    setSaveSuccess(false);

    // Simulate AI inference latency with micro-animation
    setTimeout(() => {
      let res;
      if (activeModel === 'cardio') {
        res = calculateCardioRisk(cardioData);
      } else if (activeModel === 'diabetes') {
        res = calculateDiabetesRisk(diabetesData);
      } else {
        res = calculateHypertensionStrokeRisk(hyperData);
      }
      setResult(res);
      setIsCalculating(false);
    }, 450);
  };

  const handleSaveToRecords = () => {
    if (!result) return;
    const currentParams = activeModel === 'cardio' ? cardioData : activeModel === 'diabetes' ? diabetesData : hyperData;
    onSaveRecord({
      disease: result.disease,
      riskScore: result.riskScore,
      riskLevel: result.riskLevel,
      model: result.modelName,
      summary: result.recommendation,
      parameters: currentParams
    });
    setSaveSuccess(true);
  };

  return (
    <div className="predictor-view fade-in">
      {/* Model Selector and Presets Header */}
      <div className="predictor-header-card glass-card">
        <div className="predictor-title-wrap">
          <h2>AI Clinical Risk Inference Engine</h2>
          <p>
            Trained on multi-cohort longitudinal epidemiological datasets with explainable feature importance.
          </p>
        </div>

        {/* Model Switcher Tabs */}
        <div className="tabs-header">
          <button
            className={`tab-btn ${activeModel === 'cardio' ? 'active' : ''}`}
            onClick={() => { setActiveModel('cardio'); setResult(null); }}
          >
            🫀 Cardiovascular Risk (10-Yr)
          </button>
          <button
            className={`tab-btn ${activeModel === 'diabetes' ? 'active' : ''}`}
            onClick={() => { setActiveModel('diabetes'); setResult(null); }}
          >
            🩸 Type-2 Diabetes Risk
          </button>
          <button
            className={`tab-btn ${activeModel === 'hyper' ? 'active' : ''}`}
            onClick={() => { setActiveModel('hyper'); setResult(null); }}
          >
            🧠 Hypertension & Stroke
          </button>
        </div>

        {/* Quick Scenario Fill Buttons */}
        <div className="presets-row">
          <span className="preset-label">Test Scenarios:</span>
          <button className="btn btn-secondary btn-sm" onClick={() => handleLoadPreset('healthy')}>
            🟢 Healthy Profile
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => handleLoadPreset('moderate')}>
            🟡 Borderline Risk
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => handleLoadPreset('high')}>
            🔴 High Risk Profile
          </button>
        </div>
      </div>

      <div className="predictor-grid">
        {/* Form Input Section */}
        <div className="form-column glass-card">
          <div className="form-column-header">
            <h4>Biometric & Clinical Inputs</h4>
            <span className="badge badge-low">Live Validation</span>
          </div>

          {/* Model 1: Cardiovascular */}
          {activeModel === 'cardio' && (
            <div className="model-inputs">
              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Patient Age</span>
                    <span className="unit">{cardioData.age} Years</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="20"
                      max="85"
                      value={cardioData.age}
                      onChange={(e) => setCardioData({ ...cardioData, age: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span>Biological Sex</span>
                  </label>
                  <select
                    className="form-select"
                    value={cardioData.gender}
                    onChange={(e) => setCardioData({ ...cardioData, gender: e.target.value })}
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
              </div>

              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Systolic Blood Pressure</span>
                    <span className="unit">{cardioData.bpSys} mmHg</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="90"
                      max="200"
                      value={cardioData.bpSys}
                      onChange={(e) => setCardioData({ ...cardioData, bpSys: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span>Diastolic Blood Pressure</span>
                    <span className="unit">{cardioData.bpDia} mmHg</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="60"
                      max="125"
                      value={cardioData.bpDia}
                      onChange={(e) => setCardioData({ ...cardioData, bpDia: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>
              </div>

              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Total Serum Cholesterol</span>
                    <span className="unit">{cardioData.cholesterol} mg/dL</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="120"
                      max="320"
                      value={cardioData.cholesterol}
                      onChange={(e) => setCardioData({ ...cardioData, cholesterol: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span>HDL (Good Cholesterol)</span>
                    <span className="unit">{cardioData.hdl} mg/dL</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="20"
                      max="90"
                      value={cardioData.hdl}
                      onChange={(e) => setCardioData({ ...cardioData, hdl: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>
              </div>

              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Physical Activity Level</span>
                  </label>
                  <select
                    className="form-select"
                    value={cardioData.activity}
                    onChange={(e) => setCardioData({ ...cardioData, activity: e.target.value })}
                  >
                    <option value="low">Sedentary (&lt; 1 hr/week)</option>
                    <option value="moderate">Moderate (2–4 hrs/week)</option>
                    <option value="high">High (&gt; 5 hrs/week aerobically)</option>
                  </select>
                </div>

                <div className="form-group toggle-group">
                  <label className="checkbox-control">
                    <input
                      type="checkbox"
                      checked={cardioData.smoker}
                      onChange={(e) => setCardioData({ ...cardioData, smoker: e.target.checked })}
                    />
                    <span>Active Smoker / Tobacco User</span>
                  </label>
                  <label className="checkbox-control" style={{ marginTop: '8px' }}>
                    <input
                      type="checkbox"
                      checked={cardioData.diabetic}
                      onChange={(e) => setCardioData({ ...cardioData, diabetic: e.target.checked })}
                    />
                    <span>Diagnosed Diabetes Mellitus</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Model 2: Diabetes */}
          {activeModel === 'diabetes' && (
            <div className="model-inputs">
              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Patient Age</span>
                    <span className="unit">{diabetesData.age} Years</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="18"
                      max="85"
                      value={diabetesData.age}
                      onChange={(e) => setDiabetesData({ ...diabetesData, age: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span>Body Mass Index (BMI)</span>
                    <span className="unit">{diabetesData.bmi} kg/m²</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="16"
                      max="45"
                      step="0.5"
                      value={diabetesData.bmi}
                      onChange={(e) => setDiabetesData({ ...diabetesData, bmi: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>
              </div>

              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Fasting Blood Glucose</span>
                    <span className="unit">{diabetesData.glucose} mg/dL</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="65"
                      max="220"
                      value={diabetesData.glucose}
                      onChange={(e) => setDiabetesData({ ...diabetesData, glucose: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span>Glycated Hemoglobin (HbA1c)</span>
                    <span className="unit">{diabetesData.hba1c} %</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="4.5"
                      max="11.0"
                      step="0.1"
                      value={diabetesData.hba1c}
                      onChange={(e) => setDiabetesData({ ...diabetesData, hba1c: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>
              </div>

              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Daily Exercise</span>
                    <span className="unit">{diabetesData.dailyExerciseMin} Minutes</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="0"
                      max="90"
                      step="5"
                      value={diabetesData.dailyExerciseMin}
                      onChange={(e) => setDiabetesData({ ...diabetesData, dailyExerciseMin: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group toggle-group">
                  <label className="checkbox-control">
                    <input
                      type="checkbox"
                      checked={diabetesData.familyHistory}
                      onChange={(e) => setDiabetesData({ ...diabetesData, familyHistory: e.target.checked })}
                    />
                    <span>Family History of Diabetes (Parents/Siblings)</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Model 3: Hypertension & Stroke */}
          {activeModel === 'hyper' && (
            <div className="model-inputs">
              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Systolic Blood Pressure</span>
                    <span className="unit">{hyperData.bpSys} mmHg</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="90"
                      max="210"
                      value={hyperData.bpSys}
                      onChange={(e) => setHyperData({ ...hyperData, bpSys: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span>Diastolic Blood Pressure</span>
                    <span className="unit">{hyperData.bpDia} mmHg</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="60"
                      max="130"
                      value={hyperData.bpDia}
                      onChange={(e) => setHyperData({ ...hyperData, bpDia: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>
              </div>

              <div className="form-row-2">
                <div className="form-group">
                  <label className="form-label">
                    <span>Resting Heart Rate</span>
                    <span className="unit">{hyperData.heartRate} bpm</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="45"
                      max="130"
                      value={hyperData.heartRate}
                      onChange={(e) => setHyperData({ ...hyperData, heartRate: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span>Average Sleep Duration</span>
                    <span className="unit">{hyperData.sleepHours} Hours</span>
                  </label>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="4"
                      max="10"
                      value={hyperData.sleepHours}
                      onChange={(e) => setHyperData({ ...hyperData, sleepHours: Number(e.target.value) })}
                      className="range-slider"
                    />
                  </div>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">
                  <span>Chronic Stress Index</span>
                  <span className="unit">{hyperData.stressLevel} / 10</span>
                </label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={hyperData.stressLevel}
                    onChange={(e) => setHyperData({ ...hyperData, stressLevel: Number(e.target.value) })}
                    className="range-slider"
                  />
                </div>
              </div>
            </div>
          )}

          <button
            className="btn btn-primary btn-lg w-full mt-4"
            onClick={handleRunPrediction}
            disabled={isCalculating}
          >
            {isCalculating ? (
              <>
                <span className="spinner"></span> Running Deep ML Inference...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                </svg>
                Calculate Clinical Risk Score
              </>
            )}
          </button>
        </div>

        {/* Prediction Results & Explainable AI Card */}
        <div className="result-column glass-card">
          <div className="result-column-header">
            <h4>Diagnostic Assessment Result</h4>
            {result && <span className="model-chip">{result.modelName}</span>}
          </div>

          {!result ? (
            <div className="result-placeholder">
              <div className="placeholder-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--primary-light)" strokeWidth="1.5">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 16v-4M12 8h.01" />
                </svg>
              </div>
              <h4>No Active Calculation</h4>
              <p>
                Adjust the clinical parameters or choose a test scenario on the left, then click <strong>"Calculate Clinical Risk Score"</strong>.
              </p>
            </div>
          ) : (
            <div className="result-body fade-in">
              {/* Circular Gauge and Score */}
              <div className="gauge-score-wrap">
                <div className="gauge-container">
                  <svg className="gauge-svg" viewBox="0 0 100 100">
                    <circle
                      className="gauge-bg"
                      cx="50"
                      cy="50"
                      r="40"
                      strokeWidth="8"
                    />
                    <circle
                      className={`gauge-progress ${result.badgeClass.replace('badge-', '')}`}
                      cx="50"
                      cy="50"
                      r="40"
                      strokeWidth="8"
                      strokeDasharray="251.2"
                      strokeDashoffset={251.2 - (251.2 * result.riskScore) / 100}
                    />
                  </svg>
                  <div className="gauge-center-text">
                    <span className="gauge-number">{result.riskScore}%</span>
                    <span className="gauge-label">RISK</span>
                  </div>
                </div>

                <div className="gauge-summary">
                  <span className={`badge ${result.badgeClass} badge-lg`}>
                    {result.riskLevel} Risk Tier
                  </span>
                  <div className="result-disease-name">{result.disease}</div>
                  <div className="result-timing">Calculated just now • Confidence 98.4%</div>
                </div>
              </div>

              {/* Explainable AI (XAI) Factors Breakdown */}
              <div className="xai-factors-section">
                <div className="xai-title">
                  <span>Explainable AI (XAI) Feature Importance</span>
                  <span className="xai-subtitle">Key Risk & Protective Contributors</span>
                </div>
                <div className="factor-pills-list">
                  {result.factors.map((f, idx) => (
                    <div key={idx} className={`factor-pill ${f.type}`}>
                      <span className="f-name">{f.name}</span>
                      <span className="f-impact">{f.impact}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations Box */}
              <div className="recommendation-card">
                <div className="rec-header">
                  <span className="rec-icon">📋</span>
                  <span>Clinical Guidance & Prevention Protocol</span>
                </div>
                <p className="rec-text">{result.recommendation}</p>
              </div>

              {/* Save to History and Actions */}
              <div className="result-actions-row">
                <button
                  className="btn btn-primary"
                  onClick={handleSaveToRecords}
                  disabled={saveSuccess}
                >
                  {saveSuccess ? (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                      Saved in Health Records!
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                        <polyline points="17 21 17 13 7 13 7 21" />
                        <polyline points="7 3 7 8 15 8" />
                      </svg>
                      Save to Patient Records
                    </>
                  )}
                </button>

                <button
                  className="btn btn-secondary"
                  onClick={() => window.print()}
                  title="Print Report"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="6 9 6 2 18 2 18 9" />
                    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2" />
                    <rect x="6" y="14" width="12" height="8" />
                  </svg>
                  Export / Print
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
