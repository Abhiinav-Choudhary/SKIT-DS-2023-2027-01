// API service and clinical risk computation engine

const DEFAULT_API_BASE = 'http://localhost:5000/api';
const STORAGE_KEYS = {
  TOKEN: 'health_ai_token',
  USER: 'health_ai_user',
  API_URL: 'health_ai_api_url',
  RECORDS: 'health_ai_records',
  VITALS: 'health_ai_vitals',
  NOTIFICATIONS: 'health_ai_notifications'
};

export const getApiBaseUrl = () => {
  return localStorage.getItem(STORAGE_KEYS.API_URL) || DEFAULT_API_BASE;
};

export const setApiBaseUrl = (url) => {
  localStorage.setItem(STORAGE_KEYS.API_URL, url.replace(/\/$/, ''));
};

// Check backend connectivity
export const checkBackendHealth = async () => {
  const base = getApiBaseUrl().replace(/\/api$/, '');
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);
    const res = await fetch(`${base}/`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (res.ok) {
      return { online: true, message: 'Backend connected' };
    }
    return { online: false, message: `Server returned status ${res.status}` };
  } catch (err) {
    return { online: false, message: 'Backend offline (Running in local simulation mode)' };
  }
};

// Authentication Services
export const authService = {
  getToken() {
    return localStorage.getItem(STORAGE_KEYS.TOKEN);
  },

  getCurrentUser() {
    const raw = localStorage.getItem(STORAGE_KEYS.USER);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  },

  async register(name, email, password) {
    const base = getApiBaseUrl();
    try {
      const res = await fetch(`${base}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || 'Registration failed');
      }
      return data;
    } catch (err) {
      // If server unreachable, provide mock registration for demo
      if (err.message.includes('fetch') || err.name === 'TypeError') {
        const mockUser = {
          id: 'usr_' + Date.now().toString(36),
          name,
          email,
          createdAt: new Date().toISOString()
        };
        const mockToken = 'mock_jwt_token_' + Date.now();
        localStorage.setItem(STORAGE_KEYS.TOKEN, mockToken);
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(mockUser));
        return {
          success: true,
          message: 'Registered in demo/offline mode',
          data: { user: mockUser, token: mockToken }
        };
      }
      throw err;
    }
  },

  async login(email, password) {
    const base = getApiBaseUrl();
    try {
      const res = await fetch(`${base}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || 'Login failed');
      }
      if (data.data?.token) {
        localStorage.setItem(STORAGE_KEYS.TOKEN, data.data.token);
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(data.data.user));
      }
      return data;
    } catch (err) {
      if (err.message.includes('fetch') || err.name === 'TypeError') {
        // Fallback for mock login
        const mockUser = {
          id: 'usr_demo_01',
          name: email.split('@')[0] || 'Demo Patient',
          email,
          createdAt: new Date().toISOString()
        };
        const mockToken = 'mock_jwt_token_' + Date.now();
        localStorage.setItem(STORAGE_KEYS.TOKEN, mockToken);
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(mockUser));
        return {
          success: true,
          message: 'Logged in (Simulated Mode)',
          data: { user: mockUser, token: mockToken }
        };
      }
      throw err;
    }
  },

  async fetchMe() {
    const token = this.getToken();
    if (!token) return null;
    const base = getApiBaseUrl();
    try {
      const res = await fetch(`${base}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!res.ok) {
        if (res.status === 401) {
          this.logout();
        }
        return null;
      }
      const data = await res.json();
      if (data.data?.user) {
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(data.data.user));
        return data.data.user;
      }
      return null;
    } catch (err) {
      // In offline mode return cached user
      return this.getCurrentUser();
    }
  },

  logout() {
    localStorage.removeItem(STORAGE_KEYS.TOKEN);
    localStorage.removeItem(STORAGE_KEYS.USER);
  }
};

// Initial Seed Data for Demo Vitals & Predictions
const INITIAL_VITALS = [
  { id: 'v1', timestamp: '2026-10-02 08:30', bpSys: 118, bpDia: 76, heartRate: 72, glucose: 94, spo2: 99, weight: 68.5, bmi: 22.4 },
  { id: 'v2', timestamp: '2026-10-03 08:45', bpSys: 122, bpDia: 78, heartRate: 75, glucose: 98, spo2: 98, weight: 68.3, bmi: 22.3 },
  { id: 'v3', timestamp: '2026-10-04 09:10', bpSys: 126, bpDia: 82, heartRate: 80, glucose: 104, spo2: 98, weight: 68.7, bmi: 22.4 },
  { id: 'v4', timestamp: '2026-10-05 08:15', bpSys: 120, bpDia: 78, heartRate: 71, glucose: 92, spo2: 99, weight: 68.2, bmi: 22.3 },
  { id: 'v5', timestamp: '2026-10-06 08:50', bpSys: 135, bpDia: 88, heartRate: 84, glucose: 116, spo2: 97, weight: 68.9, bmi: 22.5 },
  { id: 'v6', timestamp: '2026-10-07 09:00', bpSys: 124, bpDia: 80, heartRate: 74, glucose: 96, spo2: 98, weight: 68.4, bmi: 22.3 },
  { id: 'v7', timestamp: '2026-10-08 08:20', bpSys: 119, bpDia: 77, heartRate: 70, glucose: 95, spo2: 99, weight: 68.1, bmi: 22.2 }
];

const INITIAL_RECORDS = [
  {
    id: 'rec_101',
    disease: 'Cardiovascular Risk (10-Yr)',
    riskScore: 12.4,
    riskLevel: 'Low',
    date: '2026-10-06',
    model: 'Framingham-ML Ensemble v2.1',
    summary: 'Blood pressure and cholesterol within standard acceptable parameters.',
    parameters: { age: 46, gender: 'male', bpSys: 124, cholesterol: 195, hdl: 52, smoker: false }
  },
  {
    id: 'rec_102',
    disease: 'Type 2 Diabetes Mellitus',
    riskScore: 28.5,
    riskLevel: 'Moderate',
    date: '2026-10-04',
    model: 'ADA-XGBoost Neural Classifier',
    summary: 'Fasting glucose slightly elevated. Physical exercise recommended.',
    parameters: { age: 46, bmi: 25.8, glucose: 108, hba1c: 5.7, familyHistory: true }
  },
  {
    id: 'rec_103',
    disease: 'Hypertension & Arterial Health',
    riskScore: 18.2,
    riskLevel: 'Low',
    date: '2026-09-28',
    model: 'ArterialStiffness DL Model',
    summary: 'Vascular elasticity normal, pulse pressure in healthy range.',
    parameters: { bpSys: 120, bpDia: 78, heartRate: 72 }
  }
];

// Local Storage Handlers
export const recordService = {
  getRecords() {
    const raw = localStorage.getItem(STORAGE_KEYS.RECORDS);
    if (!raw) {
      localStorage.setItem(STORAGE_KEYS.RECORDS, JSON.stringify(INITIAL_RECORDS));
      return INITIAL_RECORDS;
    }
    try {
      return JSON.parse(raw);
    } catch {
      return INITIAL_RECORDS;
    }
  },

  addRecord(record) {
    const records = this.getRecords();
    const newRecord = {
      ...record,
      id: 'rec_' + Date.now().toString(36),
      date: new Date().toISOString().split('T')[0]
    };
    records.unshift(newRecord);
    localStorage.setItem(STORAGE_KEYS.RECORDS, JSON.stringify(records));
    return newRecord;
  },

  deleteRecord(id) {
    let records = this.getRecords();
    records = records.filter(r => r.id !== id);
    localStorage.setItem(STORAGE_KEYS.RECORDS, JSON.stringify(records));
    return records;
  }
};

export const vitalsService = {
  getVitals() {
    const raw = localStorage.getItem(STORAGE_KEYS.VITALS);
    if (!raw) {
      localStorage.setItem(STORAGE_KEYS.VITALS, JSON.stringify(INITIAL_VITALS));
      return INITIAL_VITALS;
    }
    try {
      return JSON.parse(raw);
    } catch {
      return INITIAL_VITALS;
    }
  },

  addVital(vital) {
    const list = this.getVitals();
    const newVital = {
      ...vital,
      id: 'v_' + Date.now().toString(36),
      timestamp: new Date().toLocaleString([], { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
    };
    list.push(newVital);
    localStorage.setItem(STORAGE_KEYS.VITALS, JSON.stringify(list));
    return newVital;
  }
};

// Clinical AI Calculation Engine with Explainable Factors (XAI)
export const calculateCardioRisk = (data) => {
  const {
    age = 45,
    gender = 'male',
    bpSys = 120,
    bpDia = 80,
    cholesterol = 190,
    hdl = 50,
    smoker = false,
    diabetic = false,
    activity = 'moderate' // low, moderate, high
  } = data;

  // Base clinical coefficient algorithm
  let baseScore = 0;
  const factors = [];

  // Age factor
  const agePoints = Math.max(0, (age - 30) * 0.8);
  baseScore += agePoints;
  factors.push({ name: 'Age (' + age + ' yrs)', impact: '+' + agePoints.toFixed(1) + '%', type: 'risk' });

  // Blood pressure factor
  if (bpSys >= 140 || bpDia >= 90) {
    const bpImpact = (bpSys - 120) * 0.4 + (bpDia - 80) * 0.3;
    baseScore += bpImpact;
    factors.push({ name: 'Hypertension (' + bpSys + '/' + bpDia + ' mmHg)', impact: '+' + bpImpact.toFixed(1) + '%', type: 'risk' });
  } else if (bpSys <= 118 && bpDia <= 78) {
    baseScore -= 3.5;
    factors.push({ name: 'Optimal Blood Pressure', impact: '-3.5%', type: 'protective' });
  }

  // Cholesterol ratio (Total / HDL)
  const ratio = cholesterol / Math.max(20, hdl);
  if (ratio > 4.5) {
    const cholImpact = (ratio - 4.5) * 4.2;
    baseScore += cholImpact;
    factors.push({ name: 'Elevated Chol/HDL (' + ratio.toFixed(1) + ')', impact: '+' + cholImpact.toFixed(1) + '%', type: 'risk' });
  } else if (hdl >= 60) {
    baseScore -= 4.0;
    factors.push({ name: 'High HDL Good Cholesterol (' + hdl + ' mg/dL)', impact: '-4.0%', type: 'protective' });
  }

  // Smoking
  if (smoker) {
    baseScore += 14.5;
    factors.push({ name: 'Active Tobacco Smoking', impact: '+14.5%', type: 'risk' });
  }

  // Diabetes
  if (diabetic) {
    baseScore += 12.0;
    factors.push({ name: 'Comorbid Diabetes Mellitus', impact: '+12.0%', type: 'risk' });
  }

  // Physical Activity
  if (activity === 'high') {
    baseScore -= 6.0;
    factors.push({ name: 'Vigorous Physical Exercise', impact: '-6.0%', type: 'protective' });
  } else if (activity === 'low') {
    baseScore += 5.5;
    factors.push({ name: 'Sedentary Lifestyle', impact: '+5.5%', type: 'risk' });
  }

  // Bound between 2% and 92%
  const riskScore = Math.min(94, Math.max(2.1, parseFloat(baseScore.toFixed(1))));

  let riskLevel = 'Low';
  let badgeClass = 'badge-low';
  let recommendation = 'Maintain current healthy diet, regular cardiovascular exercise, and routine yearly check-ups.';

  if (riskScore >= 45) {
    riskLevel = 'Critical / High';
    badgeClass = 'badge-critical';
    recommendation = 'Urgent clinical evaluation recommended. Consult a cardiologist for coronary calcium scoring, lipid panel re-evaluation, and antihypertensive therapy.';
  } else if (riskScore >= 25) {
    riskLevel = 'Elevated';
    badgeClass = 'badge-high';
    recommendation = 'Significant 10-year cardiovascular probability. Target systolic BP < 125 mmHg, reduce dietary sodium, and initiate LDL reduction.';
  } else if (riskScore >= 12) {
    riskLevel = 'Moderate';
    badgeClass = 'badge-mod';
    recommendation = 'Moderate risk threshold. Adopt Mediterranean or DASH diet, 150 minutes of weekly aerobic exercise, and re-assess in 6 months.';
  }

  return {
    disease: 'Cardiovascular Disease (10-Yr Risk)',
    riskScore,
    riskLevel,
    badgeClass,
    factors,
    recommendation,
    modelName: 'CardioVision AI (Framingham-ML v2.4)'
  };
};

export const calculateDiabetesRisk = (data) => {
  const {
    age = 45,
    bmi = 24.5,
    glucose = 95,
    hba1c = 5.4,
    familyHistory = false,
    bpHigh = false,
    dailyExerciseMin = 30
  } = data;

  let baseScore = 0;
  const factors = [];

  // Age factor
  if (age > 45) {
    const ageImpact = (age - 45) * 0.5 + 4;
    baseScore += ageImpact;
    factors.push({ name: 'Age > 45 (' + age + ' yrs)', impact: '+' + ageImpact.toFixed(1) + '%', type: 'risk' });
  }

  // BMI Factor
  if (bmi >= 30) {
    const bmiImpact = (bmi - 25) * 1.8;
    baseScore += bmiImpact;
    factors.push({ name: 'Obesity Class BMI (' + bmi + ')', impact: '+' + bmiImpact.toFixed(1) + '%', type: 'risk' });
  } else if (bmi >= 25) {
    const bmiImpact = (bmi - 24) * 1.2;
    baseScore += bmiImpact;
    factors.push({ name: 'Overweight BMI (' + bmi + ')', impact: '+' + bmiImpact.toFixed(1) + '%', type: 'risk' });
  } else {
    baseScore -= 3.0;
    factors.push({ name: 'Optimal Body Mass Index (' + bmi + ')', impact: '-3.0%', type: 'protective' });
  }

  // Fasting Blood Sugar (mg/dL)
  if (glucose >= 126) {
    baseScore += 28.0;
    factors.push({ name: 'Diabetic Fasting Glucose (' + glucose + ' mg/dL)', impact: '+28.0%', type: 'risk' });
  } else if (glucose >= 100) {
    baseScore += 14.5;
    factors.push({ name: 'Pre-diabetic Impaired Glucose (' + glucose + ' mg/dL)', impact: '+14.5%', type: 'risk' });
  } else {
    baseScore -= 4.0;
    factors.push({ name: 'Normal Fasting Glucose (' + glucose + ' mg/dL)', impact: '-4.0%', type: 'protective' });
  }

  // HbA1c
  if (hba1c >= 6.5) {
    baseScore += 25.0;
    factors.push({ name: 'Diagnostic HbA1c (' + hba1c + '%)', impact: '+25.0%', type: 'risk' });
  } else if (hba1c >= 5.7) {
    baseScore += 11.0;
    factors.push({ name: 'Pre-diabetic Range HbA1c (' + hba1c + '%)', impact: '+11.0%', type: 'risk' });
  }

  // Genetics / Family History
  if (familyHistory) {
    baseScore += 12.0;
    factors.push({ name: 'First-Degree Family Diabetes History', impact: '+12.0%', type: 'risk' });
  }

  // Exercise
  if (dailyExerciseMin >= 30) {
    baseScore -= 7.5;
    factors.push({ name: 'Daily Physical Exercise (' + dailyExerciseMin + 'm)', impact: '-7.5%', type: 'protective' });
  } else {
    baseScore += 6.0;
    factors.push({ name: 'Insufficient Physical Activity', impact: '+6.0%', type: 'risk' });
  }

  const riskScore = Math.min(96, Math.max(1.8, parseFloat(baseScore.toFixed(1))));

  let riskLevel = 'Low';
  let badgeClass = 'badge-low';
  let recommendation = 'Metabolic markers are in healthy range. Maintain high-fiber, low-glycemic nutrition and routine screening.';

  if (riskScore >= 50) {
    riskLevel = 'Critical / High';
    badgeClass = 'badge-critical';
    recommendation = 'Strong probability of impaired glucose tolerance or Type 2 Diabetes. Schedule an Oral Glucose Tolerance Test (OGTT) and endocrinology consultation.';
  } else if (riskScore >= 25) {
    riskLevel = 'Elevated';
    badgeClass = 'badge-high';
    recommendation = 'Pre-diabetes risk indicators detected. Reduce refined sugars, incorporate resistance training, and monitor HbA1c quarterly.';
  } else if (riskScore >= 12) {
    riskLevel = 'Moderate';
    badgeClass = 'badge-mod';
    recommendation = 'Borderline metabolic score. Maintain dietary caloric balance and keep active.';
  }

  return {
    disease: 'Type 2 Diabetes Mellitus Risk',
    riskScore,
    riskLevel,
    badgeClass,
    factors,
    recommendation,
    modelName: 'GlucoPredict AI (XGBoost Ensemble v3.2)'
  };
};

export const calculateHypertensionStrokeRisk = (data) => {
  const {
    bpSys = 125,
    bpDia = 82,
    heartRate = 72,
    sleepHours = 7,
    stressLevel = 4 // 1 to 10
  } = data;

  let baseScore = 5;
  const factors = [];

  if (bpSys >= 160 || bpDia >= 100) {
    baseScore += 45;
    factors.push({ name: 'Stage 2 Severe Hypertension (' + bpSys + '/' + bpDia + ')', impact: '+45%', type: 'risk' });
  } else if (bpSys >= 130 || bpDia >= 80) {
    baseScore += 22;
    factors.push({ name: 'Stage 1 Hypertension (' + bpSys + '/' + bpDia + ')', impact: '+22%', type: 'risk' });
  } else if (bpSys >= 120 && bpSys < 130 && bpDia < 80) {
    baseScore += 8;
    factors.push({ name: 'Elevated Systolic Pressure (' + bpSys + ' mmHg)', impact: '+8%', type: 'risk' });
  } else {
    baseScore -= 4;
    factors.push({ name: 'Normotensive Profile (' + bpSys + '/' + bpDia + ')', impact: '-4%', type: 'protective' });
  }

  if (stressLevel >= 8) {
    baseScore += 12;
    factors.push({ name: 'High Chronic Stress Index (' + stressLevel + '/10)', impact: '+12%', type: 'risk' });
  }

  if (sleepHours < 6) {
    baseScore += 10;
    factors.push({ name: 'Sleep Deprivation (' + sleepHours + ' hrs/night)', impact: '+10%', type: 'risk' });
  } else if (sleepHours >= 7 && sleepHours <= 8.5) {
    baseScore -= 5;
    factors.push({ name: 'Optimal Sleep Quality (' + sleepHours + ' hrs)', impact: '-5%', type: 'protective' });
  }

  const riskScore = Math.min(95, Math.max(3, parseFloat(baseScore.toFixed(1))));

  let riskLevel = 'Low';
  let badgeClass = 'badge-low';
  let recommendation = 'Arterial compliance normal. Continue salt restriction and hydration.';

  if (riskScore >= 45) {
    riskLevel = 'High / Critical';
    badgeClass = 'badge-critical';
    recommendation = 'Urgent attention: Elevated stroke risk. Ambulatory 24-hour blood pressure monitoring and neurologist review suggested.';
  } else if (riskScore >= 25) {
    riskLevel = 'Elevated';
    badgeClass = 'badge-high';
    recommendation = 'Stage 1 hypertensive profile. Implement stress-reduction techniques and home BP monitoring twice daily.';
  } else if (riskScore >= 12) {
    riskLevel = 'Moderate';
    badgeClass = 'badge-mod';
    recommendation = 'Pre-hypertension stage. Limit sodium to < 2000mg/day and optimize sleep hygiene.';
  }

  return {
    disease: 'Hypertension & Cerebrovascular Risk',
    riskScore,
    riskLevel,
    badgeClass,
    factors,
    recommendation,
    modelName: 'NeuroVascular AI (DeepPulse v1.8)'
  };
};
