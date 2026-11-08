const { spawn } = require('child_process');
const path = require('path');
const Prediction = require('../models/Prediction');

// Path to the Python inference bridge script
const BRIDGE_PATH = path.resolve(__dirname, '../ml/inference_bridge.py');

// Python binary detection (prefers python, falls back to py or python3)
const PYTHON_BIN = process.env.PYTHON_BIN || 'python';

/**
 * Execute Python ML inference bridge via subprocess
 */
const runPythonInference = (modelType, inputPayload) => {
  return new Promise((resolve, reject) => {
    const pyProcess = spawn(PYTHON_BIN, [BRIDGE_PATH, modelType], {
      cwd: path.resolve(__dirname, '../../../'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let stdoutData = '';
    let stderrData = '';

    pyProcess.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    pyProcess.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    pyProcess.on('error', (err) => {
      const error = new Error(`Failed to start ML inference service: ${err.message}`);
      error.statusCode = 503;
      reject(error);
    });

    pyProcess.on('close', (code) => {
      // Find the JSON output line from stdout
      const lines = stdoutData.split(/\r?\n/).filter((l) => l.trim().length > 0);
      let parsed = null;

      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith('{') && line.endsWith('}')) {
          try {
            parsed = JSON.parse(line);
            break;
          } catch {
            // continue looking
          }
        }
      }

      if (code !== 0 || !parsed || parsed.success === false) {
        const errMsg = parsed?.error || stderrData.trim() || `Inference process exited with code ${code}`;
        const error = new Error(errMsg);
        // Distinguish validation vs runtime error
        error.statusCode = errMsg.toLowerCase().includes('invalid') || errMsg.toLowerCase().includes('missing') ? 400 : 500;
        return reject(error);
      }

      resolve(parsed.data);
    });

    // Write input payload JSON to stdin
    pyProcess.stdin.write(JSON.stringify(inputPayload));
    pyProcess.stdin.end();
  });
};

const REQUIRED_DIABETES_FIELDS = [
  'HighBP',
  'HighChol',
  'CholCheck',
  'BMI',
  'Smoker',
  'Stroke',
  'HeartDiseaseorAttack',
  'PhysActivity',
  'Fruits',
  'Veggies',
  'HvyAlcoholConsump',
  'AnyHealthcare',
  'NoDocbcCost',
  'GenHlth',
  'MentHlth',
  'PhysHlth',
  'DiffWalk',
  'Sex',
  'Age',
  'Education',
  'Income',
];

const REQUIRED_HEART_FIELDS = [
  'age',
  'sex',
  'cp',
  'trestbps',
  'chol',
  'fbs',
  'restecg',
  'thalach',
  'exang',
  'oldpeak',
  'slope',
  'ca',
  'thal',
];

/**
 * Validate diabetes input feature payload
 */
const validateDiabetesInput = (data) => {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    const error = new Error('Diabetes prediction requires an input data object');
    error.statusCode = 400;
    throw error;
  }

  const missing = REQUIRED_DIABETES_FIELDS.filter((f) => data[f] === undefined || data[f] === null);
  if (missing.length > 0) {
    const error = new Error(`Missing required diabetes input features: ${missing.join(', ')}`);
    error.statusCode = 400;
    throw error;
  }

  // Validate value ranges
  const numBMI = Number(data.BMI);
  if (isNaN(numBMI) || numBMI < 12 || numBMI > 100) {
    const error = new Error('BMI must be a number between 12 and 100');
    error.statusCode = 400;
    throw error;
  }

  const numGenHlth = Number(data.GenHlth);
  if (!Number.isInteger(numGenHlth) || numGenHlth < 1 || numGenHlth > 5) {
    const error = new Error('GenHlth must be an integer between 1 and 5');
    error.statusCode = 400;
    throw error;
  }

  const numAge = Number(data.Age);
  if (!Number.isInteger(numAge) || numAge < 1 || numAge > 13) {
    const error = new Error('Age category must be an integer between 1 and 13');
    error.statusCode = 400;
    throw error;
  }
};

/**
 * Validate heart disease input feature payload
 */
const validateHeartInput = (data) => {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    const error = new Error('Heart disease prediction requires an input data object');
    error.statusCode = 400;
    throw error;
  }

  const missing = REQUIRED_HEART_FIELDS.filter((f) => data[f] === undefined || data[f] === null);
  if (missing.length > 0) {
    const error = new Error(`Missing required heart disease input features: ${missing.join(', ')}`);
    error.statusCode = 400;
    throw error;
  }

  const numAge = Number(data.age);
  if (isNaN(numAge) || numAge < 1 || numAge > 120) {
    const error = new Error('Age must be a valid number between 1 and 120');
    error.statusCode = 400;
    throw error;
  }

  const numTrestbps = Number(data.trestbps);
  if (isNaN(numTrestbps) || numTrestbps <= 0) {
    const error = new Error('Resting blood pressure (trestbps) must be a positive number');
    error.statusCode = 400;
    throw error;
  }

  const numChol = Number(data.chol);
  if (isNaN(numChol) || numChol <= 0) {
    const error = new Error('Serum cholesterol (chol) must be a positive number');
    error.statusCode = 400;
    throw error;
  }
};

/**
 * Run Diabetes Risk prediction and persist results
 */
const predictDiabetes = async (userId, inputData) => {
  validateDiabetesInput(inputData);

  const inferenceResult = await runPythonInference('diabetes', inputData);

  const savedPrediction = await Prediction.create({
    user: userId,
    modelType: 'diabetes',
    inputs: inputData,
    predictedClass: inferenceResult.predictedClass,
    riskScore: inferenceResult.riskScore,
    riskLevel: inferenceResult.riskLevel,
    probability: inferenceResult.probability,
    recommendation: inferenceResult.recommendation,
    modelName: inferenceResult.modelName,
    modelVersion: inferenceResult.modelVersion,
  });

  return savedPrediction;
};

/**
 * Run Heart Disease Risk prediction and persist results
 */
const predictHeartDisease = async (userId, inputData) => {
  validateHeartInput(inputData);

  const inferenceResult = await runPythonInference('heart', inputData);

  const savedPrediction = await Prediction.create({
    user: userId,
    modelType: 'heart',
    inputs: inputData,
    predictedClass: inferenceResult.predictedClass,
    riskScore: inferenceResult.riskScore,
    riskLevel: inferenceResult.riskLevel,
    probabilities: inferenceResult.probabilities,
    recommendation: inferenceResult.recommendation,
    modelName: inferenceResult.modelName,
    modelVersion: inferenceResult.modelVersion,
  });

  return savedPrediction;
};

/**
 * List predictions for authenticated user with pagination and optional type filter
 */
const listPredictions = async (userId, query = {}) => {
  const page = Number(query.page ?? 1);
  const limit = Number(query.limit ?? 20);

  if (!Number.isInteger(page) || page < 1 || !Number.isInteger(limit) || limit < 1 || limit > 100) {
    const error = new Error('Page must be a positive integer and limit must be between 1 and 100');
    error.statusCode = 400;
    throw error;
  }

  const filter = { user: userId };
  if (query.modelType) {
    if (!['diabetes', 'heart'].includes(query.modelType)) {
      const error = new Error('Invalid modelType filter. Allowed values: diabetes, heart');
      error.statusCode = 400;
      throw error;
    }
    filter.modelType = query.modelType;
  }

  const [predictions, total] = await Promise.all([
    Prediction.find(filter)
      .sort({ createdAt: -1 })
      .skip((page - 1) * limit)
      .limit(limit),
    Prediction.countDocuments(filter),
  ]);

  return {
    predictions,
    page,
    limit,
    pagination: {
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit) || 1,
    },
  };
};

/**
 * Retrieve single prediction by ID scoped to authenticated user
 */
const getPredictionById = async (userId, predictionId) => {
  const prediction = await Prediction.findOne({ _id: predictionId, user: userId });
  if (!prediction) {
    const error = new Error('Prediction record not found');
    error.statusCode = 404;
    throw error;
  }
  return prediction;
};

/**
 * Delete prediction by ID scoped to authenticated user
 */
const deletePrediction = async (userId, predictionId) => {
  const prediction = await Prediction.findOneAndDelete({ _id: predictionId, user: userId });
  if (!prediction) {
    const error = new Error('Prediction record not found');
    error.statusCode = 404;
    throw error;
  }
  return prediction;
};

module.exports = {
  runPythonInference,
  predictDiabetes,
  predictHeartDisease,
  listPredictions,
  getPredictionById,
  deletePrediction,
};

