const { test, describe, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const request = require('supertest');
const bcrypt = require('bcryptjs');

process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test_super_secret_jwt_key_for_testing_1234567890';
process.env.JWT_EXPIRES_IN = '1h';

const app = require('../src/app');
const User = require('../src/models/User');
const HealthRecord = require('../src/models/HealthRecord');
const Prediction = require('../src/models/Prediction');
const { generateToken } = require('../src/utils/jwt');

// In-memory data stores for testing
let inMemoryUsers = [];
let inMemoryHealthRecords = [];
let inMemoryPredictions = [];
let userCounter = 1;
let predCounter = 1;

// Setup in-memory User mock methods
User.findById = (id) => {
  const found = inMemoryUsers.find((u) => u._id === id);
  if (!found) {
    return {
      select: () => Promise.resolve(null),
      then: (res) => Promise.resolve(res(null)),
    };
  }

  const doc = {
    ...found,
    id: found._id,
    comparePassword: async (pwd) => bcrypt.compare(pwd, found.password),
    save: async function () {
      const idx = inMemoryUsers.findIndex((u) => u._id === found._id);
      if (idx !== -1) {
        if (this.password && this.password !== found.password) {
          const salt = await bcrypt.genSalt(10);
          inMemoryUsers[idx].password = await bcrypt.hash(this.password, salt);
        }
        inMemoryUsers[idx].updatedAt = new Date();
      }
      return this;
    },
    toJSON() {
      const copy = { ...this };
      delete copy.password;
      return copy;
    },
  };

  return {
    select: (fields) => {
      return Promise.resolve(doc);
    },
    then: (resolve) => Promise.resolve(resolve(doc)),
  };
};

User.findByIdAndDelete = async (id) => {
  const idx = inMemoryUsers.findIndex((u) => u._id === id);
  if (idx === -1) return null;
  return inMemoryUsers.splice(idx, 1)[0];
};

// Setup in-memory HealthRecord mock methods
HealthRecord.deleteMany = async (query) => {
  inMemoryHealthRecords = inMemoryHealthRecords.filter((r) => r.user !== query.user);
  return { acknowledged: true };
};

// Setup in-memory Prediction mock methods
Prediction.create = async (data) => {
  const pred = {
    _id: `pred_${predCounter++}`,
    user: data.user,
    modelType: data.modelType,
    inputs: data.inputs,
    predictedClass: data.predictedClass,
    riskScore: data.riskScore,
    riskLevel: data.riskLevel,
    probability: data.probability,
    probabilities: data.probabilities,
    recommendation: data.recommendation,
    modelName: data.modelName,
    modelVersion: data.modelVersion,
    recordedAt: data.recordedAt || new Date(),
    createdAt: new Date(),
    updatedAt: new Date(),
  };
  pred.id = pred._id;
  inMemoryPredictions.push(pred);
  return pred;
};

Prediction.find = (query) => {
  let matched = inMemoryPredictions.filter((p) => {
    if (query.user && p.user !== query.user) return false;
    if (query.modelType && p.modelType !== query.modelType) return false;
    return true;
  });

  const queryObj = {
    _data: matched,
    sort(sortObj) {
      if (sortObj && sortObj.createdAt === -1) {
        this._data.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      }
      return this;
    },
    skip(count) {
      this._data = this._data.slice(count);
      return this;
    },
    limit(count) {
      this._data = this._data.slice(0, count);
      return Promise.resolve(this._data);
    },
    then(resolve) {
      return Promise.resolve(resolve(this._data));
    },
  };

  return queryObj;
};

Prediction.countDocuments = async (query) => {
  return inMemoryPredictions.filter((p) => {
    if (query.user && p.user !== query.user) return false;
    if (query.modelType && p.modelType !== query.modelType) return false;
    return true;
  }).length;
};

Prediction.findOne = async (query) => {
  const found = inMemoryPredictions.find((p) => {
    if (query._id && p._id !== query._id) return false;
    if (query.user && p.user !== query.user) return false;
    return true;
  });
  return found || null;
};

Prediction.findOneAndDelete = async (query) => {
  const idx = inMemoryPredictions.findIndex((p) => {
    if (query._id && p._id !== query._id) return false;
    if (query.user && p.user !== query.user) return false;
    return true;
  });
  if (idx === -1) return null;
  return inMemoryPredictions.splice(idx, 1)[0];
};

Prediction.deleteMany = async (query) => {
  inMemoryPredictions = inMemoryPredictions.filter((p) => p.user !== query.user);
  return { acknowledged: true };
};

describe('AI Prediction API Integration Suite', () => {
  let userToken;
  let userId;
  let otherUserToken;
  let otherUserId;

  const validDiabetesInput = {
    HighBP: 1,
    HighChol: 1,
    CholCheck: 1,
    BMI: 28.5,
    Smoker: 0,
    Stroke: 0,
    HeartDiseaseorAttack: 0,
    PhysActivity: 1,
    Fruits: 1,
    Veggies: 1,
    HvyAlcoholConsump: 0,
    AnyHealthcare: 1,
    NoDocbcCost: 0,
    GenHlth: 3,
    MentHlth: 0,
    PhysHlth: 2,
    DiffWalk: 0,
    Sex: 1,
    Age: 8,
    Education: 5,
    Income: 6,
  };

  const validHeartInput = {
    age: 58,
    sex: 1,
    cp: 2,
    trestbps: 135,
    chol: 245,
    fbs: 0,
    restecg: 1,
    thalach: 140,
    exang: 0,
    oldpeak: 1.2,
    slope: 2,
    ca: 1,
    thal: 3,
  };

  beforeEach(async () => {
    inMemoryUsers = [];
    inMemoryHealthRecords = [];
    inMemoryPredictions = [];
    userCounter = 1;
    predCounter = 1;

    // Create primary user
    const salt = await bcrypt.genSalt(10);
    const hash = await bcrypt.hash('secretPass123', salt);
    userId = `usr_${userCounter++}`;
    inMemoryUsers.push({
      _id: userId,
      name: 'Primary Test Patient',
      email: 'primary@test.com',
      password: hash,
      createdAt: new Date(),
      updatedAt: new Date(),
    });
    userToken = generateToken({ id: userId });

    // Create secondary user for isolation tests
    otherUserId = `usr_${userCounter++}`;
    inMemoryUsers.push({
      _id: otherUserId,
      name: 'Secondary Test Patient',
      email: 'secondary@test.com',
      password: hash,
      createdAt: new Date(),
      updatedAt: new Date(),
    });
    otherUserToken = generateToken({ id: otherUserId });
  });

  // 1. Authentication check
  test('1. Rejects unauthenticated prediction requests with 401', async () => {
    const res = await request(app)
      .post('/api/predictions/diabetes')
      .send(validDiabetesInput);

    assert.equal(res.status, 401);
    assert.equal(res.body.success, false);
  });

  // 2. Diabetes input validation
  test('2. Rejects invalid diabetes input with 400 Bad Request', async () => {
    const invalidInput = { ...validDiabetesInput, BMI: 5 }; // BMI < 12 is invalid
    const res = await request(app)
      .post('/api/predictions/diabetes')
      .set('Authorization', `Bearer ${userToken}`)
      .send(invalidInput);

    assert.equal(res.status, 400);
    assert.equal(res.body.success, false);
    assert.match(res.body.message, /BMI/i);
  });

  // 3. Heart input validation
  test('3. Rejects invalid heart disease input with 400 Bad Request', async () => {
    const invalidInput = { ...validHeartInput, trestbps: -10 }; // negative BP
    const res = await request(app)
      .post('/api/predictions/heart')
      .set('Authorization', `Bearer ${userToken}`)
      .send(invalidInput);

    assert.equal(res.status, 400);
    assert.equal(res.body.success, false);
    assert.match(res.body.message, /trestbps/i);
  });

  // 4. Diabetes live prediction and persistence
  test('4. POST /api/predictions/diabetes performs inference and persists record', async () => {
    const res = await request(app)
      .post('/api/predictions/diabetes')
      .set('Authorization', `Bearer ${userToken}`)
      .send(validDiabetesInput);

    assert.equal(res.status, 201);
    assert.equal(res.body.success, true);
    assert.ok(res.body.data.prediction);
    assert.equal(res.body.data.prediction.user, userId);
    assert.equal(res.body.data.prediction.modelType, 'diabetes');
    assert.ok(typeof res.body.data.prediction.riskScore === 'number');
    assert.ok(['Low', 'Moderate', 'High', 'Critical'].includes(res.body.data.prediction.riskLevel));
    assert.ok(res.body.data.prediction.recommendation);
    assert.ok(res.body.data.prediction.modelName);
  });

  // 5. Heart disease live prediction and persistence
  test('5. POST /api/predictions/heart performs inference and persists record', async () => {
    const res = await request(app)
      .post('/api/predictions/heart')
      .set('Authorization', `Bearer ${userToken}`)
      .send(validHeartInput);

    assert.equal(res.status, 201);
    assert.equal(res.body.success, true);
    assert.ok(res.body.data.prediction);
    assert.equal(res.body.data.prediction.user, userId);
    assert.equal(res.body.data.prediction.modelType, 'heart');
    assert.ok(typeof res.body.data.prediction.riskScore === 'number');
    assert.ok(['Low', 'Mild', 'Moderate', 'High', 'Critical', 'Elevated'].includes(res.body.data.prediction.riskLevel));
    assert.ok(Array.isArray(res.body.data.prediction.probabilities));
  });

  // 6. User isolation and list predictions
  test('6. GET /api/predictions lists predictions scoped to authenticated user', async () => {
    // Create prediction for primary user
    await request(app)
      .post('/api/predictions/diabetes')
      .set('Authorization', `Bearer ${userToken}`)
      .send(validDiabetesInput);

    // Create prediction for other user
    await request(app)
      .post('/api/predictions/heart')
      .set('Authorization', `Bearer ${otherUserToken}`)
      .send(validHeartInput);

    // Primary user list
    const res = await request(app)
      .get('/api/predictions')
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.data.predictions.length, 1);
    assert.equal(res.body.data.predictions[0].user, userId);
    assert.equal(res.body.data.predictions[0].modelType, 'diabetes');
    assert.equal(res.body.data.pagination.total, 1);
  });

  // 7. Get single prediction by ID
  test('7. GET /api/predictions/:id retrieves owned prediction and hides other user prediction', async () => {
    const created = await request(app)
      .post('/api/predictions/diabetes')
      .set('Authorization', `Bearer ${userToken}`)
      .send(validDiabetesInput);

    const predId = created.body.data.prediction.id || created.body.data.prediction._id;

    // Owner fetch
    const ownerRes = await request(app)
      .get(`/api/predictions/${predId}`)
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(ownerRes.status, 200);
    assert.equal(ownerRes.body.data.prediction.modelType, 'diabetes');

    // Other user fetch -> should return 404
    const otherRes = await request(app)
      .get(`/api/predictions/${predId}`)
      .set('Authorization', `Bearer ${otherUserToken}`);

    assert.equal(otherRes.status, 404);
  });

  // 8. Delete prediction
  test('8. DELETE /api/predictions/:id deletes owned prediction', async () => {
    const created = await request(app)
      .post('/api/predictions/heart')
      .set('Authorization', `Bearer ${userToken}`)
      .send(validHeartInput);

    const predId = created.body.data.prediction.id || created.body.data.prediction._id;

    const delRes = await request(app)
      .delete(`/api/predictions/${predId}`)
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(delRes.status, 200);
    assert.equal(delRes.body.success, true);

    // Verify deleted
    const getRes = await request(app)
      .get(`/api/predictions/${predId}`)
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(getRes.status, 404);
  });

  // 9. Cascade delete on account removal
  test('9. DELETE /api/users/profile cascades and deletes all user predictions', async () => {
    await request(app)
      .post('/api/predictions/diabetes')
      .set('Authorization', `Bearer ${userToken}`)
      .send(validDiabetesInput);

    assert.equal(inMemoryPredictions.length, 1);

    const delAccountRes = await request(app)
      .delete('/api/users/profile')
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(delAccountRes.status, 200);
    assert.equal(inMemoryPredictions.length, 0);
  });
});
