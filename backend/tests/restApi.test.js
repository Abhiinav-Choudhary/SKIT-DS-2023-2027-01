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
const { generateToken } = require('../src/utils/jwt');

// In-memory data stores for mocking Mongoose operations
let inMemoryUsers = [];
let inMemoryHealthRecords = [];
let userCounter = 1;
let recordCounter = 1;

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
    _id: found._id,
    id: found._id,
    name: found.name,
    email: found.email,
    password: found.password,
    age: found.age,
    gender: found.gender,
    bloodGroup: found.bloodGroup,
    phone: found.phone,
    heightCm: found.heightCm,
    weightKg: found.weightKg,
    createdAt: found.createdAt,
    updatedAt: found.updatedAt,
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

  const queryObj = {
    select: (fields) => Promise.resolve(doc),
    then: (resolve, reject) => Promise.resolve(doc).then(resolve, reject),
  };

  return queryObj;
};

User.findByIdAndUpdate = async (id, updates, options) => {
  const found = inMemoryUsers.find((u) => u._id === id);
  if (!found) return null;

  Object.assign(found, updates);
  found.updatedAt = new Date();

  return {
    _id: found._id,
    id: found._id,
    name: found.name,
    email: found.email,
    age: found.age,
    gender: found.gender,
    bloodGroup: found.bloodGroup,
    phone: found.phone,
    heightCm: found.heightCm,
    weightKg: found.weightKg,
    createdAt: found.createdAt,
    updatedAt: found.updatedAt,
  };
};

User.findByIdAndDelete = async (id) => {
  const idx = inMemoryUsers.findIndex((u) => u._id === id);
  if (idx === -1) return null;
  const removed = inMemoryUsers.splice(idx, 1)[0];
  return removed;
};

// Setup in-memory HealthRecord mock methods
HealthRecord.create = async (data) => {
  const record = {
    _id: `rec_${recordCounter++}`,
    user: data.user,
    recordedAt: data.recordedAt ? new Date(data.recordedAt) : new Date(),
    measurements: { ...data.measurements },
    notes: data.notes || '',
    createdAt: new Date(),
    updatedAt: new Date(),
  };
  record.id = record._id;
  inMemoryHealthRecords.push(record);
  return record;
};

HealthRecord.find = (query) => {
  let matched = inMemoryHealthRecords.filter((r) => {
    if (query.user && r.user !== query.user) return false;
    if (query._id && r._id !== query._id) return false;
    return true;
  });

  const queryObj = {
    _data: matched,
    sort(sortObj) {
      if (sortObj && sortObj.recordedAt === -1) {
        this._data.sort((a, b) => new Date(b.recordedAt) - new Date(a.recordedAt));
      }
      return this;
    },
    skip(count) {
      this._data = this._data.slice(count);
      return this;
    },
    limit(count) {
      this._data = this._data.slice(0, count);
      return this;
    },
    then(resolve, reject) {
      return Promise.resolve(this._data).then(resolve, reject);
    },
  };

  return queryObj;
};

HealthRecord.countDocuments = async (query) => {
  return inMemoryHealthRecords.filter((r) => {
    if (query.user && r.user !== query.user) return false;
    return true;
  }).length;
};

HealthRecord.findOne = async (query) => {
  const found = inMemoryHealthRecords.find((r) => {
    if (query._id && r._id !== query._id) return false;
    if (query.user && r.user !== query.user) return false;
    return true;
  });

  if (!found) return null;

  return {
    ...found,
    id: found._id,
    measurements: { ...found.measurements },
  };
};

HealthRecord.findOneAndUpdate = async (query, updates, options) => {
  const found = inMemoryHealthRecords.find((r) => {
    if (query._id && r._id !== query._id) return false;
    if (query.user && r.user !== query.user) return false;
    return true;
  });

  if (!found) return null;

  if (updates.notes !== undefined) found.notes = updates.notes;
  if (updates.recordedAt !== undefined) found.recordedAt = new Date(updates.recordedAt);
  if (updates.measurements) {
    found.measurements = { ...found.measurements, ...updates.measurements };
  }
  found.updatedAt = new Date();

  return {
    ...found,
    id: found._id,
    measurements: { ...found.measurements },
  };
};

HealthRecord.findOneAndDelete = async (query) => {
  const idx = inMemoryHealthRecords.findIndex((r) => {
    if (query._id && r._id !== query._id) return false;
    if (query.user && r.user !== query.user) return false;
    return true;
  });

  if (idx === -1) return null;
  const removed = inMemoryHealthRecords.splice(idx, 1)[0];
  return removed;
};

HealthRecord.deleteMany = async (query) => {
  const initialLength = inMemoryHealthRecords.length;
  inMemoryHealthRecords = inMemoryHealthRecords.filter((r) => {
    if (query.user && r.user === query.user) return false;
    return true;
  });
  return { deletedCount: initialLength - inMemoryHealthRecords.length };
};

describe('REST API Test Suite (User & Health Data Endpoints)', () => {
  let userToken;
  let testUserId;

  beforeEach(async () => {
    inMemoryUsers = [];
    inMemoryHealthRecords = [];
    userCounter = 1;
    recordCounter = 1;

    // Seed primary authenticated user
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash('Password123!', salt);
    testUserId = `user_lead_${userCounter++}`;

    inMemoryUsers.push({
      _id: testUserId,
      id: testUserId,
      name: 'Abhinav Kumar Chaudhary',
      email: 'abhinav.lead@example.com',
      password: hashedPassword,
      age: 21,
      gender: 'male',
      bloodGroup: 'B+',
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    userToken = generateToken({ id: testUserId });
  });

  // 1. GET /api/users/profile
  test('1. GET /api/users/profile returns authenticated user profile', async () => {
    const res = await request(app)
      .get('/api/users/profile')
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.data.user.name, 'Abhinav Kumar Chaudhary');
    assert.equal(res.body.data.user.email, 'abhinav.lead@example.com');
    assert.equal(res.body.data.user.bloodGroup, 'B+');
    assert.equal(res.body.data.user.password, undefined);
  });

  // 2. PATCH /api/users/profile
  test('2. PATCH /api/users/profile updates user details successfully', async () => {
    const res = await request(app)
      .patch('/api/users/profile')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        name: 'Abhinav K. Chaudhary',
        phone: '+91 9179818502',
        heightCm: 178,
        weightKg: 72,
      });

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.data.user.name, 'Abhinav K. Chaudhary');
    assert.equal(res.body.data.user.phone, '+91 9179818502');
    assert.equal(res.body.data.user.heightCm, 178);
    assert.equal(res.body.data.user.weightKg, 72);
  });

  // 3. PATCH /api/users/profile with no allowed fields
  test('3. PATCH /api/users/profile rejects empty or disallowed fields with 400', async () => {
    const res = await request(app)
      .patch('/api/users/profile')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ invalidKey: 'unsupported' });

    assert.equal(res.status, 400);
    assert.equal(res.body.success, false);
  });

  // 4. PUT /api/users/change-password successfully
  test('4. PUT /api/users/change-password changes password when credentials valid', async () => {
    const res = await request(app)
      .put('/api/users/change-password')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        currentPassword: 'Password123!',
        newPassword: 'BrandNewPassword456!',
      });

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.message, 'Password changed successfully');
  });

  // 5. PUT /api/users/change-password rejects incorrect current password
  test('5. PUT /api/users/change-password rejects wrong current password with 400', async () => {
    const res = await request(app)
      .put('/api/users/change-password')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        currentPassword: 'WrongPassword!',
        newPassword: 'BrandNewPassword456!',
      });

    assert.equal(res.status, 400);
    assert.equal(res.body.success, false);
    assert.match(res.body.message, /Current password is incorrect/);
  });

  // 6. POST /api/health-records creates health record successfully
  test('6. POST /api/health-records creates health record and returns 201', async () => {
    const res = await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        recordedAt: '2026-10-22T08:00:00.000Z',
        measurements: {
          systolicBP: 120,
          diastolicBP: 80,
          heartRateBpm: 72,
          bloodGlucoseMgDl: 95,
          oxygenSaturationPercent: 99,
        },
        notes: 'Routine baseline morning checkup',
      });

    assert.equal(res.status, 201);
    assert.equal(res.body.success, true);
    assert.ok(res.body.data.record.id);
    assert.equal(res.body.data.record.measurements.systolicBP, 120);
    assert.equal(res.body.data.record.measurements.bloodGlucoseMgDl, 95);
  });

  // 7. POST /api/health-records rejects invalid BP (systolic <= diastolic)
  test('7. POST /api/health-records rejects systolicBP <= diastolicBP with 400', async () => {
    const res = await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        measurements: {
          systolicBP: 80,
          diastolicBP: 120,
        },
      });

    assert.equal(res.status, 400);
    assert.equal(res.body.success, false);
    assert.match(res.body.message, /Systolic blood pressure must exceed diastolic/);
  });

  // 8. GET /api/health-records lists records with pagination
  test('8. GET /api/health-records lists records scoped to user', async () => {
    // Add two records
    await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ measurements: { heartRateBpm: 70 } });

    await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ measurements: { heartRateBpm: 75 } });

    const res = await request(app)
      .get('/api/health-records?page=1&limit=10')
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.data.records.length, 2);
    assert.equal(res.body.data.pagination.total, 2);
  });

  // 9. GET /api/health-records/summary returns vitals analytics and averages
  test('9. GET /api/health-records/summary calculates health stats correctly', async () => {
    await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        measurements: { systolicBP: 120, diastolicBP: 80, heartRateBpm: 70, bloodGlucoseMgDl: 100 },
      });

    await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        measurements: { systolicBP: 130, diastolicBP: 90, heartRateBpm: 80, bloodGlucoseMgDl: 110 },
      });

    const res = await request(app)
      .get('/api/health-records/summary')
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.data.totalRecords, 2);
    assert.equal(res.body.data.averages.systolicBP, 125);
    assert.equal(res.body.data.averages.diastolicBP, 85);
    assert.equal(res.body.data.averages.heartRateBpm, 75);
    assert.equal(res.body.data.averages.bloodGlucoseMgDl, 105);
  });

  // 10. GET /api/health-records/:id retrieves single record
  test('10. GET /api/health-records/:id retrieves single record', async () => {
    const created = await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        measurements: { heartRateBpm: 68 },
        notes: 'Single fetch test',
      });

    const recordId = created.body.data.record.id;

    const res = await request(app)
      .get(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.data.record.id, recordId);
    assert.equal(res.body.data.record.notes, 'Single fetch test');
  });

  // 11. PATCH /api/health-records/:id updates record
  test('11. PATCH /api/health-records/:id updates measurements and notes', async () => {
    const created = await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        measurements: { systolicBP: 120, diastolicBP: 80 },
        notes: 'Initial note',
      });

    const recordId = created.body.data.record.id;

    const res = await request(app)
      .patch(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        notes: 'Updated note after rest',
        measurements: { heartRateBpm: 65 },
      });

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.equal(res.body.data.record.notes, 'Updated note after rest');
    assert.equal(res.body.data.record.measurements.heartRateBpm, 65);
    assert.equal(res.body.data.record.measurements.systolicBP, 120);
  });

  // 12. DELETE /api/health-records/:id removes record
  test('12. DELETE /api/health-records/:id deletes health record', async () => {
    const created = await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({
        measurements: { heartRateBpm: 88 },
      });

    const recordId = created.body.data.record.id;

    const res = await request(app)
      .delete(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);

    // Verify it is gone
    const fetchAgain = await request(app)
      .get(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(fetchAgain.status, 404);
  });

  // 13. DELETE /api/users/profile deletes account and cascades
  test('13. DELETE /api/users/profile deletes user and cascades health records', async () => {
    await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ measurements: { heartRateBpm: 72 } });

    const res = await request(app)
      .delete('/api/users/profile')
      .set('Authorization', `Bearer ${userToken}`);

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.match(res.body.message, /Account and associated health data deleted/);

    assert.equal(inMemoryHealthRecords.length, 0);
  });

  // 14. Unauthenticated access rejected
  test('14. Protected endpoints reject requests without token with 401', async () => {
    const resProfile = await request(app).get('/api/users/profile');
    assert.equal(resProfile.status, 401);

    const resRecords = await request(app).get('/api/health-records');
    assert.equal(resRecords.status, 401);
  });
});
