const { test, describe, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const request = require('supertest');
const bcrypt = require('bcryptjs');
const mongoose = require('mongoose');

process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test_super_secret_jwt_key_for_testing_1234567890';
process.env.JWT_EXPIRES_IN = '1h';

const app = require('../src/app');
const User = require('../src/models/User');
const HealthRecord = require('../src/models/HealthRecord');

let users = [];
let records = [];
let userIdCounter = 1;
let recordIdCounter = 1;

User.findOne = (query) => ({
  select: () => {
    const found = users.find((user) => user.email === query.email);
    if (!found) return Promise.resolve(null);
    return Promise.resolve({
      ...found,
      comparePassword: (password) => bcrypt.compare(password, found.password),
    });
  },
  then(resolve, reject) {
    return this.select().then(resolve, reject);
  },
});

User.create = async ({ name, email, password }) => {
  const user = {
    _id: `user_id_${userIdCounter++}`,
    name,
    email,
    password: await bcrypt.hash(password, await bcrypt.genSalt(10)),
  };
  users.push(user);
  return user;
};

User.findById = async (id) => users.find((user) => user._id === id) || null;

HealthRecord.create = async (data) => {
  const record = {
    ...data,
    _id: `record_id_${recordIdCounter++}`,
    recordedAt: new Date(data.recordedAt || Date.now()),
    createdAt: new Date(),
    updatedAt: new Date(),
  };
  record.id = record._id;
  records.push(record);
  return record;
};

HealthRecord.find = (query) => {
  let result = records.filter((record) => record.user === query.user);
  const queryBuilder = {
    sort() {
      result = [...result].sort((left, right) => right.recordedAt - left.recordedAt);
      return queryBuilder;
    },
    skip(count) {
      result = result.slice(count);
      return queryBuilder;
    },
    limit(count) {
      result = result.slice(0, count);
      return Promise.resolve(result);
    },
  };
  return queryBuilder;
};

HealthRecord.findOne = async (query) =>
  records.find((record) => record._id === query._id && record.user === query.user) || null;

HealthRecord.findOneAndUpdate = async (query, updates) => {
  const record = records.find(
    (item) => item._id === query._id && item.user === query.user
  );
  if (!record) return null;
  Object.assign(record, updates, { updatedAt: new Date() });
  return record;
};

HealthRecord.findOneAndDelete = async (query) => {
  const index = records.findIndex(
    (record) => record._id === query._id && record.user === query.user
  );
  if (index < 0) return null;
  return records.splice(index, 1)[0];
};

const createUserToken = async (email) => {
  await request(app).post('/api/auth/register').send({
    name: 'Test Patient',
    email,
    password: 'securePassword123',
  });
  const response = await request(app).post('/api/auth/login').send({
    email,
    password: 'securePassword123',
  });
  return response.body.data.token;
};

describe('Health Record API', () => {
  beforeEach(() => {
    users = [];
    records = [];
    userIdCounter = 1;
    recordIdCounter = 1;
  });

  test('requires authentication for health-record endpoints', async () => {
    const response = await request(app).get('/api/health-records');
    assert.equal(response.status, 401);
  });

  test('creates and lists only the authenticated user records', async () => {
    const patientToken = await createUserToken('patient@example.com');
    const otherToken = await createUserToken('other@example.com');
    const created = await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${patientToken}`)
      .send({
        measurements: { systolicBP: 120, diastolicBP: 80, heartRateBpm: 72 },
        notes: 'Morning reading',
        user: 'user_id_2',
      });

    assert.equal(created.status, 201);
    assert.equal(created.body.data.record.user, 'user_id_1');

    const patientRecords = await request(app)
      .get('/api/health-records?limit=1')
      .set('Authorization', `Bearer ${patientToken}`);
    const otherRecords = await request(app)
      .get('/api/health-records')
      .set('Authorization', `Bearer ${otherToken}`);

    assert.equal(patientRecords.status, 200);
    assert.equal(patientRecords.body.data.records.length, 1);
    assert.equal(patientRecords.body.data.limit, 1);
    assert.equal(otherRecords.status, 200);
    assert.equal(otherRecords.body.data.records.length, 0);
  });

  test('supports owner CRUD and hides another user record as not found', async () => {
    const ownerToken = await createUserToken('owner@example.com');
    const otherToken = await createUserToken('visitor@example.com');
    const created = await request(app)
      .post('/api/health-records')
      .set('Authorization', `Bearer ${ownerToken}`)
      .send({ measurements: { systolicBP: 120, diastolicBP: 80, heartRateBpm: 65 } });
    const recordId = created.body.data.record.id;

    const fetched = await request(app)
      .get(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${ownerToken}`);
    const invalidMeasurementUpdate = await request(app)
      .patch(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${ownerToken}`)
      .send({ measurements: { systolicBP: 70 } });
    const updated = await request(app)
      .patch(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${ownerToken}`)
      .send({ measurements: { systolicBP: 125 }, notes: 'Updated note', user: 'user_id_2' });
    const foreignRead = await request(app)
      .get(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${otherToken}`);
    const foreignDelete = await request(app)
      .delete(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${otherToken}`);
    const deleted = await request(app)
      .delete(`/api/health-records/${recordId}`)
      .set('Authorization', `Bearer ${ownerToken}`);

    assert.equal(fetched.status, 200);
    assert.equal(invalidMeasurementUpdate.status, 400);
    assert.equal(updated.status, 200);
    assert.equal(updated.body.data.record.user, 'user_id_1');
    assert.equal(updated.body.data.record.measurements.diastolicBP, 80);
    assert.equal(foreignRead.status, 404);
    assert.equal(foreignDelete.status, 404);
    assert.equal(deleted.status, 200);
    assert.equal(records.length, 0);
  });

  test('rejects invalid pagination and updates without writable fields', async () => {
    const token = await createUserToken('paging@example.com');
    const invalidPage = await request(app)
      .get('/api/health-records?page=0')
      .set('Authorization', `Bearer ${token}`);
    const invalidLimit = await request(app)
      .get('/api/health-records?limit=101')
      .set('Authorization', `Bearer ${token}`);
    const invalidUpdate = await request(app)
      .patch('/api/health-records/record_id_1')
      .set('Authorization', `Bearer ${token}`)
      .send({ user: 'user_id_1' });

    assert.equal(invalidPage.status, 400);
    assert.equal(invalidLimit.status, 400);
    assert.equal(invalidUpdate.status, 400);
  });
});

describe('Health Record Mongoose Schema', () => {
  test('requires at least one measurement and validates blood pressure ordering', async () => {
    const user = new mongoose.Types.ObjectId();
    const empty = new HealthRecord({ user, measurements: {} });
    const invalidBloodPressure = new HealthRecord({
      user,
      measurements: { systolicBP: 70, diastolicBP: 80 },
    });

    await assert.rejects(empty.validate(), { name: 'ValidationError' });
    await assert.rejects(invalidBloodPressure.validate(), { name: 'ValidationError' });
  });

  test('indexes patient records by owner and recording date', () => {
    assert.ok(
      HealthRecord.schema.indexes().some(([fields]) =>
        fields.user === 1 && fields.recordedAt === -1
      )
    );
  });
});