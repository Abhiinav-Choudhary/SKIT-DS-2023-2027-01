const { test, describe, before, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const request = require('supertest');
const bcrypt = require('bcryptjs');

// Set test environment variables before requiring the app
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test_super_secret_jwt_key_for_testing_1234567890';
process.env.JWT_EXPIRES_IN = '1h';

const app = require('../src/app');
const User = require('../src/models/User');

// In-memory mock store for test execution without external DB dependencies
let inMemoryUsers = [];
let idCounter = 1;

// Mock User Mongoose model methods to simulate database operations reliably
User.findOne = (query) => {
  return {
    select: (fields) => {
      return new Promise((resolve) => {
        const found = inMemoryUsers.find((u) => u.email === query.email);
        if (!found) return resolve(null);

        // Return user document with comparePassword method
        const doc = {
          _id: found._id,
          id: found._id,
          name: found.name,
          email: found.email,
          password: found.password,
          createdAt: found.createdAt,
          updatedAt: found.updatedAt,
          comparePassword: async (pwd) => bcrypt.compare(pwd, found.password),
        };
        resolve(doc);
      });
    },
    then: function (resolve, reject) {
      return this.select().then(resolve, reject);
    },
  };
};

User.create = async ({ name, email, password }) => {
  const salt = await bcrypt.genSalt(10);
  const hashedPassword = await bcrypt.hash(password, salt);
  const user = {
    _id: `user_id_${idCounter++}`,
    name,
    email,
    password: hashedPassword,
    createdAt: new Date(),
    updatedAt: new Date(),
  };
  inMemoryUsers.push(user);

  return {
    _id: user._id,
    id: user._id,
    name: user.name,
    email: user.email,
    createdAt: user.createdAt,
  };
};

User.findById = async (id) => {
  const found = inMemoryUsers.find((u) => u._id === id);
  if (!found) return null;
  return {
    _id: found._id,
    id: found._id,
    name: found.name,
    email: found.email,
    createdAt: found.createdAt,
    updatedAt: found.updatedAt,
  };
};

describe('Authentication & Authorization Test Suite', () => {
  beforeEach(() => {
    inMemoryUsers = [];
    idCounter = 1;
  });

  // 1. Successful registration
  test('1. Successful registration creates user and returns 201 without password hash', async () => {
    const res = await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav Kumar Chaudhary',
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    assert.equal(res.status, 201);
    assert.equal(res.body.success, true);
    assert.equal(res.body.message, 'User registered successfully');
    assert.ok(res.body.data.user.id);
    assert.equal(res.body.data.user.name, 'Abhinav Kumar Chaudhary');
    assert.equal(res.body.data.user.email, 'abhinav@example.com');
    assert.equal(res.body.data.user.password, undefined);
  });

  // 2. Duplicate registration
  test('2. Duplicate registration returns 409 Conflict', async () => {
    // First registration
    await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav Kumar Chaudhary',
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    // Attempt duplicate registration
    const res = await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav Duplicate',
        email: 'abhinav@example.com',
        password: 'anotherPassword123',
      });

    assert.equal(res.status, 409);
    assert.equal(res.body.success, false);
    assert.match(res.body.message, /already registered/i);
  });

  // 3. Invalid registration data
  test('3. Invalid registration data (missing fields, invalid email, short password) returns 400', async () => {
    // Missing fields
    const resMissing = await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav',
      });
    assert.equal(resMissing.status, 400);
    assert.equal(resMissing.body.success, false);

    // Invalid email format
    const resBadEmail = await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav',
        email: 'invalid-email-format',
        password: 'securePassword123',
      });
    assert.equal(resBadEmail.status, 400);
    assert.equal(resBadEmail.body.success, false);

    // Password too short (< 6 chars)
    const resShortPass = await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav',
        email: 'valid@example.com',
        password: '123',
      });
    assert.equal(resShortPass.status, 400);
    assert.equal(resShortPass.body.success, false);
  });

  // 4. Successful login
  test('4. Successful login returns 200 with JWT and user profile', async () => {
    // Register first
    await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav Kumar Chaudhary',
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    // Login with valid credentials
    const res = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.ok(res.body.data.token);
    assert.equal(typeof res.body.data.token, 'string');
    assert.equal(res.body.data.user.email, 'abhinav@example.com');
  });

  // 5. Incorrect password
  test('5. Login with incorrect password returns 401 Unauthorized', async () => {
    await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav Kumar Chaudhary',
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    const res = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'abhinav@example.com',
        password: 'wrongPassword123',
      });

    assert.equal(res.status, 401);
    assert.equal(res.body.success, false);
    assert.equal(res.body.message, 'Invalid credentials');
  });

  // 6. Non-existent user login
  test('6. Login with non-existent user returns 401 Unauthorized', async () => {
    const res = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'nobody@example.com',
        password: 'anyPassword123',
      });

    assert.equal(res.status, 401);
    assert.equal(res.body.success, false);
    assert.equal(res.body.message, 'Invalid credentials');
  });

  // 7. GET /api/auth/me with valid JWT
  test('7. GET /api/auth/me with valid JWT returns authenticated user details', async () => {
    await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav Kumar Chaudhary',
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    const loginRes = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    const token = loginRes.body.data.token;

    const meRes = await request(app)
      .get('/api/auth/me')
      .set('Authorization', `Bearer ${token}`);

    assert.equal(meRes.status, 200);
    assert.equal(meRes.body.success, true);
    assert.equal(meRes.body.data.user.email, 'abhinav@example.com');
  });

  // 8. GET /api/auth/me without JWT
  test('8. GET /api/auth/me without JWT returns 401 Unauthorized', async () => {
    const res = await request(app).get('/api/auth/me');

    assert.equal(res.status, 401);
    assert.equal(res.body.success, false);
    assert.match(res.body.message, /Access denied/i);
  });

  // 9. GET /api/auth/me with invalid JWT
  test('9. GET /api/auth/me with invalid JWT returns 401 Unauthorized', async () => {
    const res = await request(app)
      .get('/api/auth/me')
      .set('Authorization', 'Bearer invalid.token.signature');

    assert.equal(res.status, 401);
    assert.equal(res.body.success, false);
    assert.equal(res.body.message, 'Invalid token');
  });

  // 10. Protected test route
  test('10. GET /api/test/protected succeeds with valid token and fails without', async () => {
    // Register and login to obtain token
    await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Abhinav Kumar Chaudhary',
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    const loginRes = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'abhinav@example.com',
        password: 'securePassword123',
      });

    const token = loginRes.body.data.token;

    // Call without token -> 401
    const unauthRes = await request(app).get('/api/test/protected');
    assert.equal(unauthRes.status, 401);
    assert.equal(unauthRes.body.success, false);

    // Call with valid token -> 200
    const authRes = await request(app)
      .get('/api/test/protected')
      .set('Authorization', `Bearer ${token}`);
    assert.equal(authRes.status, 200);
    assert.equal(authRes.body.success, true);
    assert.equal(authRes.body.message, 'Authenticated successfully');
  });

  // 11. Logout behavior
  test('11. POST /api/auth/logout returns 200 and confirms logout', async () => {
    const res = await request(app).post('/api/auth/logout');

    assert.equal(res.status, 200);
    assert.equal(res.body.success, true);
    assert.match(res.body.message, /Logged out successfully/i);
  });
});
