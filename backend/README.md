# AI-Powered Health Risk Prediction and Monitoring System - Backend API

Backend service providing RESTful architecture, MongoDB connectivity, and JWT-based authentication for the AI-Powered Health Risk Prediction and Monitoring System.

---

## 1. Project Overview & Current Phase

- **Project ID**: SKIT/DS/2023-2027/01
- **Current Development Phase**: Backend Architecture, Authentication, REST APIs & AI Prediction API Integration (Sprint 3 & Sprint 4: Active as of 13/11/2026)
- **Status**: Implemented & Fully Verified (40 Automated Tests Passing across 4 Suites)
- **Assigned Lead**: Abhinav Kumar Chaudhary (Team Lead - Backend & API Architecture)

*Note: Health Monitoring Module & Risk-Score tracking and Alert systems belong to subsequent project phases (December 2026 onwards).*

---

## 2. Technology Stack

- **Runtime**: Node.js (`v24+`)
- **Framework**: Express.js (`v4.x`)
- **Language**: JavaScript (ES6+ CommonJS)
- **Database**: MongoDB (Atlas) via Mongoose (`v8.x`)
- **Authentication**: JSON Web Tokens (`jsonwebtoken`)
- **Password Hashing**: `bcryptjs`
- **Testing**: Node.js native test runner (`node:test`) + `supertest`

---

## 3. Architecture & Directory Structure

The backend follows a clean, modular layer separation:
```
backend/
├── src/
│   ├── config/
│   │   └── db.js                    # MongoDB Mongoose connection
│   ├── controllers/
│   │   ├── auth.controller.js       # Authentication route handlers
│   │   ├── user.controller.js       # User profile & account handlers
│   │   ├── healthRecord.controller.js # Health record CRUD & analytics handlers
│   │   └── prediction.controller.js # AI risk prediction & history handlers
│   ├── middleware/
│   │   ├── auth.middleware.js       # JWT verification (requireAuth)
│   │   └── error.middleware.js      # Centralized error & 404 handler
│   ├── ml/
│   │   └── inference_bridge.py      # Python subprocess bridge to trained models
│   ├── models/
│   │   ├── User.js                  # User schema with profile fields & bcrypt
│   │   ├── HealthRecord.js          # Health vitals schema with BP validation
│   │   └── Prediction.js            # AI prediction persistence schema
│   ├── routes/
│   │   ├── auth.routes.js           # Auth endpoints (/api/auth)
│   │   ├── test.routes.js           # Protected test endpoint (/api/test)
│   │   ├── user.routes.js           # User profile endpoints (/api/users)
│   │   ├── healthRecord.routes.js   # Health data endpoints (/api/health-records)
│   │   └── prediction.routes.js     # AI prediction endpoints (/api/predictions)
│   ├── services/
│   │   ├── auth.service.js          # Authentication business logic & DB transactions
│   │   ├── user.service.js          # User profile & password management logic
│   │   ├── healthRecord.service.js  # Health record CRUD & vitals analytics logic
│   │   └── prediction.service.js    # ML inference bridge execution & DB persistence
│   ├── utils/
│   │   └── jwt.js                   # Token signing & verification
│   ├── app.js                       # Express application setup
│   └── server.js                    # Application entry point
├── tests/
│   ├── auth.test.js                 # Authentication automated tests (11 scenarios)
│   ├── healthRecord.test.js         # Health record isolation tests (6 scenarios)
│   ├── restApi.test.js              # User & Health REST API tests (14 scenarios)
│   └── prediction.test.js           # AI prediction & inference tests (9 scenarios)
├── .env.example                     # Sample environment variables
├── .gitignore                       # Git ignore rules
├── package.json                     # Project dependencies and scripts
└── README.md                        # Backend documentation
```

### Flow Pattern
```
HTTP Request
  └─► Router (routes/)
        └─► Controller (controllers/)
              └─► Service (services/)
                    └─► Model / Database (models/ & config/)
```

---

## 4. Environment Variables

Create a `.env` file in the `backend/` directory based on `.env.example`:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# MongoDB Atlas Connection String
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/health_risk_db?retryWrites=true&w=majority

# JWT Authentication
JWT_SECRET=your_jwt_secret_key_here_at_least_32_characters_long
JWT_EXPIRES_IN=7d
```

---

## 5. MongoDB Atlas Setup

1. Create a free cluster on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Set up a Database User with read/write privileges in **Database Access**.
3. Add your current IP address (or `0.0.0.0/0` during development) in **Network Access**.
4. Obtain the connection string from **Database > Connect > Drivers**.
5. Paste the connection string into `backend/.env` under `MONGO_URI`, replacing `<username>` and `<password>`.

---

## 6. Installation & Execution

### Prerequisites
- Node.js installed (`v18+` recommended)
- npm installed

### Install Dependencies
```bash
# From within the backend directory:
npm install
```

### Start Backend Server
```bash
# Production / Normal start
npm start

# Development mode (auto-reload via nodemon)
npm run dev
```

The server will start listening at `http://localhost:5000`.

---

## 7. Authentication Flow

1. **Registration (`POST /api/auth/register`)**:
   - Accepts `name`, `email`, and `password`.
   - Validates email format and ensures password is at least 6 characters.
   - Verifies email uniqueness (returns `409 Conflict` if duplicate).
   - Hashes password using `bcryptjs` salt rounds before persisting to MongoDB.
   - Returns sanitized user details (excluding password hash).

2. **Login (`POST /api/auth/login`)**:
   - Accepts `email` and `password`.
   - Finds user and compares hashed password.
   - Generates signed JWT payload containing user ID with configurable expiry.
   - Returns JWT token and sanitized user details.

3. **Protected Access (`GET /api/auth/me` & `GET /api/test/protected`)**:
   - Client sends token in header: `Authorization: Bearer <token>`.
   - `requireAuth` middleware verifies token validity, checks expiration, and retrieves user profile from database.

4. **Logout (`POST /api/auth/logout`)**:
   - Informs client to discard the stored JWT token.

---

## 8. Current API Endpoints

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/` | Public | Service health & status message |
| `POST` | `/api/auth/register` | Public | Register a new user |
| `POST` | `/api/auth/login` | Public | Authenticate user & receive JWT |
| `GET` | `/api/auth/me` | Protected | Get authenticated user profile |
| `POST` | `/api/auth/logout` | Public | Invalidate client session / token |
| `GET` | `/api/test/protected` | Protected | Verification route for authorization |
| `GET` | `/api/users/profile` | Protected | Retrieve full user profile & vitals metrics |
| `PATCH` | `/api/users/profile` | Protected | Update user profile attributes (age, phone, weight, etc.) |
| `PUT` | `/api/users/change-password` | Protected | Change password with verification |
| `DELETE` | `/api/users/profile` | Protected | Delete account & cascade delete health records |
| `POST` | `/api/health-records` | Protected | Create user health vitals record |
| `GET` | `/api/health-records` | Protected | List user health records (pagination, filtering) |
| `GET` | `/api/health-records/summary` | Protected | Get vitals aggregate analytics & latest record |
| `GET` | `/api/health-records/:id` | Protected | Get single health record by ID |
| `PATCH` | `/api/health-records/:id` | Protected | Update health record measurements/notes |
| `DELETE` | `/api/health-records/:id` | Protected | Delete health record by ID |
| `POST` | `/api/predictions/diabetes` | Protected | Run live AI diabetes risk inference & persist result |
| `POST` | `/api/predictions/heart` | Protected | Run live AI heart disease risk inference & persist result |
| `GET` | `/api/predictions` | Protected | List user prediction history (pagination, type filter) |
| `GET` | `/api/predictions/:id` | Protected | Retrieve specific prediction record by ID |
| `DELETE` | `/api/predictions/:id` | Protected | Delete specific prediction record by ID |

---

## 9. Example Requests & Responses

### 1. Register User
**Request**:
```http
POST /api/auth/register
Content-Type: application/json

{
  "name": "Abhinav Kumar Chaudhary",
  "email": "abhinav@example.com",
  "password": "securePassword123"
}
```

**Success Response (201 Created)**:
```json
{
  "success": true,
  "message": "User registered successfully",
  "data": {
    "user": {
      "id": "6704b123456789abcdef0123",
      "name": "Abhinav Kumar Chaudhary",
      "email": "abhinav@example.com",
      "createdAt": "2026-10-08T10:00:00.000Z"
    }
  }
}
```

### 2. Login User
**Request**:
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "abhinav@example.com",
  "password": "securePassword123"
}
```

**Success Response (200 OK)**:
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "6704b123456789abcdef0123",
      "name": "Abhinav Kumar Chaudhary",
      "email": "abhinav@example.com"
    }
  }
}
```

### 3. Get Current User (`/api/auth/me`)
**Request**:
```http
GET /api/auth/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Success Response (200 OK)**:
```json
{
  "success": true,
  "message": "Current user retrieved successfully",
  "data": {
    "user": {
      "id": "6704b123456789abcdef0123",
      "name": "Abhinav Kumar Chaudhary",
      "email": "abhinav@example.com",
      "createdAt": "2026-10-08T10:00:00.000Z",
      "updatedAt": "2026-10-08T10:00:00.000Z"
    }
  }
}
```

### 4. Protected Test Route (`/api/test/protected`)
**Request**:
```http
GET /api/test/protected
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Success Response (200 OK)**:
```json
{
  "success": true,
  "message": "Authenticated successfully"
}
```

**Unauthorized Response (401 Unauthorized)**:
```json
{
  "success": false,
  "message": "Access denied. No token provided."
}
```

### 5. Logout
**Request**:
```http
POST /api/auth/logout
```

**Response (200 OK)**:
```json
{
  "success": true,
  "message": "Logged out successfully. Please clear the authentication token on the client."
}
```

---

## 10. Testing Instructions

The repository includes a comprehensive automated test suite verifying all 40 required scenarios across 4 test suites:
- **Authentication & Authorization Suite** (`tests/auth.test.js` - 11 tests)
- **Health Record Isolation & Schema Suite** (`tests/healthRecord.test.js` - 6 tests)
- **REST API Suite for User & Health Data** (`tests/restApi.test.js` - 14 tests)
- **AI Prediction API & Inference Suite** (`tests/prediction.test.js` - 9 tests)

Run the full test suite with:
```bash
npm test
```

Or run targeted test suites:
```bash
npm run test:auth
npm run test:health
npm run test:rest
npm run test:prediction
```

---

## 11. Sprint Status & Subsequent Phases

- [x] **Backend Architecture & Express Setup** (11/08/2026 – 10/09/2026) - **Completed**
- [x] **Authentication & JWT Authorization** (11/09/2026 – 17/10/2026) - **Completed**
- [x] **Develop REST API (User & Health Data)** (18/10/2026 – 31/10/2026) - **Completed**
- [x] **AI Prediction API Integration** (01/11/2026 – 30/11/2026) - **Completed & Verified (as of 13/11/2026)**
- [ ] **Health Monitoring Module & Risk-Score Services** (01/12/2026 – 20/12/2026) - **Upcoming**
- [ ] **Health-Risk Alert System** (21/12/2026 – 02/01/2027) - **Upcoming**
- [ ] **Testing & Deployment** (07/02/2027 – 06/03/2027) - **Upcoming**
