# AI-Powered Health Risk Prediction and Monitoring System - Backend API

Backend service providing RESTful architecture, MongoDB connectivity, and JWT-based authentication for the AI-Powered Health Risk Prediction and Monitoring System.

---

## 1. Project Overview & Current Phase

- **Project ID**: SKIT/DS/2023-2027/01
- **Current Development Phase**: Backend Architecture + Authentication & Authorization (Phase 1)
- **Status**: Implemented & Verified
- **Assigned Lead**: Abhinav Kumar Chaudhary (Backend & API Architecture)

*Note: Future components (Health Records CRUD, AI Prediction Services, Health Monitoring, Alert Systems) belong to subsequent project phases and are not yet implemented.*

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
│   │   └── db.js               # MongoDB Mongoose connection
│   ├── controllers/
│   │   └── auth.controller.js  # HTTP request/response handlers
│   ├── middleware/
│   │   ├── auth.middleware.js  # JWT verification (requireAuth)
│   │   └── error.middleware.js # Centralized error & 404 handler
│   ├── models/
│   │   └── User.js             # User Mongoose schema with bcrypt hashing
│   ├── routes/
│   │   ├── auth.routes.js      # Auth endpoints (/api/auth)
│   │   └── test.routes.js      # Protected test endpoint (/api/test)
│   ├── services/
│   │   └── auth.service.js     # Business logic & DB transactions
│   ├── utils/
│   │   └── jwt.js              # Token signing & verification
│   ├── app.js                  # Express application setup
│   └── server.js               # Application entry point
├── tests/
│   └── auth.test.js            # Automated test suite (11 scenarios)
├── .env.example                # Sample environment variables
├── .gitignore                  # Git ignore rules
├── package.json                # Project dependencies and scripts
└── README.md                   # Backend documentation
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

The repository includes a comprehensive automated test suite verifying all 11 required scenarios:
1. Successful registration
2. Duplicate registration (returns 409)
3. Invalid registration data (returns 400)
4. Successful login
5. Incorrect password rejection (returns 401)
6. Non-existent user rejection (returns 401)
7. `GET /api/auth/me` with valid JWT
8. `GET /api/auth/me` without JWT (returns 401)
9. `GET /api/auth/me` with invalid JWT (returns 401)
10. Protected test route validation (`/api/test/protected`)
11. Logout response confirmation

Run the tests with:
```bash
npm test
```

---

## 11. Planned Work (Subsequent Phases)

- **REST API Development** (18/10/2026 – 31/10/2026): Health record schemas, CRUD endpoints.
- **AI Prediction API Integration** (01/11/2026 – 30/11/2026): Connect ML/DL models with backend.
- **Health Monitoring Module** (01/12/2026 – 20/12/2026): Health monitoring services and risk-score tracking.
- **Health-Risk Alert System** (21/12/2026 – 02/01/2027): Anomaly threshold detection and notifications.
- **Testing & Deployment** (07/02/2027 – 06/03/2027): End-to-end integration and Docker deployment.
