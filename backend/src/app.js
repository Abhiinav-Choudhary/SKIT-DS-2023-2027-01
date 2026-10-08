const express = require('express');
const cors = require('cors');
const authRoutes = require('./routes/auth.routes');
const testRoutes = require('./routes/test.routes');
const { errorHandler, notFoundHandler } = require('./middleware/error.middleware');

const app = express();

// Standard middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Root health status endpoint
app.get('/', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'AI-Powered Health Risk Prediction and Monitoring System Backend API is active',
  });
});

// Mount modular API routes
app.use('/api/auth', authRoutes);
app.use('/api/test', testRoutes);

// 404 Catch-all route handler
app.use(notFoundHandler);

// Centralized error handling middleware
app.use(errorHandler);

module.exports = app;
