const express = require('express');
const cors = require('cors');
const { errorHandler, notFoundHandler } = require('./middleware/error.middleware');

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'AI-Powered Health Risk Prediction and Monitoring System Backend API is active',
  });
});

app.use(notFoundHandler);
app.use(errorHandler);

module.exports = app;
