const express = require('express');
const healthRecordController = require('../controllers/healthRecord.controller');
const { requireAuth } = require('../middleware/auth.middleware');

const router = express.Router();

// Require JWT authentication for all health record operations
router.use(requireAuth);

// Collection routes
router
  .route('/')
  .post(healthRecordController.create)
  .get(healthRecordController.list);

// Analytics & vitals summary
router.get('/summary', healthRecordController.getSummary);

// Single record item routes
router
  .route('/:id')
  .get(healthRecordController.getById)
  .patch(healthRecordController.update)
  .delete(healthRecordController.remove);

module.exports = router;
