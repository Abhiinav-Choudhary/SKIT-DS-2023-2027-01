const express = require('express');
const predictionController = require('../controllers/prediction.controller');
const { requireAuth } = require('../middleware/auth.middleware');

const router = express.Router();

// Require JWT authentication for all AI prediction operations
router.use(requireAuth);

// Prediction generation endpoints
router.post('/diabetes', predictionController.predictDiabetes);
router.post('/heart', predictionController.predictHeart);

// Prediction history management endpoints
router.get('/', predictionController.list);
router
  .route('/:id')
  .get(predictionController.getById)
  .delete(predictionController.remove);

module.exports = router;

