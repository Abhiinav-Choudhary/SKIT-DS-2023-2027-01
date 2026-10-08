const express = require('express');
const healthRecordController = require('../controllers/healthRecord.controller');
const { requireAuth } = require('../middleware/auth.middleware');

const router = express.Router();

router.use(requireAuth);
router.route('/').get(healthRecordController.list).post(healthRecordController.create);
router
  .route('/:id')
  .get(healthRecordController.getById)
  .patch(healthRecordController.update)
  .delete(healthRecordController.remove);

module.exports = router;