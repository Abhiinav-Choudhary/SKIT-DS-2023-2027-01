const express = require('express');
const { requireAuth } = require('../middleware/auth.middleware');

const router = express.Router();

/**
 * Protected test route to verify authentication flow
 * GET /api/test/protected
 */
router.get('/protected', requireAuth, (req, res) => {
  res.status(200).json({
    success: true,
    message: 'Authenticated successfully',
  });
});

module.exports = router;
