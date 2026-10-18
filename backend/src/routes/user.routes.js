const express = require('express');
const userController = require('../controllers/user.controller');
const { requireAuth } = require('../middleware/auth.middleware');

const router = express.Router();

// All user management routes require JWT authentication
router.use(requireAuth);

router.get('/profile', userController.getProfile);
router.patch('/profile', userController.updateProfile);
router.put('/change-password', userController.changePassword);
router.delete('/profile', userController.deleteAccount);

module.exports = router;
