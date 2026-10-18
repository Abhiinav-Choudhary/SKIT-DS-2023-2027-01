const User = require('../models/User');
const HealthRecord = require('../models/HealthRecord');
const Prediction = require('../models/Prediction');

const ALLOWED_PROFILE_UPDATES = [
  'name',
  'age',
  'gender',
  'bloodGroup',
  'phone',
  'heightCm',
  'weightKg',
];

/**
 * Retrieve user profile by ID
 */
const getUserProfile = async (userId) => {
  const user = await User.findById(userId);
  if (!user) {
    const error = new Error('User not found');
    error.statusCode = 404;
    throw error;
  }
  return user;
};

/**
 * Update user profile details
 */
const updateUserProfile = async (userId, updateData) => {
  if (!updateData || typeof updateData !== 'object') {
    const error = new Error('Provide valid profile update fields');
    error.statusCode = 400;
    throw error;
  }

  const updates = {};
  for (const field of ALLOWED_PROFILE_UPDATES) {
    if (updateData[field] !== undefined) {
      updates[field] = typeof updateData[field] === 'string' ? updateData[field].trim() : updateData[field];
    }
  }

  if (Object.keys(updates).length === 0) {
    const error = new Error('Provide at least one allowed profile field to update');
    error.statusCode = 400;
    throw error;
  }

  const user = await User.findByIdAndUpdate(userId, updates, {
    new: true,
    runValidators: true,
  });

  if (!user) {
    const error = new Error('User not found');
    error.statusCode = 404;
    throw error;
  }

  return user;
};

/**
 * Change authenticated user's password
 */
const changePassword = async (userId, { currentPassword, newPassword }) => {
  if (!currentPassword || !newPassword) {
    const error = new Error('Please provide both current and new passwords');
    error.statusCode = 400;
    throw error;
  }

  if (newPassword.length < 6) {
    const error = new Error('New password must be at least 6 characters long');
    error.statusCode = 400;
    throw error;
  }

  const user = await User.findById(userId).select('+password');
  if (!user) {
    const error = new Error('User not found');
    error.statusCode = 404;
    throw error;
  }

  const isMatch = await user.comparePassword(currentPassword);
  if (!isMatch) {
    const error = new Error('Current password is incorrect');
    error.statusCode = 400;
    throw error;
  }

  user.password = newPassword;
  await user.save();

  return { message: 'Password changed successfully' };
};

/**
 * Delete user account and cascade delete their health records
 */
const deleteUserAccount = async (userId) => {
  const user = await User.findByIdAndDelete(userId);
  if (!user) {
    const error = new Error('User not found');
    error.statusCode = 404;
    throw error;
  }

  // Cascade delete associated health records and prediction records
  await HealthRecord.deleteMany({ user: userId });
  if (Prediction && typeof Prediction.deleteMany === 'function') {
    try {
      await Prediction.deleteMany({ user: userId });
    } catch {
      // In-memory or disconnected fallback
    }
  }

  return { message: 'Account and associated health data deleted successfully' };
};

module.exports = {
  getUserProfile,
  updateUserProfile,
  changePassword,
  deleteUserAccount,
};
