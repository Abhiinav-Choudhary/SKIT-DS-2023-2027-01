const jwt = require('jsonwebtoken');

/**
 * Sign a new JWT token with the provided payload
 * @param {Object} payload - Data to embed in token (e.g. { id: user._id })
 * @returns {string} - Signed JWT
 */
const generateToken = (payload) => {
  const secret = process.env.JWT_SECRET;
  const expiresIn = process.env.JWT_EXPIRES_IN || '7d';

  if (!secret) {
    throw new Error('JWT_SECRET is not configured in environment variables');
  }

  return jwt.sign(payload, secret, { expiresIn });
};

/**
 * Verify and decode an existing JWT token
 * @param {string} token - JWT string to verify
 * @returns {Object} - Decoded payload
 */
const verifyToken = (token) => {
  const secret = process.env.JWT_SECRET;

  if (!secret) {
    throw new Error('JWT_SECRET is not configured in environment variables');
  }

  return jwt.verify(token, secret);
};

module.exports = {
  generateToken,
  verifyToken,
};
