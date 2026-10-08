const authService = require('../services/auth.service');

/**
 * Handle user registration
 * POST /api/auth/register
 */
const register = async (req, res, next) => {
  try {
    const { name, email, password } = req.body;
    const user = await authService.registerUser({ name, email, password });

    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      data: { user },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle user login
 * POST /api/auth/login
 */
const login = async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const result = await authService.loginUser({ email, password });

    res.status(200).json({
      success: true,
      message: 'Login successful',
      data: result,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Get current authenticated user profile
 * GET /api/auth/me
 */
const getMe = async (req, res, next) => {
  try {
    res.status(200).json({
      success: true,
      message: 'Current user retrieved successfully',
      data: { user: req.user },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Handle user logout
 * POST /api/auth/logout
 * Note: Since JWT is stateless, logout informs the client to discard the stored token.
 */
const logout = async (req, res) => {
  res.status(200).json({
    success: true,
    message: 'Logged out successfully. Please clear the authentication token on the client.',
  });
};

module.exports = {
  register,
  login,
  getMe,
  logout,
};
