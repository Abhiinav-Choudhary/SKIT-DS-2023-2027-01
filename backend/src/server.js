const dotenv = require('dotenv');

// Load environment variables before importing modules that depend on them
dotenv.config();

const app = require('./app');
const connectDB = require('./config/db');

const PORT = process.env.PORT || 5000;

// Connect to MongoDB Atlas and start Express server
connectDB()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`[Server] Running in ${process.env.NODE_ENV || 'development'} mode on port ${PORT}`);
    });
  })
  .catch((err) => {
    console.error(`[Server Error] Failed to start server: ${err.message}`);
    process.exit(1);
  });
