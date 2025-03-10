// db.js
const mongoose = require('mongoose');

// MongoDB connection function
const connectDB = async () => {
  try {
    await mongoose.connect('mongodb://host.docker.internal:27018/foodiesc', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('MongoDB connected successfully');
  } catch (err) {
    console.error('MongoDB connection error:', err);
    process.exit(1);  // Exit process if connection fails
  }
};

module.exports = connectDB;