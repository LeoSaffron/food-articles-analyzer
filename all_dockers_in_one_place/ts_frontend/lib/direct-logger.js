/**
 * Direct file system logger that doesn't rely on imports
 * This is used as a fallback when other logging methods fail
 */

const fs = require('fs');
const path = require('path');

// Define log paths
const LOG_DIR = path.join(process.cwd(), 'logs');
const ACCESS_LOG = path.join(LOG_DIR, 'access.log');
const DEBUG_LOG = path.join(LOG_DIR, 'debug.log');
const ERROR_LOG = path.join(LOG_DIR, 'error.log');

// Ensure log directory exists
function ensureLogDir() {
  try {
    if (!fs.existsSync(LOG_DIR)) {
      fs.mkdirSync(LOG_DIR, { recursive: true, mode: 0o777 });
    } else {
      fs.chmodSync(LOG_DIR, 0o777);
    }
    return true;
  } catch (error) {
    console.error('Failed to create log directory:', error);
    return false;
  }
}

// Format log entry
function formatLogEntry(level, message, data, requestId) {
  const timestamp = new Date().toISOString();
  const logData = {
    timestamp,
    level,
    message,
  };
  
  if (requestId) {
    logData.requestId = requestId;
  }
  
  if (data) {
    logData.data = data;
  }
  
  return JSON.stringify(logData) + '\n';
}

// Write log entry
function writeLog(level, message, data, requestId) {
  try {
    // Ensure log directory exists
    ensureLogDir();
    
    // Determine log file
    const logFile = level === 'error' ? ERROR_LOG : 
                   level === 'debug' ? DEBUG_LOG : ACCESS_LOG;
    
    // Create log file if it doesn't exist
    if (!fs.existsSync(logFile)) {
      fs.writeFileSync(logFile, '', { mode: 0o666 });
    } else {
      fs.chmodSync(logFile, 0o666);
    }
    
    // Format log entry
    const logEntry = formatLogEntry(level, message, data, requestId);
    
    // Write to log file
    fs.appendFileSync(logFile, logEntry, { encoding: 'utf8' });
    
    // Also log to console
    console.log(`[DIRECT-LOGGER] ${level.toUpperCase()} ${message}`);
    
    return true;
  } catch (error) {
    console.error('Direct logger failed:', error);
    return false;
  }
}

module.exports = {
  writeLog,
  ensureLogDir,
  ACCESS_LOG,
  DEBUG_LOG,
  ERROR_LOG
};