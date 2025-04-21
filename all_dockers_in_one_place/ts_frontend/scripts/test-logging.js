#!/usr/bin/env node

/**
 * Test script to verify logging functionality
 * 
 * Usage: node scripts/test-logging.js
 */

const fs = require('fs');
const path = require('path');

// Define log paths
const LOG_DIR = path.join(process.cwd(), 'logs');
const ACCESS_LOG = path.join(LOG_DIR, 'access.log');
const DEBUG_LOG = path.join(LOG_DIR, 'debug.log');
const ERROR_LOG = path.join(LOG_DIR, 'error.log');

// Ensure log directory exists
if (!fs.existsSync(LOG_DIR)) {
  console.log(`Creating log directory: ${LOG_DIR}`);
  fs.mkdirSync(LOG_DIR, { recursive: true, mode: 0o755 });
}

// Test writing to log files directly
function writeTestLog() {
  const timestamp = new Date().toISOString();
  const testEntry = JSON.stringify({
    timestamp,
    level: 'test',
    message: 'Test log entry',
    data: { test: true }
  }) + '\n';
  
  console.log('Writing test log entries...');
  
  try {
    fs.appendFileSync(ACCESS_LOG, testEntry, { encoding: 'utf8', mode: 0o644 });
    console.log(`Successfully wrote to ${ACCESS_LOG}`);
  } catch (error) {
    console.error(`Failed to write to ${ACCESS_LOG}:`, error);
  }
  
  try {
    fs.appendFileSync(ERROR_LOG, testEntry, { encoding: 'utf8', mode: 0o644 });
    console.log(`Successfully wrote to ${ERROR_LOG}`);
  } catch (error) {
    console.error(`Failed to write to ${ERROR_LOG}:`, error);
  }
}

// Check if log files exist and are writable
function checkLogFiles() {
  console.log('Checking log files...');
  
  [ACCESS_LOG, DEBUG_LOG, ERROR_LOG].forEach(logFile => {
    try {
      if (fs.existsSync(logFile)) {
        const stats = fs.statSync(logFile);
        console.log(`${logFile} exists, size: ${stats.size} bytes, mode: ${stats.mode.toString(8)}`);
        
        // Check if file is writable
        fs.accessSync(logFile, fs.constants.W_OK);
        console.log(`${logFile} is writable`);
      } else {
        console.log(`${logFile} does not exist, creating it...`);
        fs.writeFileSync(logFile, '', { mode: 0o644 });
        console.log(`Created ${logFile}`);
      }
    } catch (error) {
      console.error(`Error with ${logFile}:`, error);
    }
  });
}

// Check permissions of log directory
function checkLogDirectory() {
  console.log('Checking log directory...');
  
  try {
    const stats = fs.statSync(LOG_DIR);
    console.log(`${LOG_DIR} exists, mode: ${stats.mode.toString(8)}`);
    
    // Check if directory is writable
    fs.accessSync(LOG_DIR, fs.constants.W_OK);
    console.log(`${LOG_DIR} is writable`);
  } catch (error) {
    console.error(`Error with ${LOG_DIR}:`, error);
  }
}

// Run tests
checkLogDirectory();
checkLogFiles();
writeTestLog();

console.log('\nLogging test complete. Check the console output above for any errors.');