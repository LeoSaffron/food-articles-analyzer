#!/usr/bin/env node

/**
 * Simple script to tail log files in real-time
 * 
 * Usage: node scripts/logs/tail.js [access|error|debug]
 */

const fs = require('fs');
const path = require('path');

// Define log paths
const LOG_DIR = path.join(process.cwd(), 'logs');
const ACCESS_LOG = path.join(LOG_DIR, 'access.log');
const DEBUG_LOG = path.join(LOG_DIR, 'debug.log');
const ERROR_LOG = path.join(LOG_DIR, 'error.log');

// Determine which log file to tail
let logFile = ACCESS_LOG; // Default to access.log
const logType = process.argv[2];

if (logType === 'error') {
  logFile = ERROR_LOG;
} else if (logType === 'debug') {
  logFile = DEBUG_LOG;
}

console.log(`Tailing ${logFile}...`);
console.log('Press Ctrl+C to exit');

// Check if log file exists
if (!fs.existsSync(logFile)) {
  console.error(`Log file ${logFile} does not exist`);
  process.exit(1);
}

// Get initial file size
let position = fs.statSync(logFile).size;

// Function to format log entries for better readability
function formatLogEntry(entry) {
  try {
    const data = JSON.parse(entry);
    const timestamp = data.timestamp;
    const level = data.level.toUpperCase();
    const message = data.message;
    
    // Get source IP (either from top level or from data)
    const sourceIp = data.sourceIp || 
                    (data.data && (data.data.sourceIp || data.data.ip)) || 
                    'unknown';
    
    let details = ` from ${sourceIp}`; // Always include source IP
    
    if (data.data) {
      if (data.data.statusCode) {
        details += ` status=${data.data.statusCode}`;
      }
      if (data.data.duration) {
        details += ` duration=${data.data.duration}ms`;
      }
    }
    
    return `${timestamp} [${level}] ${message}${details}`;
  } catch (error) {
    return entry; // Return raw entry if parsing fails
  }
}

// Watch the file for changes
fs.watch(logFile, (eventType) => {
  if (eventType === 'change') {
    const stats = fs.statSync(logFile);
    const newSize = stats.size;
    
    if (newSize > position) {
      // Read only the new data
      const buffer = Buffer.alloc(newSize - position);
      const fileDescriptor = fs.openSync(logFile, 'r');
      
      fs.readSync(fileDescriptor, buffer, 0, newSize - position, position);
      fs.closeSync(fileDescriptor);
      
      // Update position for next read
      position = newSize;
      
      // Process and display new log entries
      const newData = buffer.toString();
      const lines = newData.split('\n').filter(line => line.trim() !== '');
      
      lines.forEach(line => {
        console.log(formatLogEntry(line));
      });
    }
  }
});

// Read and display the last 10 lines of the file initially
fs.readFile(logFile, 'utf8', (err, data) => {
  if (err) {
    console.error(`Error reading ${logFile}:`, err);
    return;
  }
  
  const lines = data.split('\n').filter(line => line.trim() !== '');
  const lastLines = lines.slice(-10); // Get last 10 lines
  
  console.log('Last 10 log entries:');
  lastLines.forEach(line => {
    console.log(formatLogEntry(line));
  });
  console.log('\nWatching for new log entries...');
});