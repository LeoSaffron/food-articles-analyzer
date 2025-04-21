import { v4 as uuidv4 } from 'uuid';

// Check if we're in a Node.js environment
const isNode = typeof process !== 'undefined' && 
               process.versions != null && 
               process.versions.node != null;

// Check if we're in an Edge runtime
const isEdgeRuntime = typeof EdgeRuntime !== 'undefined';

// Define log paths based on environment
let LOG_DIR = '';
let ACCESS_LOG = '';
let DEBUG_LOG = '';
let ERROR_LOG = '';

// Only import Node.js specific modules if we're in a Node.js environment
if (isNode && !isEdgeRuntime) {
  // Dynamic imports for Node.js environment
  const fs = require('fs');
  const path = require('path');
  
  LOG_DIR = path.join(process.cwd(), 'logs');
  ACCESS_LOG = path.join(LOG_DIR, 'access.log');
  DEBUG_LOG = path.join(LOG_DIR, 'debug.log');
  ERROR_LOG = path.join(LOG_DIR, 'error.log');
} else {
  // For Edge runtime, we'll use these as identifiers only
  LOG_DIR = 'logs';
  ACCESS_LOG = 'access.log';
  DEBUG_LOG = 'debug.log';
  ERROR_LOG = 'error.log';
}

// Export constants
export { LOG_DIR, ACCESS_LOG, DEBUG_LOG, ERROR_LOG };

// Maximum log file size in bytes (5MB)
export const MAX_LOG_SIZE = 5 * 1024 * 1024;

// Maximum number of rotated log files to keep
export const MAX_LOG_FILES = 5;

// Ensure log directory exists (Node.js only)
export function ensureLogDir() {
  if (!isNode || isEdgeRuntime) {
    // Skip in Edge runtime
    return;
  }
  
  try {
    const fs = require('fs');
    const path = require('path');
    
    // Make sure LOG_DIR is properly set
    if (!LOG_DIR) {
      LOG_DIR = path.join(process.cwd(), 'logs');
      ACCESS_LOG = path.join(LOG_DIR, 'access.log');
      DEBUG_LOG = path.join(LOG_DIR, 'debug.log');
      ERROR_LOG = path.join(LOG_DIR, 'error.log');
    }
    
    // Create directory if it doesn't exist
    if (!fs.existsSync(LOG_DIR)) {
      console.log(`Creating log directory: ${LOG_DIR}`);
      fs.mkdirSync(LOG_DIR, { recursive: true, mode: 0o777 }); // More permissive mode
    } else {
      // Ensure directory is writable
      fs.chmodSync(LOG_DIR, 0o777); // More permissive mode
    }
    
    // Create empty log files if they don't exist and ensure they're writable
    [ACCESS_LOG, DEBUG_LOG, ERROR_LOG].forEach(logFile => {
      if (!fs.existsSync(logFile)) {
        console.log(`Creating log file: ${logFile}`);
        fs.writeFileSync(logFile, '', { mode: 0o666 }); // More permissive mode
      } else {
        // Ensure file is writable
        fs.chmodSync(logFile, 0o666); // More permissive mode
      }
    });

    return true; // Successfully ensured log directory and files
  } catch (error) {
    console.error('Failed to create log directory or files:', error);
    return false; // Failed to ensure log directory and files
  }
}

// Rotate log file if it exceeds the maximum size (Node.js only)
export function rotateLogFileIfNeeded(logFile: string) {
  if (!isNode || isEdgeRuntime) {
    // Skip in Edge runtime
    return;
  }
  
  try {
    const fs = require('fs');
    if (!fs.existsSync(logFile)) {
      return;
    }

    const stats = fs.statSync(logFile);
    if (stats.size < MAX_LOG_SIZE) {
      return;
    }

    // Rotate existing backup logs
    for (let i = MAX_LOG_FILES - 1; i > 0; i--) {
      const oldFile = `${logFile}.${i}`;
      const newFile = `${logFile}.${i + 1}`;
      if (fs.existsSync(oldFile)) {
        if (i === MAX_LOG_FILES - 1) {
          // Delete the oldest log file
          fs.unlinkSync(oldFile);
        } else {
          // Rename to next number
          fs.renameSync(oldFile, newFile);
        }
      }
    }

    // Rename current log to .1
    fs.renameSync(logFile, `${logFile}.1`);
    
    // Create a new empty log file with proper permissions
    fs.writeFileSync(logFile, '', { mode: 0o644 });
  } catch (error) {
    console.error(`Failed to rotate log file ${logFile}:`, error);
  }
}

// Generate a unique request ID (works in all environments)
export function generateRequestId(): string {
  // Use uuid if available, otherwise fallback to a simple random ID
  try {
    return uuidv4();
  } catch (error) {
    return Math.random().toString(36).substring(2, 15) + 
           Math.random().toString(36).substring(2, 15);
  }
}

// Format log entry as JSON (works in all environments)
export function formatLogEntry(level: string, message: string, data?: any, requestId?: string, sourceIp?: string): string {
  const timestamp = new Date().toISOString();
  const processInfo = isNode ? {
    pid: process.pid,
    ppid: process.ppid,
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
    memory: process.memoryUsage ? process.memoryUsage() : null,
  } : {};
  
  const logData = {
    timestamp,
    level,
    message,
    ...(sourceIp && { sourceIp }), // Add source IP at the top level if provided
    ...(requestId && { requestId }),
    ...(data && { data }),
    process: processInfo,
    environment: process.env.NODE_ENV || 'unknown',
  };
  
  return JSON.stringify(logData) + '\n';
}