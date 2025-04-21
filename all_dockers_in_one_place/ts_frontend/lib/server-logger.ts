import {
  ACCESS_LOG,
  DEBUG_LOG,
  ERROR_LOG,
  ensureLogDir,
  rotateLogFileIfNeeded,
  formatLogEntry
} from './log-utils';

// Check if we're in a Node.js environment
const isNode = typeof process !== 'undefined' && 
               process.versions != null && 
               process.versions.node != null;

// Check if we're in an Edge runtime
const isEdgeRuntime = typeof EdgeRuntime !== 'undefined';

// Ensure log directory exists (only in Node.js environment)
const logsReady = ensureLogDir();

// In-memory log storage for Edge runtime
const memoryLogs: Record<string, string[]> = {
  access: [],
  debug: [],
  error: []
};

// Maximum number of in-memory logs to keep
const MAX_MEMORY_LOGS = 1000;

/**
 * Write log to appropriate destination based on environment
 */
export async function writeToLog(logFile: string, level: string, message: string, data?: any, requestId?: string) {
  try {
    // Format log entry as JSON
    const logEntry = formatLogEntry(level, message, data, requestId);
    
    // Always log to console
    console.log(`[${level.toUpperCase()}] ${message}`, data ? JSON.stringify(data) : '');

    // In Node.js environment, write to file
    if (isNode && !isEdgeRuntime) {
      // Direct file system access for more reliable logging
      try {
        const fs = require('fs');
        
        // Ensure the log directory exists before writing
        if (!logsReady) {
          ensureLogDir();
        }
        
        // Check if log rotation is needed
        rotateLogFileIfNeeded(logFile);
        
        // Write to log file using synchronous method
        fs.appendFileSync(logFile, logEntry, { encoding: 'utf8', mode: 0o666 });
        return true; // Successfully wrote log
      } catch (fsError) {
        console.error('Failed to write log:', fsError);
        return false; // Failed to write log
      }
    } 
    // In Edge runtime, store in memory (with limit)
    else if (isEdgeRuntime) {
      const logType = logFile.includes('error') ? 'error' : 
                     logFile.includes('debug') ? 'debug' : 'access';
      
      memoryLogs[logType].push(logEntry);
      
      // Limit the size of in-memory logs
      if (memoryLogs[logType].length > MAX_MEMORY_LOGS) {
        memoryLogs[logType].shift();
      }
      return true; // Successfully stored log in memory
    }
    
    return false; // No logging method available
  } catch (error) {
    console.error('Failed to write to log:', error);
    return false; // Failed to write log
  }
}

/**
 * Get in-memory logs (useful for Edge runtime)
 */
export function getMemoryLogs(type: 'access' | 'debug' | 'error'): string[] {
  return memoryLogs[type] || [];
}

/**
 * Direct log writing function that doesn't use async/await
 * This is more reliable in some environments
 */
export function writeLogSync(level: string, message: string, data?: any, requestId?: string, sourceIp?: string) {
  if (!isNode || isEdgeRuntime) {
    console.log(`[${level.toUpperCase()}] ${message}`, data ? JSON.stringify(data) : '');
    return;
  }
  
  try {
    const fs = require('fs');
    const logFile = level === 'error' ? ERROR_LOG : 
                   level === 'debug' ? DEBUG_LOG : ACCESS_LOG;
    
    // Format log entry
    const logEntry = formatLogEntry(level, message, data, requestId, sourceIp);
    
    // Write directly to file
    fs.appendFileSync(logFile, logEntry, { encoding: 'utf8', mode: 0o666 });
    
    // Also log to console
    console.log(`[${level.toUpperCase()}] ${message}${sourceIp ? ` from ${sourceIp}` : ''}`, data ? JSON.stringify(data) : '');
  } catch (error) {
    console.error('Failed to write log synchronously:', error);
  }
}

/**
 * Server logger that works in both Node.js and Edge runtime
 */
export const serverLogger = {
  access: async (message: string, data?: any, requestId?: string) => {
    // Try sync method first
    writeLogSync('access', message, data, requestId);
    // Also try async method as backup
    return await writeToLog(ACCESS_LOG, 'access', message, data, requestId);
  },
  
  debug: async (message: string, data?: any, requestId?: string) => {
    // Always log debug information regardless of ENABLE_DEBUG setting
    // This ensures we capture everything for troubleshooting
    
    // Try sync method first
    writeLogSync('debug', message, data, requestId);
    // Also try async method as backup
    return await writeToLog(DEBUG_LOG, 'debug', message, data, requestId);
  },

  error: async (message: string, error?: Error, data?: any, requestId?: string) => {
    const errorData = error ? {
      message: error.message,
      stack: error.stack,
      name: error.name,
      constructor: error.constructor?.name,
      ...data
    } : data;
    
    // Try sync method first
    writeLogSync('error', message, errorData, requestId);
    // Also try async method as backup
    return await writeToLog(ERROR_LOG, 'error', message, errorData, requestId);
  },
  
  // Log with custom level
  log: async (level: string, message: string, data?: any, requestId?: string) => {
    const logFile = level === 'error' ? ERROR_LOG : 
                   level === 'debug' ? DEBUG_LOG : ACCESS_LOG;
    
    // Try sync method first
    writeLogSync(level, message, data, requestId);
    // Also try async method as backup
    return await writeToLog(logFile, level, message, data, requestId);
  }
};