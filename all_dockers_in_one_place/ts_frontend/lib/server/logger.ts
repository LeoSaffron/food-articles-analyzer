import fs from 'fs';
import { LOG_CONFIG } from '../config/logging';

// Ensure log directory exists
if (!fs.existsSync(LOG_CONFIG.dir)) {
  fs.mkdirSync(LOG_CONFIG.dir, { recursive: true });
}

function formatLogEntry(level: string, message: string, data?: any): string {
  const timestamp = new Date().toISOString();
  const dataStr = data ? `\n${JSON.stringify(data, null, 2)}` : '';
  return `[${timestamp}] [${level}] ${message}${dataStr}\n`;
}

function writeLog(filename: string, entry: string) {
  const logPath = `${LOG_CONFIG.dir}/${filename}`;
  fs.appendFileSync(logPath, entry, { encoding: 'utf8', mode: 0o644 });
  
  // Also log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.log(entry);
  }
}

export const serverLogger = {
  access: (message: string, data?: any) => {
    const entry = formatLogEntry('ACCESS', message, data);
    writeLog(LOG_CONFIG.files.access, entry);
  },

  debug: (message: string, data?: any) => {
    if (process.env.ENABLE_DEBUG === 'true') {
      const entry = formatLogEntry('DEBUG', message, data);
      writeLog(LOG_CONFIG.files.debug, entry);
    }
  },

  error: (message: string, error?: Error, data?: any) => {
    const errorData = error ? {
      message: error.message,
      stack: error.stack,
      ...data
    } : data;
    const entry = formatLogEntry('ERROR', message, errorData);
    writeLog(LOG_CONFIG.files.error, entry);
  }
};