import path from 'path';

export const LOG_CONFIG = {
  dir: path.join(process.cwd(), 'logs'),
  files: {
    access: 'access.log',
    debug: 'debug.log',
    error: 'error.log'
  },
  format: {
    timestamp: true,
    color: process.env.NODE_ENV === 'development'
  }
};