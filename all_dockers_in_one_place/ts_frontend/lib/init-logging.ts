/**
 * Initialize all logging systems
 */
import { setupUnhandledErrorLogging } from './unhandled-error-logger';
import { scheduleResourceChecks } from './resource-monitor';
import { ensureLogDir } from './log-utils';

/**
 * Initialize all logging systems
 */
export function initLogging() {
  // Only run in Node.js environment
  if (typeof process === 'undefined' || typeof process.on !== 'function') {
    return;
  }
  
  console.log('Initializing logging systems...');
  
  // Ensure log directory exists
  ensureLogDir();
  
  // Set up unhandled error logging
  setupUnhandledErrorLogging();
  
  // Schedule resource checks (every 5 minutes)
  scheduleResourceChecks(300);
  
  // Force debug logging to be enabled
  if (!process.env.ENABLE_DEBUG) {
    process.env.ENABLE_DEBUG = 'true';
    console.log('Debug logging has been enabled');
  }
  
  console.log('Logging systems initialized');
}

// Auto-initialize in server environment
if (typeof window === 'undefined') {
  initLogging();
}