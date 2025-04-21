/**
 * Utility for capturing and logging unhandled errors
 */
import { serverLogger } from './server-logger';
import { generateRequestId } from './log-utils';

/**
 * Set up global error handlers for unhandled errors
 */
export function setupUnhandledErrorLogging() {
  // Only run in Node.js environment
  if (typeof process === 'undefined' || typeof process.on !== 'function') {
    return;
  }
  
  // Handle uncaught exceptions
  process.on('uncaughtException', async (error) => {
    const requestId = generateRequestId();
    
    try {
      // Log to console for immediate visibility
      console.error('UNCAUGHT EXCEPTION:', error);
      
      // Prepare context
      const context = {
        type: 'uncaughtException',
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV || 'development',
        runtime: {
          memory: process.memoryUsage ? process.memoryUsage() : null,
          uptime: process.uptime ? process.uptime() : null,
        },
      };
      
      // Log to server logger
      await serverLogger.error(
        `Uncaught Exception: ${error.message}`,
        error,
        context,
        requestId
      );
      
      // Also log a debug entry with the same requestId for correlation
      await serverLogger.debug(
        `Debug context for uncaught exception: ${error.message}`,
        {
          ...context,
          error: {
            message: error.message,
            stack: error.stack,
            name: error.name,
            constructor: error.constructor?.name,
          },
        },
        requestId
      );
    } catch (loggingError) {
      // If logging fails, at least try to log to console
      console.error('Failed to log uncaught exception:', loggingError);
      console.error('Original error:', error);
    }
    
    // In production, we might want to exit the process after an uncaught exception
    if (process.env.NODE_ENV === 'production') {
      // Allow some time for logs to be written
      setTimeout(() => {
        process.exit(1);
      }, 1000);
    }
  });
  
  // Handle unhandled promise rejections
  process.on('unhandledRejection', async (reason, promise) => {
    const requestId = generateRequestId();
    
    try {
      // Log to console for immediate visibility
      console.error('UNHANDLED REJECTION:', reason);
      
      // Convert reason to error if it's not already
      const error = reason instanceof Error ? reason : new Error(String(reason));
      
      // Prepare context
      const context = {
        type: 'unhandledRejection',
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV || 'development',
        runtime: {
          memory: process.memoryUsage ? process.memoryUsage() : null,
          uptime: process.uptime ? process.uptime() : null,
        },
      };
      
      // Log to server logger
      await serverLogger.error(
        `Unhandled Promise Rejection: ${error.message}`,
        error,
        context,
        requestId
      );
      
      // Also log a debug entry with the same requestId for correlation
      await serverLogger.debug(
        `Debug context for unhandled rejection: ${error.message}`,
        {
          ...context,
          error: {
            message: error.message,
            stack: error.stack,
            name: error.name,
            constructor: error.constructor?.name,
          },
        },
        requestId
      );
    } catch (loggingError) {
      // If logging fails, at least try to log to console
      console.error('Failed to log unhandled rejection:', loggingError);
      console.error('Original reason:', reason);
    }
  });
  
  // Log when the process is about to exit
  process.on('exit', (code) => {
    console.log(`Process is about to exit with code: ${code}`);
  });
  
  console.log('Unhandled error logging has been set up');
}