/**
 * Global error handler for the application
 * This utility provides functions to capture and log errors from anywhere in the app
 */
import { serverLogger } from './server-logger';
import { generateRequestId } from './log-utils';

/**
 * Log an error with full context
 */
export async function logError(
  error: Error | unknown,
  context?: {
    message?: string;
    source?: string;
    url?: string;
    userId?: string;
    requestId?: string;
    [key: string]: any;
  }
): Promise<string> {
  // Generate a request ID if not provided
  const requestId = context?.requestId || generateRequestId();
  
  // Format the error message
  const message = context?.message || 
                 (error instanceof Error ? error.message : 'Unknown error');
  
  // Prepare the error object
  const errorObj = error instanceof Error ? error : new Error(String(error));
  
  // Prepare the context data
  const fullContext = {
    ...context,
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    runtime: {
      memory: process.memoryUsage ? process.memoryUsage() : null,
    },
  };
  
  // Log to console for immediate visibility
  console.error(`[ERROR] ${message}`, {
    error: errorObj.message,
    stack: errorObj.stack,
    source: context?.source,
    requestId,
  });
  
  // Log to server logger with full debug information
  await serverLogger.error(
    `${context?.source ? `[${context.source}] ` : ''}${message}`,
    errorObj,
    fullContext,
    requestId
  );
  
  // Always log a debug entry with the same requestId for correlation
  await serverLogger.debug(
    `Debug context for error: ${message}`,
    {
      ...fullContext,
      error: {
        message: errorObj.message,
        stack: errorObj.stack,
        name: errorObj.name,
        constructor: errorObj.constructor?.name,
      },
    },
    requestId
  );
  
  return requestId;
}

/**
 * Create a wrapped function that logs any errors
 */
export function withErrorLogging<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options?: {
    source?: string;
    rethrow?: boolean;
    onError?: (error: Error, requestId: string) => void;
  }
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  return async (...args: Parameters<T>): Promise<ReturnType<T>> => {
    try {
      return await fn(...args);
    } catch (error) {
      // Generate a meaningful message
      const message = `Error in ${options?.source || fn.name || 'anonymous function'}`;
      
      // Log the error
      const requestId = await logError(error, {
        message,
        source: options?.source,
        args: args.map(arg => 
          // Sanitize arguments to avoid logging sensitive data
          typeof arg === 'object' ? 
            (arg instanceof Error ? { message: arg.message, name: arg.name } : 'object') : 
            String(arg)
        ),
      });
      
      // Call the onError callback if provided
      if (options?.onError) {
        options.onError(error instanceof Error ? error : new Error(String(error)), requestId);
      }
      
      // Rethrow the error if specified
      if (options?.rethrow !== false) {
        throw error;
      }
      
      // Return a default value if not rethrowing
      return undefined as unknown as ReturnType<T>;
    }
  };
}

/**
 * Log an application issue (not necessarily an error)
 */
export async function logApplicationIssue(
  message: string,
  details: any,
  options?: {
    level?: 'debug' | 'info' | 'warn' | 'error';
    source?: string;
    requestId?: string;
  }
): Promise<string> {
  // Generate a request ID if not provided
  const requestId = options?.requestId || generateRequestId();
  
  // Determine the log level
  const level = options?.level || 'warn';
  
  // Prepare the context data
  const context = {
    source: options?.source,
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
  };
  
  // Log to console for immediate visibility
  console[level](`[${level.toUpperCase()}] ${message}`, {
    details,
    source: options?.source,
    requestId,
  });
  
  // Log to server logger
  if (level === 'error') {
    await serverLogger.error(message, undefined, { ...details, context }, requestId);
  } else {
    await serverLogger.log(level, message, { ...details, context }, requestId);
  }
  
  // Always log a debug entry with the same requestId for correlation
  await serverLogger.debug(
    `Debug context for ${level}: ${message}`,
    { ...details, context },
    requestId
  );
  
  return requestId;
}