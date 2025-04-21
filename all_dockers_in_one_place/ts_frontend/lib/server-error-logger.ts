/**
 * Utility for capturing and logging server-side errors
 */
import { NextRequest } from 'next/server';
import { serverLogger } from './server-logger';
import { generateRequestId } from './log-utils';

/**
 * Log a server-side error with full context
 */
export async function logServerError(
  error: Error | unknown,
  request?: NextRequest,
  context?: Record<string, any>
): Promise<string> {
  // Generate a request ID or use the one from the request
  const requestId = request?.headers.get('X-Request-ID') || generateRequestId();
  
  // Extract request information if available
  const requestInfo = request ? {
    url: request.url,
    method: request.method,
    path: new URL(request.url).pathname,
    query: Object.fromEntries(new URL(request.url).searchParams.entries()),
    headers: Object.fromEntries(request.headers.entries()),
    sourceIp: request.headers.get('x-forwarded-for') || 
             request.headers.get('x-real-ip') || 
             request.ip || 
             'unknown',
  } : undefined;
  
  // Prepare the error object
  const errorObj = error instanceof Error ? error : new Error(String(error));
  
  // Prepare the full context
  const fullContext = {
    ...context,
    request: requestInfo,
    environment: {
      nodeEnv: process.env.NODE_ENV,
      timestamp: new Date().toISOString(),
    },
    runtime: {
      memory: process.memoryUsage ? process.memoryUsage() : null,
      uptime: process.uptime ? process.uptime() : null,
    },
  };
  
  // Log to console for immediate visibility
  console.error(`Server Error: ${errorObj.message}`, {
    path: requestInfo?.path,
    method: requestInfo?.method,
    error: errorObj.message,
    stack: errorObj.stack,
    requestId,
  });
  
  // Log to server logger with full debug information
  await serverLogger.error(
    `Server Error: ${errorObj.message}`,
    errorObj,
    fullContext,
    requestId
  );
  
  // Always log a debug entry with the same requestId for correlation
  await serverLogger.debug(
    `Debug context for server error: ${errorObj.message}`,
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
 * Create a wrapped function that logs any server-side errors
 */
export function withServerErrorLogging<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options?: {
    source?: string;
    rethrow?: boolean;
    getRequest?: (...args: Parameters<T>) => NextRequest | undefined;
    getContext?: (...args: Parameters<T>) => Record<string, any> | undefined;
  }
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  return async (...args: Parameters<T>): Promise<ReturnType<T>> => {
    try {
      return await fn(...args);
    } catch (error) {
      // Extract request if available
      const request = options?.getRequest ? options.getRequest(...args) : undefined;
      
      // Extract additional context if available
      const context = {
        source: options?.source || fn.name || 'server-function',
        ...(options?.getContext ? options.getContext(...args) : {}),
      };
      
      // Log the error
      await logServerError(error, request, context);
      
      // Rethrow the error if specified
      if (options?.rethrow !== false) {
        throw error;
      }
      
      // Return a default value if not rethrowing
      return undefined as unknown as ReturnType<T>;
    }
  };
}