/**
 * Utility for capturing and logging API-specific issues
 */
import { NextRequest } from 'next/server';
import { serverIssueLogger } from './server-issue-logger';

// Interface for API request context
interface ApiRequestContext {
  request: NextRequest;
  endpoint: string;
  method: string;
  requestId?: string;
  userId?: string;
  startTime?: number;
}

// Log API request issue
export function logApiIssue(message: string, context: ApiRequestContext, data?: any, error?: Error) {
  // Extract request information
  const { request, endpoint, method, requestId, userId, startTime } = context;
  
  // Calculate request duration if startTime is provided
  const duration = startTime ? Date.now() - startTime : undefined;
  
  // Extract headers and query parameters
  const headers = Object.fromEntries(request.headers.entries());
  const url = new URL(request.url);
  const queryParams = Object.fromEntries(url.searchParams.entries());
  
  // Get client IP
  const clientIp = request.headers.get('x-forwarded-for') || 
                  request.headers.get('x-real-ip') || 
                  request.ip || 
                  'unknown';
  
  // Prepare request context
  const requestContext = {
    endpoint,
    method,
    url: request.url,
    path: url.pathname,
    queryParams,
    headers: {
      // Include only safe headers
      'user-agent': headers['user-agent'],
      'content-type': headers['content-type'],
      'accept': headers['accept'],
      'referer': headers['referer'],
      'x-request-id': headers['x-request-id'],
    },
    clientIp,
    userId,
    duration,
  };
  
  // Log the issue
  return serverIssueLogger.medium(
    message,
    'api',
    { ...data, request: requestContext },
    error,
    { apiEndpoint: endpoint },
    requestId || headers['x-request-id']
  );
}

// Log API validation error
export function logApiValidationIssue(message: string, context: ApiRequestContext, validationErrors: any) {
  return logApiIssue(
    `Validation error: ${message}`,
    context,
    { validationErrors },
    new Error('API validation failed')
  );
}

// Log API performance issue
export function logApiPerformanceIssue(context: ApiRequestContext, duration: number, threshold: number) {
  const { endpoint, method, requestId } = context;
  
  return serverIssueLogger.performance(
    `Slow API request to ${method} ${endpoint}`,
    'api_performance',
    { duration, endpoint, method },
    { duration: threshold },
    requestId
  );
}

// API request wrapper that logs issues
export async function withApiIssueLogging<T>(
  context: ApiRequestContext,
  handler: () => Promise<T>,
  options?: { performanceThreshold?: number }
): Promise<T> {
  const startTime = Date.now();
  const { endpoint, method, requestId } = context;
  
  try {
    // Execute the handler
    const result = await handler();
    
    // Check for performance issues
    const duration = Date.now() - startTime;
    if (options?.performanceThreshold && duration > options.performanceThreshold) {
      await logApiPerformanceIssue(
        { ...context, startTime },
        duration,
        options.performanceThreshold
      );
    }
    
    return result;
  } catch (error) {
    // Log the API error
    await logApiIssue(
      `Error in API request to ${method} ${endpoint}`,
      { ...context, startTime },
      { duration: Date.now() - startTime },
      error instanceof Error ? error : new Error(String(error))
    );
    
    // Re-throw the error
    throw error;
  }
}