import { NextRequest, NextResponse } from 'next/server';
import { serverLogger } from '@/lib/server-logger';
import { generateRequestId } from '@/lib/log-utils';

/**
 * Global error handler for API routes
 * This captures all errors that occur in API routes and logs them with full context
 */
export async function withErrorLogging(
  request: NextRequest,
  handler: () => Promise<NextResponse>,
): Promise<NextResponse> {
  // Generate or use existing request ID
  const requestId = request.headers.get('X-Request-ID') || generateRequestId();
  
  // Log the incoming request for debugging
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;
  const sourceIp = request.headers.get('x-forwarded-for') || 
                  request.headers.get('x-real-ip') || 
                  request.ip || 
                  'unknown';
  
  // Always log the request details for debugging
  await serverLogger.debug(
    `API Request: ${method} ${path}`,
    {
      url: request.url,
      method,
      path,
      query: Object.fromEntries(url.searchParams.entries()),
      headers: Object.fromEntries(request.headers.entries()),
      sourceIp,
      timestamp: new Date().toISOString(),
    },
    requestId
  );
  
  try {
    // Execute the handler
    const response = await handler();
    
    // Log the response for debugging
    await serverLogger.debug(
      `API Response: ${method} ${path} - ${response.status}`,
      {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries()),
        url: request.url,
        method,
        path,
        timestamp: new Date().toISOString(),
      },
      requestId
    );
    
    return response;
  } catch (error) {
    // Log the error with full context
    console.error(`API Error: ${method} ${path}`, error);
    
    // Capture full request context
    const fullContext = {
      request: {
        url: request.url,
        method,
        path,
        query: Object.fromEntries(url.searchParams.entries()),
        headers: Object.fromEntries(request.headers.entries()),
        sourceIp,
      },
      environment: {
        nodeEnv: process.env.NODE_ENV,
        timestamp: new Date().toISOString(),
      },
      runtime: {
        memory: process.memoryUsage ? process.memoryUsage() : null,
      }
    };
    
    // Log to server logger with full debug information
    await serverLogger.error(
      `API Error: ${method} ${path}`,
      error instanceof Error ? error : new Error(String(error)),
      fullContext,
      requestId
    );
    
    // Also log a debug entry with the same requestId for correlation
    await serverLogger.debug(
      `Debug context for API error: ${method} ${path}`,
      {
        ...fullContext,
        error: error instanceof Error
          ? {
              message: error.message,
              stack: error.stack,
              name: error.name,
              constructor: error.constructor?.name,
            }
          : String(error),
      },
      requestId
    );
    
    // Return an error response
    return NextResponse.json(
      {
        error: 'Internal Server Error',
        message: 'An unexpected error occurred',
        requestId, // Include the request ID for troubleshooting
      },
      { status: 500 }
    );
  }
}