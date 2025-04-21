import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { generateRequestId } from './lib/log-utils';
import { serverLogger } from './lib/server-logger';
import './lib/init-logging'; // Initialize logging systems

export function middleware(request: NextRequest) {
  // Get existing request ID or generate a new one
  const requestId = request.headers.get('X-Request-ID') || generateRequestId();
  
  // Clone the request headers
  const requestHeaders = new Headers(request.headers);
  
  // Add or update the request ID header
  requestHeaders.set('X-Request-ID', requestId);
  
  // Log the request for debugging
  const url = new URL(request.url);
  const path = url.pathname;
  
  // Skip logging for static assets and API routes (they're logged elsewhere)
  if (!path.startsWith('/_next/') && 
      !path.startsWith('/static/') && 
      !path.includes('.') && 
      !path.startsWith('/api/')) {
    
    // Log the request
    serverLogger.debug(
      `Request: ${request.method} ${path}`,
      {
        url: request.url,
        method: request.method,
        path,
        query: Object.fromEntries(url.searchParams.entries()),
        headers: {
          'user-agent': request.headers.get('user-agent'),
          'referer': request.headers.get('referer'),
          'accept-language': request.headers.get('accept-language'),
        },
        ip: request.headers.get('x-forwarded-for') || 
            request.headers.get('x-real-ip') || 
            request.ip || 
            'unknown',
      },
      requestId
    );
  }
  
  // Return the response with the updated headers
  return NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });
}

// Configure middleware to run on all routes
export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};