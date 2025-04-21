import { NextRequest, NextResponse } from 'next/server';
import { withErrorLogging } from '@/app/api/_middleware/error-handler';

/**
 * Wrapper for API route handlers that adds error logging and request tracing
 * 
 * Usage example:
 * 
 * ```ts
 * import { withApiRoute } from '@/lib/api-route-wrapper';
 * 
 * export const GET = withApiRoute(async (request) => {
 *   // Your handler code here
 *   return NextResponse.json({ data: 'example' });
 * });
 * ```
 */
export function withApiRoute(
  handler: (request: NextRequest) => Promise<NextResponse>
) {
  return async (request: NextRequest) => {
    return withErrorLogging(request, () => handler(request));
  };
}