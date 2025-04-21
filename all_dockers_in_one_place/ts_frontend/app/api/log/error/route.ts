import { NextRequest, NextResponse } from 'next/server'
import { serverLogger } from '@/lib/server-logger'
import { generateRequestId } from '@/lib/log-utils'
import { withApiRoute } from '@/lib/api-route-wrapper'

export const runtime = 'nodejs' // Force Node.js runtime for this route

export const POST = withApiRoute(async (request: NextRequest) => {
  const body = await request.json()
  const { message, error, url, context = {} } = body
  
  // Use provided request ID or generate a new one
  const requestId = request.headers.get('X-Request-ID') || generateRequestId()
  
  // Get client IP from various headers and fallbacks
  const sourceIp = request.headers.get('x-forwarded-for') || 
                 request.headers.get('x-real-ip') || 
                 request.ip || 
                 'unknown'
  
  const userAgent = request.headers.get('user-agent') || 'unknown'
  
  // Capture full request context
  const fullContext = {
    ...context,
    error: {
      ...error,
      message: error?.message || 'Unknown error',
      stack: error?.stack || null,
      digest: error?.digest || null,
    },
    request: {
      url,
      sourceIp,
      userAgent,
      method: request.method,
      referer: request.headers.get('referer') || 'direct',
      host: request.headers.get('host') || 'unknown',
      headers: Object.fromEntries(request.headers.entries()),
    },
    environment: {
      nodeEnv: process.env.NODE_ENV,
      timestamp: new Date().toISOString(),
    },
    // Capture any additional runtime information
    runtime: {
      memory: process.memoryUsage ? process.memoryUsage() : null,
    }
  }

  // Log the error with full context
  console.error(`Error logged: ${message}`, { error: error?.message, url, requestId })
  
  // Log to server logger with full debug information
  await serverLogger.error(message, error, fullContext, requestId)
  
  // Also log a debug entry with the same requestId for correlation
  await serverLogger.debug(`Debug context for error: ${message}`, fullContext, requestId)

  return NextResponse.json({ success: true, requestId })
})