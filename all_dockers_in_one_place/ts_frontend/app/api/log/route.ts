import { NextRequest, NextResponse } from 'next/server'
import { serverLogger } from '@/lib/server-logger'
import { generateRequestId } from '@/lib/log-utils'
import { withApiRoute } from '@/lib/api-route-wrapper'

export const runtime = 'nodejs' // Force Node.js runtime for this route

export const POST = withApiRoute(async (request: NextRequest) => {
  const body = await request.json()
  const { level, message, data, requestId } = body
  
  // Use provided request ID or generate a new one
  const logRequestId = requestId || request.headers.get('X-Request-ID') || generateRequestId()
  
  // Get client IP from various headers and fallbacks
  const sourceIp = request.headers.get('x-forwarded-for') || 
                  request.headers.get('x-real-ip') || 
                  request.ip || 
                  'unknown'
  
  const userAgent = request.headers.get('user-agent') || 'unknown'
  
  // Add request context to logs
  const contextData = {
    ...data,
    _context: {
      sourceIp,  // Include source IP in the log
      clientIp: sourceIp, // For backward compatibility
      userAgent,
      url: request.url,
      method: request.method,
      referer: request.headers.get('referer') || 'direct',
      host: request.headers.get('host') || 'unknown'
    }
  }

  // Log the request to console for debugging
  console.log(`API Log Request: ${level} - ${message} - IP: ${sourceIp}`);

  // Always log a debug entry for every request regardless of level
  await serverLogger.debug(
    `Log API Request: ${level} - ${message}`,
    {
      level,
      message,
      data: contextData,
      request: {
        url: request.url,
        method: request.method,
        headers: Object.fromEntries(request.headers.entries()),
        sourceIp,
      }
    },
    logRequestId
  )

  switch (level) {
    case 'access':
    case 'info':
    case 'warn':
      await serverLogger.log(level, message, contextData, logRequestId)
      break
    case 'debug':
      // Always log debug messages regardless of ENABLE_DEBUG setting
      await serverLogger.debug(message, contextData, logRequestId)
      break
    case 'error':
      await serverLogger.error(message, undefined, contextData, logRequestId)
      break
    default:
      return NextResponse.json({ error: 'Invalid log level' }, { status: 400 })
  }

  return NextResponse.json({ success: true })
})