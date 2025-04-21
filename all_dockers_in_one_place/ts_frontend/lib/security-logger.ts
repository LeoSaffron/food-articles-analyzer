/**
 * Utility for tracking and logging security-related issues
 */
import { serverIssueLogger } from './server-issue-logger';
import { NextRequest } from 'next/server';

// Interface for security event context
interface SecurityContext {
  type: 'authentication' | 'authorization' | 'input_validation' | 'rate_limit' | 'suspicious_activity' | 'other';
  source: string;
  userId?: string;
  ip?: string;
  requestId?: string;
  request?: NextRequest;
}

/**
 * Log a security issue
 */
export function logSecurityIssue(
  message: string,
  context: SecurityContext,
  data?: any,
  error?: Error
) {
  // Extract context information
  const { type, source, userId, ip, requestId, request } = context;
  
  // Get client IP from request or provided IP
  const clientIp = ip || (request ? (
    request.headers.get('x-forwarded-for') ||
    request.headers.get('x-real-ip') ||
    request.ip ||
    'unknown'
  ) : 'unknown');
  
  // Prepare security context
  const securityContext = {
    type,
    source,
    userId,
    ip: clientIp,
    userAgent: request?.headers.get('user-agent') || 'unknown',
    timestamp: new Date().toISOString(),
  };
  
  // Add request details if available
  if (request) {
    securityContext['request'] = {
      url: request.url,
      method: request.method,
      referer: request.headers.get('referer') || 'direct',
      host: request.headers.get('host') || 'unknown',
    };
  }
  
  // Always log security issues as high or critical severity
  const severity = type === 'authentication' || type === 'authorization' ? 'critical' : 'high';
  
  // Log the security issue
  return serverIssueLogger[severity](
    message,
    'security',
    { ...data, security: securityContext },
    error,
    { securityType: type, securitySource: source },
    requestId
  );
}

/**
 * Log an authentication issue
 */
export function logAuthenticationIssue(
  message: string,
  source: string,
  details: any,
  request?: NextRequest,
  error?: Error
) {
  return logSecurityIssue(
    `Authentication issue: ${message}`,
    {
      type: 'authentication',
      source,
      ip: details.ip,
      userId: details.userId,
      requestId: details.requestId,
      request,
    },
    { details },
    error
  );
}

/**
 * Log an authorization issue
 */
export function logAuthorizationIssue(
  message: string,
  source: string,
  details: {
    userId: string;
    resource: string;
    action: string;
    requiredPermissions?: string[];
    userPermissions?: string[];
  },
  request?: NextRequest,
  error?: Error
) {
  return logSecurityIssue(
    `Authorization issue: ${message}`,
    {
      type: 'authorization',
      source,
      userId: details.userId,
      request,
    },
    { details },
    error
  );
}

/**
 * Log an input validation security issue
 */
export function logInputValidationIssue(
  message: string,
  source: string,
  details: {
    input: any;
    validationErrors: any;
    userId?: string;
  },
  request?: NextRequest
) {
  return logSecurityIssue(
    `Input validation issue: ${message}`,
    {
      type: 'input_validation',
      source,
      userId: details.userId,
      request,
    },
    {
      validationErrors: details.validationErrors,
      sanitizedInput: sanitizeInput(details.input),
    }
  );
}

/**
 * Log a rate limit issue
 */
export function logRateLimitIssue(
  message: string,
  source: string,
  details: {
    limit: number;
    current: number;
    windowMs: number;
    userId?: string;
  },
  request?: NextRequest
) {
  return logSecurityIssue(
    `Rate limit issue: ${message}`,
    {
      type: 'rate_limit',
      source,
      userId: details.userId,
      request,
    },
    { details }
  );
}

/**
 * Log suspicious activity
 */
export function logSuspiciousActivity(
  message: string,
  source: string,
  details: any,
  request?: NextRequest
) {
  return logSecurityIssue(
    `Suspicious activity: ${message}`,
    {
      type: 'suspicious_activity',
      source,
      userId: details.userId,
      request,
    },
    { details }
  );
}

/**
 * Sanitize input to remove sensitive information
 */
function sanitizeInput(input: any): any {
  if (!input) return input;
  
  // Create a deep copy
  const sanitized = JSON.parse(JSON.stringify(input));
  
  // List of sensitive fields to redact
  const sensitiveFields = [
    'password', 'token', 'secret', 'key', 'auth', 'credential', 'credit', 'card',
    'cvv', 'ssn', 'social', 'license', 'passport'
  ];
  
  // Recursively sanitize the object
  function sanitizeObject(obj: any) {
    if (!obj || typeof obj !== 'object') return;
    
    for (const key of Object.keys(obj)) {
      // Check if the key contains any sensitive terms
      if (sensitiveFields.some(field => key.toLowerCase().includes(field))) {
        obj[key] = '[REDACTED]';
      } else if (typeof obj[key] === 'object') {
        sanitizeObject(obj[key]);
      }
    }
  }
  
  sanitizeObject(sanitized);
  return sanitized;
}