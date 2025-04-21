/**
 * Utility for capturing and logging database-related issues
 */
import { serverIssueLogger } from './server-issue-logger';

// Interface for database operation context
interface DbOperationContext {
  operation: string;
  entity: string;
  params?: any;
  requestId?: string;
  userId?: string;
  startTime?: number;
}

// Log database operation issue
export function logDbIssue(message: string, context: DbOperationContext, data?: any, error?: Error) {
  // Extract operation information
  const { operation, entity, params, requestId, userId, startTime } = context;
  
  // Calculate operation duration if startTime is provided
  const duration = startTime ? Date.now() - startTime : undefined;
  
  // Prepare operation context
  const operationContext = {
    operation,
    entity,
    params: sanitizeParams(params),
    userId,
    duration,
  };
  
  // Determine severity based on error type
  const severity = determineSeverity(error);
  
  // Log the issue with appropriate severity
  if (severity === 'critical' || severity === 'high') {
    return serverIssueLogger[severity](
      message,
      'database',
      { ...data, db: operationContext },
      error,
      { dbEntity: entity, dbOperation: operation },
      requestId
    );
  } else {
    return serverIssueLogger.medium(
      message,
      'database',
      { ...data, db: operationContext },
      error,
      { dbEntity: entity, dbOperation: operation },
      requestId
    );
  }
}

// Sanitize database parameters to remove sensitive information
function sanitizeParams(params?: any) {
  if (!params) return undefined;
  
  // Create a deep copy to avoid modifying the original
  const sanitized = JSON.parse(JSON.stringify(params));
  
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

// Determine issue severity based on error type
function determineSeverity(error?: Error): 'low' | 'medium' | 'high' | 'critical' {
  if (!error) return 'medium';
  
  const errorMessage = error.message.toLowerCase();
  const errorName = error.name.toLowerCase();
  
  // Critical database issues
  if (
    errorMessage.includes('connection') ||
    errorMessage.includes('timeout') ||
    errorMessage.includes('unavailable') ||
    errorMessage.includes('deadlock') ||
    errorName.includes('connection') ||
    errorName.includes('timeout')
  ) {
    return 'critical';
  }
  
  // High severity issues
  if (
    errorMessage.includes('constraint') ||
    errorMessage.includes('duplicate') ||
    errorMessage.includes('foreign key') ||
    errorMessage.includes('integrity') ||
    errorName.includes('constraint') ||
    errorName.includes('integrity')
  ) {
    return 'high';
  }
  
  // Default to medium
  return 'medium';
}

// Log database performance issue
export function logDbPerformanceIssue(context: DbOperationContext, duration: number, threshold: number) {
  const { operation, entity, requestId } = context;
  
  return serverIssueLogger.performance(
    `Slow database operation: ${operation} on ${entity}`,
    'db_performance',
    { duration, operation, entity },
    { duration: threshold },
    requestId
  );
}

// Database operation wrapper that logs issues
export async function withDbIssueLogging<T>(
  context: DbOperationContext,
  handler: () => Promise<T>,
  options?: { performanceThreshold?: number }
): Promise<T> {
  const startTime = Date.now();
  const { operation, entity, requestId } = context;
  
  try {
    // Execute the handler
    const result = await handler();
    
    // Check for performance issues
    const duration = Date.now() - startTime;
    if (options?.performanceThreshold && duration > options.performanceThreshold) {
      await logDbPerformanceIssue(
        { ...context, startTime },
        duration,
        options.performanceThreshold
      );
    }
    
    return result;
  } catch (error) {
    // Log the database error
    await logDbIssue(
      `Error in database operation: ${operation} on ${entity}`,
      { ...context, startTime },
      { duration: Date.now() - startTime },
      error instanceof Error ? error : new Error(String(error))
    );
    
    // Re-throw the error
    throw error;
  }
}