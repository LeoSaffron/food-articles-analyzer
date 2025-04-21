/**
 * Utility for tracking and logging third-party integration issues
 */
import { serverIssueLogger } from './server-issue-logger';

// Interface for integration operation context
interface IntegrationContext {
  service: string;
  operation: string;
  endpoint?: string;
  requestId?: string;
  startTime?: number;
}

/**
 * Log an integration issue
 */
export function logIntegrationIssue(
  message: string,
  context: IntegrationContext,
  data?: any,
  error?: Error
) {
  // Extract context information
  const { service, operation, endpoint, requestId, startTime } = context;
  
  // Calculate operation duration if startTime is provided
  const duration = startTime ? Date.now() - startTime : undefined;
  
  // Prepare operation context
  const operationContext = {
    service,
    operation,
    endpoint,
    duration,
  };
  
  // Determine severity based on error type and service
  const severity = determineIntegrationSeverity(service, error);
  
  // Log the issue with appropriate severity
  return serverIssueLogger[severity](
    message,
    'integration',
    { ...data, integration: operationContext },
    error,
    { integrationService: service, integrationOperation: operation },
    requestId
  );
}

/**
 * Determine issue severity based on service and error
 */
function determineIntegrationSeverity(service: string, error?: Error): 'low' | 'medium' | 'high' | 'critical' {
  if (!error) return 'medium';
  
  const errorMessage = error.message.toLowerCase();
  
  // Critical issues for authentication and core services
  if (
    service.toLowerCase().includes('auth') ||
    service.toLowerCase().includes('payment') ||
    service.toLowerCase().includes('core')
  ) {
    return 'critical';
  }
  
  // Critical issues for connection problems
  if (
    errorMessage.includes('connection') ||
    errorMessage.includes('timeout') ||
    errorMessage.includes('network') ||
    errorMessage.includes('unavailable')
  ) {
    return 'high';
  }
  
  // Default to medium
  return 'medium';
}

/**
 * Log an integration timeout issue
 */
export function logIntegrationTimeout(
  context: IntegrationContext,
  timeoutMs: number
) {
  const { service, operation, endpoint, requestId } = context;
  
  return serverIssueLogger.high(
    `Integration timeout: ${service} - ${operation}`,
    'integration_timeout',
    {
      service,
      operation,
      endpoint,
      timeoutMs,
    },
    new Error(`Integration timeout after ${timeoutMs}ms`),
    { integrationService: service, integrationOperation: operation },
    requestId
  );
}

/**
 * Log an integration rate limit issue
 */
export function logIntegrationRateLimit(
  context: IntegrationContext,
  rateLimitInfo: any
) {
  const { service, operation, endpoint, requestId } = context;
  
  return serverIssueLogger.high(
    `Integration rate limited: ${service} - ${operation}`,
    'integration_rate_limit',
    {
      service,
      operation,
      endpoint,
      rateLimitInfo,
    },
    new Error(`Integration rate limited`),
    { integrationService: service, integrationOperation: operation },
    requestId
  );
}

/**
 * Integration operation wrapper that logs issues
 */
export async function withIntegrationLogging<T>(
  context: IntegrationContext,
  handler: () => Promise<T>,
  options?: { timeoutMs?: number }
): Promise<T> {
  const startTime = Date.now();
  const { service, operation, requestId } = context;
  
  // Set up timeout if specified
  let timeoutId: NodeJS.Timeout | undefined;
  let timeoutPromise: Promise<never> | undefined;
  
  if (options?.timeoutMs) {
    timeoutPromise = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => {
        logIntegrationTimeout(
          { ...context, startTime },
          options.timeoutMs!
        );
        reject(new Error(`Integration timeout after ${options.timeoutMs}ms`));
      }, options.timeoutMs);
    });
  }
  
  try {
    // Execute the handler with optional timeout
    const result = await (timeoutPromise
      ? Promise.race([handler(), timeoutPromise])
      : handler());
    
    // Clear timeout if set
    if (timeoutId) clearTimeout(timeoutId);
    
    return result;
  } catch (error) {
    // Clear timeout if set
    if (timeoutId) clearTimeout(timeoutId);
    
    // Check for rate limit errors
    if (
      error.message?.toLowerCase().includes('rate limit') ||
      error.message?.toLowerCase().includes('too many requests') ||
      error.status === 429 ||
      error.statusCode === 429
    ) {
      await logIntegrationRateLimit(
        { ...context, startTime },
        {
          error: error.message,
          status: error.status || error.statusCode,
          headers: error.headers,
        }
      );
    } else {
      // Log the integration error
      await logIntegrationIssue(
        `Error in integration: ${service} - ${operation}`,
        { ...context, startTime },
        { duration: Date.now() - startTime },
        error instanceof Error ? error : new Error(String(error))
      );
    }
    
    // Re-throw the error
    throw error;
  }
}

/**
 * Create a wrapped API client that logs integration issues
 */
export function createLoggingIntegrationClient<T extends Record<string, Function>>(
  client: T,
  serviceName: string,
  options?: { timeoutMs?: number }
): T {
  const wrappedClient = {} as T;
  
  // Wrap each method in the client
  for (const [methodName, method] of Object.entries(client)) {
    if (typeof method === 'function') {
      wrappedClient[methodName] = async (...args: any[]) => {
        return withIntegrationLogging(
          {
            service: serviceName,
            operation: methodName,
            startTime: Date.now(),
          },
          () => method.apply(client, args),
          options
        );
      };
    } else {
      wrappedClient[methodName] = method;
    }
  }
  
  return wrappedClient;
}