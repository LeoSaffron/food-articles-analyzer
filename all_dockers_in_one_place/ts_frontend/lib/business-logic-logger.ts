/**
 * Utility for tracking and logging business logic issues
 */
import { serverIssueLogger } from './server-issue-logger';

// Interface for business logic context
interface BusinessLogicContext {
  domain: string;
  process: string;
  operation: string;
  userId?: string;
  entityId?: string | number;
  requestId?: string;
}

/**
 * Log a business logic issue
 */
export function logBusinessLogicIssue(
  message: string,
  context: BusinessLogicContext,
  data?: any,
  error?: Error
) {
  // Extract context information
  const { domain, process, operation, userId, entityId, requestId } = context;
  
  // Prepare business context
  const businessContext = {
    domain,
    process,
    operation,
    userId,
    entityId,
    timestamp: new Date().toISOString(),
  };
  
  // Log the business logic issue
  return serverIssueLogger.high(
    message,
    'business_logic',
    { ...data, business: businessContext },
    error,
    { businessDomain: domain, businessProcess: process },
    requestId
  );
}

/**
 * Log a business rule violation
 */
export function logBusinessRuleViolation(
  message: string,
  context: BusinessLogicContext,
  details: {
    rule: string;
    expected: any;
    actual: any;
  }
) {
  return logBusinessLogicIssue(
    `Business rule violation: ${message}`,
    context,
    {
      rule: details.rule,
      expected: details.expected,
      actual: details.actual,
      violation: true,
    }
  );
}

/**
 * Log a business process failure
 */
export function logBusinessProcessFailure(
  message: string,
  context: BusinessLogicContext,
  details: any,
  error?: Error
) {
  return logBusinessLogicIssue(
    `Business process failure: ${message}`,
    context,
    { ...details, processFailure: true },
    error
  );
}

/**
 * Log a business state transition issue
 */
export function logStateTransitionIssue(
  message: string,
  context: BusinessLogicContext,
  details: {
    fromState: string;
    toState: string;
    allowedTransitions?: string[];
  }
) {
  return logBusinessLogicIssue(
    `Invalid state transition: ${message}`,
    context,
    {
      fromState: details.fromState,
      toState: details.toState,
      allowedTransitions: details.allowedTransitions,
      stateTransitionIssue: true,
    }
  );
}

/**
 * Log a business constraint violation
 */
export function logBusinessConstraintViolation(
  message: string,
  context: BusinessLogicContext,
  details: {
    constraint: string;
    value: any;
    limit?: any;
  }
) {
  return logBusinessLogicIssue(
    `Business constraint violation: ${message}`,
    context,
    {
      constraint: details.constraint,
      value: details.value,
      limit: details.limit,
      constraintViolation: true,
    }
  );
}

/**
 * Log a business dependency issue
 */
export function logBusinessDependencyIssue(
  message: string,
  context: BusinessLogicContext,
  details: {
    dependency: string;
    requiredFor: string;
    error?: any;
  },
  error?: Error
) {
  return logBusinessLogicIssue(
    `Business dependency issue: ${message}`,
    context,
    {
      dependency: details.dependency,
      requiredFor: details.requiredFor,
      dependencyError: details.error,
      dependencyIssue: true,
    },
    error
  );
}

/**
 * Business operation wrapper that logs issues
 */
export async function withBusinessLogicLogging<T>(
  context: BusinessLogicContext,
  handler: () => Promise<T>
): Promise<T> {
  const { domain, process, operation, requestId } = context;
  
  try {
    // Execute the business operation
    return await handler();
  } catch (error) {
    // Log the business logic error
    await logBusinessLogicIssue(
      `Error in business operation: ${domain}/${process}/${operation}`,
      context,
      { operationFailed: true },
      error instanceof Error ? error : new Error(String(error))
    );
    
    // Re-throw the error
    throw error;
  }
}