/**
 * Utility for tracking and logging data inconsistency issues
 */
import { serverIssueLogger } from './server-issue-logger';

// Interface for data validation context
interface DataValidationContext {
  source: string;
  operation: string;
  entityType: string;
  entityId?: string | number;
  requestId?: string;
}

/**
 * Log a data validation issue
 */
export function logDataValidationIssue(
  message: string,
  context: DataValidationContext,
  validationErrors: any,
  data?: any
) {
  return serverIssueLogger.medium(
    message,
    'data_validation',
    {
      ...context,
      validationErrors,
      data: sanitizeData(data),
    },
    undefined,
    { dataValidation: true },
    context.requestId
  );
}

/**
 * Log a data inconsistency issue
 */
export function logDataInconsistencyIssue(
  message: string,
  context: DataValidationContext,
  expected: any,
  actual: any
) {
  return serverIssueLogger.high(
    message,
    'data_inconsistency',
    {
      ...context,
      expected: sanitizeData(expected),
      actual: sanitizeData(actual),
      diff: generateDiff(expected, actual),
    },
    undefined,
    { dataInconsistency: true },
    context.requestId
  );
}

/**
 * Log a data integrity issue
 */
export function logDataIntegrityIssue(
  message: string,
  context: DataValidationContext,
  details: any,
  error?: Error
) {
  return serverIssueLogger.critical(
    message,
    'data_integrity',
    {
      ...context,
      details: sanitizeData(details),
    },
    error,
    { dataIntegrity: true },
    context.requestId
  );
}

/**
 * Sanitize sensitive data
 */
function sanitizeData(data: any): any {
  if (!data) return data;
  
  // Create a deep copy
  const sanitized = JSON.parse(JSON.stringify(data));
  
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

/**
 * Generate a simple diff between expected and actual data
 */
function generateDiff(expected: any, actual: any): any {
  if (!expected || !actual) return { error: 'Cannot compare null or undefined values' };
  if (typeof expected !== 'object' || typeof actual !== 'object') {
    return { expected, actual, different: expected !== actual };
  }
  
  const diff: Record<string, any> = {};
  
  // Check for keys in expected that are different or missing in actual
  for (const key of Object.keys(expected)) {
    if (!(key in actual)) {
      diff[key] = { expected: expected[key], actual: 'MISSING' };
    } else if (typeof expected[key] === 'object' && expected[key] !== null && 
               typeof actual[key] === 'object' && actual[key] !== null) {
      const nestedDiff = generateDiff(expected[key], actual[key]);
      if (Object.keys(nestedDiff).length > 0) {
        diff[key] = nestedDiff;
      }
    } else if (expected[key] !== actual[key]) {
      diff[key] = { expected: expected[key], actual: actual[key] };
    }
  }
  
  // Check for keys in actual that are not in expected
  for (const key of Object.keys(actual)) {
    if (!(key in expected)) {
      diff[key] = { expected: 'MISSING', actual: actual[key] };
    }
  }
  
  return diff;
}

/**
 * Validate data against a schema and log any issues
 */
export function validateAndLogDataIssues(
  data: any,
  schema: any,
  context: DataValidationContext
): { valid: boolean; errors?: any } {
  // This is a placeholder for actual schema validation
  // In a real implementation, you would use a library like Joi, Yup, or Zod
  
  // Simulate validation
  const validationResult = validateAgainstSchema(data, schema);
  
  // If validation failed, log the issue
  if (!validationResult.valid) {
    logDataValidationIssue(
      `Data validation failed for ${context.entityType}`,
      context,
      validationResult.errors,
      data
    );
  }
  
  return validationResult;
}

/**
 * Placeholder for schema validation
 */
function validateAgainstSchema(data: any, schema: any): { valid: boolean; errors?: any } {
  // This is a placeholder - in a real implementation, use a validation library
  const errors: Record<string, string> = {};
  let valid = true;
  
  // Simple validation example
  for (const [field, rules] of Object.entries(schema)) {
    if (rules.required && (data[field] === undefined || data[field] === null)) {
      errors[field] = `${field} is required`;
      valid = false;
    } else if (rules.type && data[field] !== undefined) {
      if (rules.type === 'string' && typeof data[field] !== 'string') {
        errors[field] = `${field} must be a string`;
        valid = false;
      } else if (rules.type === 'number' && typeof data[field] !== 'number') {
        errors[field] = `${field} must be a number`;
        valid = false;
      } else if (rules.type === 'boolean' && typeof data[field] !== 'boolean') {
        errors[field] = `${field} must be a boolean`;
        valid = false;
      }
    }
  }
  
  return { valid, errors: valid ? undefined : errors };
}

/**
 * Compare expected data with actual data and log any inconsistencies
 */
export function compareAndLogDataInconsistencies(
  expected: any,
  actual: any,
  context: DataValidationContext
): boolean {
  // Generate diff between expected and actual
  const diff = generateDiff(expected, actual);
  
  // If there are differences, log the inconsistency
  if (Object.keys(diff).length > 0) {
    logDataInconsistencyIssue(
      `Data inconsistency detected for ${context.entityType}`,
      context,
      expected,
      actual
    );
    return false;
  }
  
  return true;
}