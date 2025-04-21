/**
 * Utility for tracking and logging configuration and environment issues
 */
import { serverIssueLogger } from './server-issue-logger';

// Interface for configuration context
interface ConfigContext {
  component: string;
  configType: 'env' | 'file' | 'database' | 'remote' | 'other';
  source?: string;
}

/**
 * Log a configuration issue
 */
export function logConfigIssue(
  message: string,
  context: ConfigContext,
  data?: any,
  error?: Error
) {
  // Extract context information
  const { component, configType, source } = context;
  
  // Prepare config context
  const configContext = {
    component,
    configType,
    source,
    timestamp: new Date().toISOString(),
  };
  
  // Log the configuration issue
  return serverIssueLogger.critical(
    message,
    'configuration',
    { ...data, config: configContext },
    error,
    { configComponent: component, configType }
  );
}

/**
 * Log a missing configuration issue
 */
export function logMissingConfig(
  component: string,
  configKey: string,
  configType: 'env' | 'file' | 'database' | 'remote' | 'other' = 'env',
  source?: string
) {
  return logConfigIssue(
    `Missing configuration: ${configKey} for ${component}`,
    { component, configType, source },
    { configKey, missing: true }
  );
}

/**
 * Log an invalid configuration issue
 */
export function logInvalidConfig(
  component: string,
  configKey: string,
  value: any,
  expectedFormat: string,
  configType: 'env' | 'file' | 'database' | 'remote' | 'other' = 'env',
  source?: string
) {
  return logConfigIssue(
    `Invalid configuration: ${configKey} for ${component}`,
    { component, configType, source },
    {
      configKey,
      value: sanitizeConfigValue(configKey, value),
      expectedFormat,
      invalid: true,
    }
  );
}

/**
 * Log a configuration conflict issue
 */
export function logConfigConflict(
  component: string,
  conflictingKeys: string[],
  details: any,
  configType: 'env' | 'file' | 'database' | 'remote' | 'other' = 'env',
  source?: string
) {
  return logConfigIssue(
    `Configuration conflict for ${component}: ${conflictingKeys.join(', ')}`,
    { component, configType, source },
    {
      conflictingKeys,
      details: sanitizeConfigValues(details),
      conflict: true,
    }
  );
}

/**
 * Log an environment issue
 */
export function logEnvironmentIssue(
  message: string,
  component: string,
  details: any,
  error?: Error
) {
  return serverIssueLogger.critical(
    `Environment issue: ${message}`,
    'environment',
    {
      component,
      details,
      environment: process.env.NODE_ENV || 'unknown',
      environmentIssue: true,
    },
    error,
    { environmentComponent: component }
  );
}

/**
 * Sanitize potentially sensitive configuration values
 */
function sanitizeConfigValue(key: string, value: any): any {
  // List of sensitive config keys
  const sensitiveKeys = [
    'password', 'secret', 'key', 'token', 'auth', 'credential',
    'apikey', 'api_key', 'access', 'private'
  ];
  
  // Check if the key contains any sensitive terms
  if (sensitiveKeys.some(sensitive => key.toLowerCase().includes(sensitive))) {
    return '[REDACTED]';
  }
  
  return value;
}

/**
 * Sanitize all potentially sensitive values in an object
 */
function sanitizeConfigValues(obj: Record<string, any>): Record<string, any> {
  if (!obj || typeof obj !== 'object') return obj;
  
  const sanitized: Record<string, any> = {};
  
  for (const [key, value] of Object.entries(obj)) {
    sanitized[key] = sanitizeConfigValue(key, value);
  }
  
  return sanitized;
}

/**
 * Validate required environment variables and log any issues
 */
export function validateRequiredEnvVars(component: string, requiredVars: string[]): boolean {
  let allPresent = true;
  
  for (const varName of requiredVars) {
    if (!process.env[varName]) {
      logMissingConfig(component, varName, 'env');
      allPresent = false;
    }
  }
  
  return allPresent;
}

/**
 * Validate environment variable format and log any issues
 */
export function validateEnvVarFormat(
  component: string,
  varName: string,
  validator: (value: string) => boolean,
  expectedFormat: string
): boolean {
  const value = process.env[varName];
  
  if (!value) {
    logMissingConfig(component, varName, 'env');
    return false;
  }
  
  if (!validator(value)) {
    logInvalidConfig(component, varName, value, expectedFormat, 'env');
    return false;
  }
  
  return true;
}