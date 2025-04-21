/**
 * Client-side logger for capturing and sending application issues to the server
 */

// Define log levels
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

// Interface for log data
interface LogData {
  message: string;
  level: LogLevel;
  data?: any;
  error?: Error;
  context?: Record<string, any>;
}

// Get browser information
function getBrowserInfo() {
  if (typeof window === 'undefined') return {};
  
  return {
    userAgent: navigator.userAgent,
    language: navigator.language,
    platform: navigator.platform,
    url: window.location.href,
    referrer: document.referrer,
    screenSize: {
      width: window.innerWidth,
      height: window.innerHeight,
      screenWidth: window.screen.width,
      screenHeight: window.screen.height,
    },
  };
}

// Get performance information
function getPerformanceInfo() {
  if (typeof window === 'undefined' || !window.performance) return {};
  
  return {
    timing: JSON.parse(JSON.stringify(performance.timing || {})),
    navigation: JSON.parse(JSON.stringify(performance.navigation || {})),
    memory: performance.memory ? {
      jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
      totalJSHeapSize: performance.memory.totalJSHeapSize,
      usedJSHeapSize: performance.memory.usedJSHeapSize,
    } : null,
  };
}

// Send log to server
async function sendLog(logData: LogData): Promise<boolean> {
  try {
    const { message, level, data, error, context } = logData;
    
    // Prepare error object if provided
    const errorObj = error ? {
      message: error.message,
      stack: error.stack,
      name: error.name,
      constructor: error.constructor?.name,
    } : undefined;
    
    // Prepare context with browser and performance info
    const fullContext = {
      ...context,
      browser: getBrowserInfo(),
      performance: getPerformanceInfo(),
      timestamp: new Date().toISOString(),
    };
    
    // Determine endpoint based on level
    const endpoint = level === 'error' 
      ? '/api/log/error' 
      : '/api/log';
    
    // Send to server
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Log-Source': 'client',
      },
      body: JSON.stringify({
        level,
        message,
        data,
        error: errorObj,
        context: fullContext,
        url: window.location.href,
      }),
    });
    
    return response.ok;
  } catch (err) {
    console.error('Failed to send log to server:', err);
    return false;
  }
}

// Client logger
export const appLogger = {
  debug: (message: string, data?: any, context?: Record<string, any>) => {
    console.debug(message, data);
    return sendLog({ message, level: 'debug', data, context });
  },
  
  info: (message: string, data?: any, context?: Record<string, any>) => {
    console.info(message, data);
    return sendLog({ message, level: 'info', data, context });
  },
  
  warn: (message: string, data?: any, context?: Record<string, any>) => {
    console.warn(message, data);
    return sendLog({ message, level: 'warn', data, context });
  },
  
  error: (message: string, error?: Error, data?: any, context?: Record<string, any>) => {
    console.error(message, error, data);
    return sendLog({ message, level: 'error', data, error, context });
  },
  
  // Log application issues that aren't errors but need attention
  issue: (message: string, data?: any, context?: Record<string, any>) => {
    console.warn(`APPLICATION ISSUE: ${message}`, data);
    return sendLog({
      message: `APPLICATION ISSUE: ${message}`,
      level: 'warn',
      data,
      context: {
        ...context,
        issueType: 'application_issue',
        needsAttention: true,
      },
    });
  },
  
  // Log unexpected behavior
  unexpected: (message: string, expected: any, actual: any, context?: Record<string, any>) => {
    console.warn(`UNEXPECTED BEHAVIOR: ${message}`, { expected, actual });
    return sendLog({
      message: `UNEXPECTED BEHAVIOR: ${message}`,
      level: 'warn',
      data: { expected, actual },
      context: {
        ...context,
        issueType: 'unexpected_behavior',
        needsAttention: true,
      },
    });
  },
};

// Global error handler
export function setupGlobalErrorHandling() {
  if (typeof window === 'undefined') return;
  
  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    appLogger.error(
      'Unhandled Promise Rejection',
      event.reason instanceof Error ? event.reason : new Error(String(event.reason)),
      { type: 'unhandledrejection' },
      { source: 'global_handler' }
    );
  });
  
  // Handle uncaught errors
  window.addEventListener('error', (event) => {
    // Avoid duplicate logging for errors that are already handled by unhandledrejection
    if (event.error && event.error._logged) return;
    
    appLogger.error(
      'Uncaught Error',
      event.error || new Error(event.message),
      {
        type: 'uncaught',
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      },
      { source: 'global_handler' }
    );
    
    // Mark as logged to prevent duplicate logging
    if (event.error) event.error._logged = true;
  });
}