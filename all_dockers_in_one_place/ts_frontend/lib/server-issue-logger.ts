/**
 * Server-side utility for capturing and logging application issues
 */
import { serverLogger } from './server-logger';
import { generateRequestId } from './log-utils';

// Define issue severity levels
export type IssueSeverity = 'low' | 'medium' | 'high' | 'critical';

// Interface for issue data
interface IssueData {
  message: string;
  severity: IssueSeverity;
  source: string;
  data?: any;
  error?: Error;
  context?: Record<string, any>;
  requestId?: string;
}

// Capture system information
function getSystemInfo() {
  return {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    memory: process.memoryUsage(),
    uptime: process.uptime(),
    env: process.env.NODE_ENV,
    timestamp: new Date().toISOString(),
  };
}

// Log application issue
export async function logApplicationIssue(issueData: IssueData): Promise<boolean> {
  try {
    const { message, severity, source, data, error, context, requestId } = issueData;
    
    // Generate request ID if not provided
    const logRequestId = requestId || generateRequestId();
    
    // Prepare full context with system info
    const fullContext = {
      ...context,
      issue: {
        severity,
        source,
        timestamp: new Date().toISOString(),
      },
      system: getSystemInfo(),
    };
    
    // Log to error log for high/critical issues, or debug log for others
    if (severity === 'high' || severity === 'critical') {
      await serverLogger.error(
        `[${severity.toUpperCase()}] ${message}`,
        error,
        { ...fullContext, data },
        logRequestId
      );
    } else {
      await serverLogger.log(
        severity === 'medium' ? 'warn' : 'debug',
        `[${severity.toUpperCase()}] ${message}`,
        { ...fullContext, data, error: error ? { message: error.message, stack: error.stack } : undefined },
        logRequestId
      );
    }
    
    // For all issues, also log detailed debug information with the same request ID
    await serverLogger.debug(
      `Debug context for issue: ${message}`,
      {
        ...fullContext,
        data,
        error: error ? { message: error.message, stack: error.stack, name: error.name } : undefined,
        process: {
          pid: process.pid,
          ppid: process.ppid,
          memoryUsage: process.memoryUsage(),
          cpuUsage: process.cpuUsage ? process.cpuUsage() : undefined,
          resourceUsage: process.resourceUsage ? process.resourceUsage() : undefined,
        },
      },
      logRequestId
    );
    
    return true;
  } catch (err) {
    console.error('Failed to log application issue:', err);
    return false;
  }
}

// Server issue logger
export const serverIssueLogger = {
  // Log low severity issues (minor, non-critical)
  low: (message: string, source: string, data?: any, context?: Record<string, any>, requestId?: string) => {
    return logApplicationIssue({ message, severity: 'low', source, data, context, requestId });
  },
  
  // Log medium severity issues (important but not critical)
  medium: (message: string, source: string, data?: any, context?: Record<string, any>, requestId?: string) => {
    return logApplicationIssue({ message, severity: 'medium', source, data, context, requestId });
  },
  
  // Log high severity issues (significant problems)
  high: (message: string, source: string, data?: any, error?: Error, context?: Record<string, any>, requestId?: string) => {
    return logApplicationIssue({ message, severity: 'high', source, data, error, context, requestId });
  },
  
  // Log critical severity issues (system stability threatened)
  critical: (message: string, source: string, data?: any, error?: Error, context?: Record<string, any>, requestId?: string) => {
    return logApplicationIssue({ message, severity: 'critical', source, data, error, context, requestId });
  },
  
  // Log unexpected behavior
  unexpected: (message: string, source: string, expected: any, actual: any, context?: Record<string, any>, requestId?: string) => {
    return logApplicationIssue({
      message: `Unexpected behavior: ${message}`,
      severity: 'medium',
      source,
      data: { expected, actual, diff: JSON.stringify(expected) !== JSON.stringify(actual) },
      context: {
        ...context,
        issueType: 'unexpected_behavior',
      },
      requestId,
    });
  },
  
  // Log performance issues
  performance: (message: string, source: string, metrics: Record<string, number>, threshold?: Record<string, number>, requestId?: string) => {
    // Determine if any metrics exceed thresholds
    let exceedsThreshold = false;
    const thresholdViolations = {};
    
    if (threshold) {
      for (const [key, value] of Object.entries(metrics)) {
        if (threshold[key] && value > threshold[key]) {
          exceedsThreshold = true;
          thresholdViolations[key] = {
            actual: value,
            threshold: threshold[key],
            difference: value - threshold[key],
            percentOver: ((value - threshold[key]) / threshold[key]) * 100,
          };
        }
      }
    }
    
    return logApplicationIssue({
      message: `Performance issue: ${message}`,
      severity: exceedsThreshold ? 'high' : 'medium',
      source,
      data: {
        metrics,
        threshold,
        thresholdViolations: Object.keys(thresholdViolations).length > 0 ? thresholdViolations : undefined,
      },
      context: {
        issueType: 'performance',
        exceedsThreshold,
      },
      requestId,
    });
  },
};