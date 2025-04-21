/**
 * Utility for monitoring and logging application state issues
 */
import { serverIssueLogger } from './server-issue-logger';

// Interface for state check result
interface StateCheckResult {
  healthy: boolean;
  issues?: string[];
  metrics?: Record<string, number>;
  details?: any;
}

// Type for state check function
type StateCheckFn = () => Promise<StateCheckResult> | StateCheckResult;

// Registry of state checks
const stateChecks: Record<string, { check: StateCheckFn; critical: boolean }> = {};

/**
 * Register a new application state check
 */
export function registerStateCheck(name: string, check: StateCheckFn, options?: { critical?: boolean }) {
  stateChecks[name] = {
    check,
    critical: options?.critical ?? false,
  };
}

/**
 * Run all registered state checks and log any issues
 */
export async function runStateChecks(requestId?: string): Promise<{
  healthy: boolean;
  criticalIssues: number;
  nonCriticalIssues: number;
  results: Record<string, StateCheckResult>;
}> {
  const results: Record<string, StateCheckResult> = {};
  let criticalIssues = 0;
  let nonCriticalIssues = 0;
  
  // Run all checks in parallel
  await Promise.all(
    Object.entries(stateChecks).map(async ([name, { check, critical }]) => {
      try {
        // Run the check
        const result = await Promise.resolve(check());
        results[name] = result;
        
        // If not healthy, log the issue
        if (!result.healthy) {
          if (critical) {
            criticalIssues++;
            await serverIssueLogger.critical(
              `Critical application state issue in ${name}`,
              'app_state',
              {
                check: name,
                issues: result.issues,
                metrics: result.metrics,
                details: result.details,
              },
              undefined,
              { stateCheck: name },
              requestId
            );
          } else {
            nonCriticalIssues++;
            await serverIssueLogger.high(
              `Application state issue in ${name}`,
              'app_state',
              {
                check: name,
                issues: result.issues,
                metrics: result.metrics,
                details: result.details,
              },
              undefined,
              { stateCheck: name },
              requestId
            );
          }
        }
      } catch (error) {
        // Log check execution error
        results[name] = {
          healthy: false,
          issues: [`Check execution failed: ${error.message}`],
        };
        
        if (critical) {
          criticalIssues++;
          await serverIssueLogger.critical(
            `Failed to execute critical state check: ${name}`,
            'app_state',
            { check: name },
            error instanceof Error ? error : new Error(String(error)),
            { stateCheck: name },
            requestId
          );
        } else {
          nonCriticalIssues++;
          await serverIssueLogger.high(
            `Failed to execute state check: ${name}`,
            'app_state',
            { check: name },
            error instanceof Error ? error : new Error(String(error)),
            { stateCheck: name },
            requestId
          );
        }
      }
    })
  );
  
  // Overall health status
  const healthy = criticalIssues === 0;
  
  // Log overall state if there are issues
  if (criticalIssues > 0 || nonCriticalIssues > 0) {
    await serverIssueLogger[criticalIssues > 0 ? 'critical' : 'high'](
      `Application state check: ${criticalIssues} critical and ${nonCriticalIssues} non-critical issues found`,
      'app_state_summary',
      {
        criticalIssues,
        nonCriticalIssues,
        totalChecks: Object.keys(stateChecks).length,
        checkResults: Object.entries(results).reduce((acc, [name, result]) => {
          acc[name] = { healthy: result.healthy, issues: result.issues };
          return acc;
        }, {}),
      },
      undefined,
      { stateCheckSummary: true },
      requestId
    );
  }
  
  return {
    healthy,
    criticalIssues,
    nonCriticalIssues,
    results,
  };
}

/**
 * Schedule periodic state checks
 */
export function scheduleStateChecks(intervalMinutes = 15) {
  // Only run in Node.js environment
  if (typeof window !== 'undefined') return;
  
  // Convert minutes to milliseconds
  const interval = intervalMinutes * 60 * 1000;
  
  // Run initial check
  setTimeout(() => {
    const requestId = `scheduled-state-check-${Date.now()}`;
    runStateChecks(requestId).catch(error => {
      console.error('Failed to run scheduled state checks:', error);
    });
    
    // Set up interval for future checks
    setInterval(() => {
      const requestId = `scheduled-state-check-${Date.now()}`;
      runStateChecks(requestId).catch(error => {
        console.error('Failed to run scheduled state checks:', error);
      });
    }, interval);
  }, 1000); // Delay first run by 1 second to allow app to initialize
}

/**
 * Example state checks for common resources
 */

// Database connection check
export function createDatabaseCheck(checkFn: () => Promise<boolean>) {
  return async (): Promise<StateCheckResult> => {
    try {
      const connected = await checkFn();
      return {
        healthy: connected,
        issues: connected ? undefined : ['Database connection failed'],
      };
    } catch (error) {
      return {
        healthy: false,
        issues: [`Database check error: ${error.message}`],
        details: { error: error.message, stack: error.stack },
      };
    }
  };
}

// Memory usage check
export function createMemoryCheck(thresholdMB = 1024) {
  return (): StateCheckResult => {
    try {
      const memoryUsage = process.memoryUsage();
      const heapUsedMB = Math.round(memoryUsage.heapUsed / 1024 / 1024);
      const rssUsedMB = Math.round(memoryUsage.rss / 1024 / 1024);
      
      const healthy = heapUsedMB < thresholdMB && rssUsedMB < thresholdMB * 1.5;
      
      return {
        healthy,
        metrics: {
          heapUsedMB,
          rssUsedMB,
          heapTotalMB: Math.round(memoryUsage.heapTotal / 1024 / 1024),
          externalMB: Math.round(memoryUsage.external / 1024 / 1024),
        },
        issues: healthy ? undefined : [
          `Memory usage exceeds threshold: Heap ${heapUsedMB}MB, RSS ${rssUsedMB}MB (threshold: ${thresholdMB}MB)`
        ],
      };
    } catch (error) {
      return {
        healthy: false,
        issues: [`Memory check error: ${error.message}`],
      };
    }
  };
}

// API dependency check
export function createApiDependencyCheck(name: string, url: string, options?: { timeout?: number }) {
  return async (): Promise<StateCheckResult> => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), options?.timeout || 5000);
      
      const startTime = Date.now();
      const response = await fetch(url, { signal: controller.signal });
      const duration = Date.now() - startTime;
      
      clearTimeout(timeout);
      
      const healthy = response.ok;
      
      return {
        healthy,
        metrics: { responseTime: duration },
        issues: healthy ? undefined : [`API dependency ${name} returned status ${response.status}`],
        details: healthy ? undefined : {
          status: response.status,
          statusText: response.statusText,
          url,
        },
      };
    } catch (error) {
      return {
        healthy: false,
        issues: [`API dependency ${name} check failed: ${error.message}`],
        details: { error: error.message, url },
      };
    }
  };
}