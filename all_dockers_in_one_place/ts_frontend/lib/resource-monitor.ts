/**
 * Utility for tracking and logging system resource issues
 */
import { serverIssueLogger } from './server-issue-logger';

// Interface for resource metrics
interface ResourceMetrics {
  memory?: {
    heapUsed: number;
    heapTotal: number;
    rss: number;
    external?: number;
  };
  cpu?: {
    user: number;
    system: number;
  };
  eventLoop?: {
    latency: number;
  };
  custom?: Record<string, number>;
}

// Default thresholds
const DEFAULT_THRESHOLDS = {
  memory: {
    heapUsedPercent: 85, // 85% of heap total
    rssGB: 1.5, // 1.5 GB
  },
  cpu: {
    usagePercent: 80, // 80% CPU usage
  },
  eventLoop: {
    latencyMs: 100, // 100ms event loop latency
  },
};

// Store the last check time to avoid too frequent checks
let lastCheckTime = 0;
const MIN_CHECK_INTERVAL = 10000; // 10 seconds

/**
 * Check system resources and log any issues
 */
export async function checkSystemResources(
  customThresholds?: typeof DEFAULT_THRESHOLDS,
  force = false
): Promise<ResourceMetrics> {
  // Skip if checked recently, unless forced
  const now = Date.now();
  if (!force && now - lastCheckTime < MIN_CHECK_INTERVAL) {
    return {};
  }
  
  lastCheckTime = now;
  
  // Merge custom thresholds with defaults
  const thresholds = {
    ...DEFAULT_THRESHOLDS,
    ...customThresholds,
    memory: {
      ...DEFAULT_THRESHOLDS.memory,
      ...customThresholds?.memory,
    },
    cpu: {
      ...DEFAULT_THRESHOLDS.cpu,
      ...customThresholds?.cpu,
    },
    eventLoop: {
      ...DEFAULT_THRESHOLDS.eventLoop,
      ...customThresholds?.eventLoop,
    },
  };
  
  const metrics: ResourceMetrics = {};
  
  // Check memory usage
  try {
    const memoryUsage = process.memoryUsage();
    metrics.memory = {
      heapUsed: memoryUsage.heapUsed,
      heapTotal: memoryUsage.heapTotal,
      rss: memoryUsage.rss,
      external: memoryUsage.external,
    };
    
    // Calculate heap used percentage
    const heapUsedPercent = (memoryUsage.heapUsed / memoryUsage.heapTotal) * 100;
    const rssMB = memoryUsage.rss / 1024 / 1024;
    const rssGB = rssMB / 1024;
    
    // Log memory issues
    if (heapUsedPercent > thresholds.memory.heapUsedPercent) {
      await serverIssueLogger.high(
        `High heap memory usage: ${heapUsedPercent.toFixed(1)}%`,
        'system_resources',
        {
          memory: {
            heapUsedMB: Math.round(memoryUsage.heapUsed / 1024 / 1024),
            heapTotalMB: Math.round(memoryUsage.heapTotal / 1024 / 1024),
            heapUsedPercent,
            threshold: thresholds.memory.heapUsedPercent,
          },
        }
      );
    }
    
    if (rssGB > thresholds.memory.rssGB) {
      await serverIssueLogger.high(
        `High RSS memory usage: ${rssGB.toFixed(2)} GB`,
        'system_resources',
        {
          memory: {
            rssMB: Math.round(rssMB),
            rssGB,
            threshold: thresholds.memory.rssGB,
          },
        }
      );
    }
  } catch (error) {
    console.error('Failed to check memory usage:', error);
  }
  
  // Check CPU usage
  try {
    if (process.cpuUsage) {
      const startCpu = process.cpuUsage();
      
      // Wait a short time to measure CPU usage
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const endCpu = process.cpuUsage(startCpu);
      metrics.cpu = {
        user: endCpu.user,
        system: endCpu.system,
      };
      
      // Calculate CPU usage percentage (very rough estimate)
      const totalCpu = endCpu.user + endCpu.system;
      const cpuUsagePercent = (totalCpu / 100000) * 100; // Normalize to percentage
      
      // Log CPU issues
      if (cpuUsagePercent > thresholds.cpu.usagePercent) {
        await serverIssueLogger.high(
          `High CPU usage: ${cpuUsagePercent.toFixed(1)}%`,
          'system_resources',
          {
            cpu: {
              user: endCpu.user,
              system: endCpu.system,
              usagePercent: cpuUsagePercent,
              threshold: thresholds.cpu.usagePercent,
            },
          }
        );
      }
    }
  } catch (error) {
    console.error('Failed to check CPU usage:', error);
  }
  
  // Check event loop latency
  try {
    const start = Date.now();
    await new Promise(resolve => setTimeout(resolve, 0));
    const latency = Date.now() - start;
    
    metrics.eventLoop = { latency };
    
    // Log event loop issues
    if (latency > thresholds.eventLoop.latencyMs) {
      await serverIssueLogger.high(
        `High event loop latency: ${latency}ms`,
        'system_resources',
        {
          eventLoop: {
            latency,
            threshold: thresholds.eventLoop.latencyMs,
          },
        }
      );
    }
  } catch (error) {
    console.error('Failed to check event loop latency:', error);
  }
  
  return metrics;
}

/**
 * Schedule periodic resource checks
 */
export function scheduleResourceChecks(intervalSeconds = 60, customThresholds?: typeof DEFAULT_THRESHOLDS) {
  // Only run in Node.js environment
  if (typeof window !== 'undefined') return;
  
  // Convert seconds to milliseconds
  const interval = intervalSeconds * 1000;
  
  // Run initial check
  setTimeout(() => {
    checkSystemResources(customThresholds).catch(error => {
      console.error('Failed to check system resources:', error);
    });
    
    // Set up interval for future checks
    setInterval(() => {
      checkSystemResources(customThresholds).catch(error => {
        console.error('Failed to check system resources:', error);
      });
    }, interval);
  }, 5000); // Delay first run by 5 seconds to allow app to initialize
}

/**
 * Monitor a specific resource and log if it exceeds thresholds
 */
export async function monitorResource(
  name: string,
  getValue: () => Promise<number> | number,
  threshold: number,
  unit = ''
): Promise<void> {
  try {
    const value = await Promise.resolve(getValue());
    
    if (value > threshold) {
      await serverIssueLogger.high(
        `Resource threshold exceeded: ${name}`,
        'system_resources',
        {
          resource: name,
          value,
          threshold,
          unit,
          percentOverThreshold: ((value - threshold) / threshold) * 100,
        }
      );
    }
  } catch (error) {
    console.error(`Failed to monitor resource ${name}:`, error);
  }
}