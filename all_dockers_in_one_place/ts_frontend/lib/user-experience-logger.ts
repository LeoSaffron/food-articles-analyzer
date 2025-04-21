/**
 * Client-side utility for tracking and logging user experience issues
 */

import { appLogger } from './app-logger';

// Interface for user interaction data
interface UserInteraction {
  action: string;
  element?: string;
  location: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

// Interface for user experience issue
interface UxIssue {
  type: 'navigation' | 'interaction' | 'rendering' | 'performance' | 'custom';
  subtype: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
  metadata?: Record<string, any>;
  interactions?: UserInteraction[];
}

// Store recent user interactions
const recentInteractions: UserInteraction[] = [];
const MAX_INTERACTIONS = 20;

/**
 * Track a user interaction
 */
export function trackInteraction(action: string, element?: string, metadata?: Record<string, any>) {
  if (typeof window === 'undefined') return;
  
  const interaction: UserInteraction = {
    action,
    element,
    location: window.location.href,
    timestamp: Date.now(),
    metadata,
  };
  
  // Add to recent interactions
  recentInteractions.push(interaction);
  
  // Keep only the most recent interactions
  if (recentInteractions.length > MAX_INTERACTIONS) {
    recentInteractions.shift();
  }
}

/**
 * Log a user experience issue
 */
export function logUxIssue(issue: Omit<UxIssue, 'interactions'>) {
  if (typeof window === 'undefined') return;
  
  // Create full issue with recent interactions
  const fullIssue: UxIssue = {
    ...issue,
    interactions: [...recentInteractions],
  };
  
  // Log based on severity
  switch (issue.severity) {
    case 'high':
      appLogger.error(
        `UX Issue: ${issue.message}`,
        new Error(`UX Issue: ${issue.type}/${issue.subtype}`),
        fullIssue
      );
      break;
    case 'medium':
      appLogger.warn(`UX Issue: ${issue.message}`, fullIssue);
      break;
    case 'low':
    default:
      appLogger.info(`UX Issue: ${issue.message}`, fullIssue);
      break;
  }
}

/**
 * Convenience methods for logging specific UX issues
 */
export const uxIssueLogger = {
  // Navigation issues
  navigationFailed: (from: string, to: string, error?: Error) => {
    logUxIssue({
      type: 'navigation',
      subtype: 'failed',
      message: `Navigation failed from ${from} to ${to}`,
      severity: 'high',
      metadata: {
        from,
        to,
        error: error ? { message: error.message, stack: error.stack } : undefined,
      },
    });
  },
  
  navigationTimeout: (from: string, to: string, durationMs: number) => {
    logUxIssue({
      type: 'navigation',
      subtype: 'timeout',
      message: `Slow navigation from ${from} to ${to} (${durationMs}ms)`,
      severity: 'medium',
      metadata: { from, to, durationMs },
    });
  },
  
  // Interaction issues
  interactionFailed: (action: string, element: string, error?: Error) => {
    logUxIssue({
      type: 'interaction',
      subtype: 'failed',
      message: `Interaction failed: ${action} on ${element}`,
      severity: 'high',
      metadata: {
        action,
        element,
        error: error ? { message: error.message, stack: error.stack } : undefined,
      },
    });
  },
  
  interactionDelayed: (action: string, element: string, durationMs: number) => {
    logUxIssue({
      type: 'interaction',
      subtype: 'delayed',
      message: `Delayed interaction: ${action} on ${element} (${durationMs}ms)`,
      severity: 'medium',
      metadata: { action, element, durationMs },
    });
  },
  
  // Rendering issues
  renderingError: (component: string, error?: Error) => {
    logUxIssue({
      type: 'rendering',
      subtype: 'error',
      message: `Rendering error in ${component}`,
      severity: 'high',
      metadata: {
        component,
        error: error ? { message: error.message, stack: error.stack } : undefined,
      },
    });
  },
  
  renderingPerformance: (component: string, durationMs: number, threshold: number) => {
    if (durationMs <= threshold) return;
    
    logUxIssue({
      type: 'rendering',
      subtype: 'performance',
      message: `Slow rendering in ${component} (${durationMs}ms)`,
      severity: durationMs > threshold * 2 ? 'high' : 'medium',
      metadata: { component, durationMs, threshold },
    });
  },
  
  // Performance issues
  layoutShift: (cls: number) => {
    if (cls < 0.1) return; // Only log significant layout shifts
    
    logUxIssue({
      type: 'performance',
      subtype: 'layout_shift',
      message: `Cumulative Layout Shift: ${cls.toFixed(3)}`,
      severity: cls > 0.25 ? 'high' : 'medium',
      metadata: { cls },
    });
  },
  
  longTask: (duration: number, taskInfo?: any) => {
    logUxIssue({
      type: 'performance',
      subtype: 'long_task',
      message: `Long task detected: ${duration}ms`,
      severity: duration > 100 ? 'high' : 'medium',
      metadata: { duration, taskInfo },
    });
  },
  
  // Custom issues
  custom: (message: string, subtype: string, severity: 'low' | 'medium' | 'high', metadata?: any) => {
    logUxIssue({
      type: 'custom',
      subtype,
      message,
      severity,
      metadata,
    });
  },
};

/**
 * Set up performance monitoring
 */
export function setupPerformanceMonitoring() {
  if (typeof window === 'undefined') return;
  
  // Monitor long tasks
  if ('PerformanceObserver' in window) {
    try {
      const longTaskObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          // Only log tasks longer than 50ms
          if (entry.duration > 50) {
            uxIssueLogger.longTask(entry.duration, {
              name: entry.name,
              startTime: entry.startTime,
              duration: entry.duration,
            });
          }
        });
      });
      
      longTaskObserver.observe({ entryTypes: ['longtask'] });
      
      // Monitor layout shifts
      const layoutShiftObserver = new PerformanceObserver((list) => {
        let cumulativeLayoutShift = 0;
        
        list.getEntries().forEach((entry) => {
          if (!entry.hadRecentInput) {
            cumulativeLayoutShift += entry.value;
          }
        });
        
        if (cumulativeLayoutShift > 0.05) {
          uxIssueLogger.layoutShift(cumulativeLayoutShift);
        }
      });
      
      layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
    } catch (error) {
      console.error('Failed to set up performance monitoring:', error);
    }
  }
  
  // Monitor navigation performance
  window.addEventListener('load', () => {
    setTimeout(() => {
      if (window.performance && window.performance.timing) {
        const timing = window.performance.timing;
        const navigationStart = timing.navigationStart;
        const loadTime = timing.loadEventEnd - navigationStart;
        const domContentLoaded = timing.domContentLoadedEventEnd - navigationStart;
        const firstPaint = timing.responseEnd - navigationStart;
        
        // Log slow page loads (over 3 seconds)
        if (loadTime > 3000) {
          appLogger.warn('Slow page load detected', {
            url: window.location.href,
            loadTime,
            domContentLoaded,
            firstPaint,
            timing: JSON.parse(JSON.stringify(timing)),
          });
        }
      }
    }, 0);
  });
}