/**
 * Client-side debugging utilities
 * These functions help capture and send debugging information to the server
 * before errors occur
 */

/**
 * Removes cookies and sensitive data from any object
 * @param obj The object to clean
 * @returns A new object with cookies and sensitive data removed
 */
function removeSensitiveData(obj: any): any {
  if (!obj || typeof obj !== 'object') {
    return obj;
  }
  
  // For arrays, map and clean each item
  if (Array.isArray(obj)) {
    return obj.map(item => removeSensitiveData(item));
  }
  
  // For objects, clean each property
  const result: Record<string, any> = {};
  
  for (const [key, value] of Object.entries(obj)) {
    // Skip any cookie-related keys entirely
    const lowerKey = key.toLowerCase();
    if (['cookie', 'cookies', 'authorization', 'token', 'password', 'secret'].some(k => lowerKey.includes(k))) {
      continue;
    }
    
    // Recursively clean nested objects
    result[key] = typeof value === 'object' && value !== null
      ? removeSensitiveData(value)
      : value;
  }
  
  return result;
}

/**
 * Sends debugging context to the server before an error might occur
 * @param componentName The name of the component where the debug is happening
 * @param componentProps The props of the component
 * @param contextData Additional context data that might be useful
 */
export async function captureDebugContext(
  componentName: string,
  componentProps: any,
  contextData?: Record<string, any>
): Promise<void> {
  try {
    // Generate a request ID
    const requestId = `debug-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    
    // Remove any cookies or sensitive data
    const cleanedProps = removeSensitiveData(componentProps);
    const cleanedContext = removeSensitiveData(contextData || {});
    
    // Prepare the payload
    const payload = {
      componentName,
      componentProps: cleanedProps,
      contextData: cleanedContext,
      browserInfo: {
        userAgent: navigator.userAgent,
        language: navigator.language,
        platform: navigator.platform,
        screenSize: {
          width: window.innerWidth,
          height: window.innerHeight,
          screenWidth: window.screen.width,
          screenHeight: window.screen.height
        },
        timestamp: new Date().toISOString(),
        referrer: document.referrer,
        url: window.location.href
      }
    };
    
    // Send the debug data to the server
    const response = await fetch('/api/debug-capture', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      console.warn('Failed to send debug context:', await response.text());
    }
  } catch (error) {
    // Don't let debugging errors affect the application
    console.warn('Error sending debug context:', error);
  }
}

/**
 * Inspects a specific object that might be causing errors
 * @param objectToInspect The object to inspect
 * @param location Where in the code this inspection is happening
 * @param operation What operation was being performed
 * @param additionalContext Any additional context
 */
export async function inspectPotentialErrorObject(
  objectToInspect: any,
  location: string,
  operation: string,
  additionalContext?: Record<string, any>
): Promise<void> {
  try {
    // Generate a request ID
    const requestId = `inspect-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    
    // Remove any cookies or sensitive data
    const cleanedObject = removeSensitiveData(objectToInspect);
    const cleanedContext = removeSensitiveData(additionalContext || {});
    
    // Prepare the payload
    const payload = {
      objectToInspect: cleanedObject,
      location,
      operation,
      additionalContext: cleanedContext,
      browserInfo: {
        userAgent: navigator.userAgent,
        url: window.location.href,
        timestamp: new Date().toISOString()
      }
    };
    
    // Send the inspection data to the server
    const response = await fetch('/api/debug-capture', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      console.warn('Failed to send object inspection:', await response.text());
    }
  } catch (error) {
    // Don't let debugging errors affect the application
    console.warn('Error sending object inspection:', error);
  }
}

/**
 * Creates a wrapped version of Object.entries that logs debugging information
 * if the input is null or undefined
 * @returns A wrapped version of Object.entries
 */
export function createSafeObjectEntries() {
  const originalEntries = Object.entries;
  
  return function safeObjectEntries(obj: any): [string, any][] {
    // Check if the object is null or undefined
    if (obj === null || obj === undefined) {
      // Capture debug information about the caller
      const stack = new Error().stack || '';
      const callerInfo = stack.split('\n')[2] || 'unknown';
      
      // Log the error locally
      console.error(`Object.entries called with ${obj === null ? 'null' : 'undefined'} at ${callerInfo}`);
      
      // Send debug information to the server
      inspectPotentialErrorObject(
        obj,
        callerInfo,
        'Object.entries',
        removeSensitiveData({
          stack,
          timestamp: new Date().toISOString(),
          url: window.location.href,
          path: window.location.pathname
        })
      );
      
      // Return an empty array instead of throwing
      return [];
    }
    
    // Call the original function
    return originalEntries(obj);
  };
}

/**
 * Higher-order component wrapper to add debug logging before rendering
 * @param Component The component to wrap
 * @param componentName The name of the component (for logging)
 */
export function withDebugLogging<P>(Component: React.ComponentType<P>, componentName: string) {
  return function DebugWrappedComponent(props: P) {
    // Log component props before rendering
    captureDebugContext(componentName, props);
    
    // Render the original component
    return <Component {...props} />;
  };
}