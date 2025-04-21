/**
 * Utilities for debugging issues with Object.entries
 */

import { inspectPotentialErrorObject } from './client-debug';

/**
 * Removes cookies and sensitive data from any object
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
 * A safe version of Object.entries that won't throw on null/undefined
 * and will log debugging information
 * @param obj The object to get entries from
 * @param location The location in code where this is called
 * @returns An array of key-value pairs or an empty array if obj is null/undefined
 */
export function safeObjectEntries(obj: any, location: string = 'unknown'): [string, any][] {
  // Check if the object is null or undefined
  if (obj === null || obj === undefined) {
    // Capture debug information
    const stack = new Error().stack || '';
    
    // Log the error locally
    console.error(`Object.entries called with ${obj === null ? 'null' : 'undefined'} at ${location}`);
    console.error('Stack trace:', stack);
    
    // Send debug information to the server
    inspectPotentialErrorObject(
      obj,
      location,
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
  
  // For non-objects that aren't null/undefined, also log but don't throw
  if (typeof obj !== 'object' || Array.isArray(obj)) {
    console.warn(`Object.entries called with non-object type: ${typeof obj} at ${location}`);
    inspectPotentialErrorObject(
      obj,
      location,
      'Object.entries with non-object',
      removeSensitiveData({ type: typeof obj, isArray: Array.isArray(obj) })
    );
    
    // Convert primitives to objects as Object.entries would
    return Object.entries(Object(obj));
  }
  
  // Normal case - just call Object.entries
  return Object.entries(obj);
}

/**
 * Monkey patches the global Object.entries to use the safe version
 * WARNING: Use with caution as this affects all code
 */
export function monkeyPatchObjectEntries(): () => void {
  // Store the original function
  const originalEntries = Object.entries;
  
  // Replace with our safe version
  Object.entries = function safeEntries(obj: any): [string, any][] {
    // Get caller information from stack trace
    const stack = new Error().stack || '';
    const stackLines = stack.split('\n');
    const callerInfo = stackLines.length > 2 ? stackLines[2] : 'unknown';
    
    return safeObjectEntries(obj, callerInfo);
  };
  
  // Return a function to restore the original
  return function restoreOriginal() {
    Object.entries = originalEntries;
  };
}

/**
 * Wraps a component with debugging for Object.entries calls
 * @param Component The component to wrap
 * @param componentName The name of the component for debugging
 */
export function withObjectEntriesDebug<P>(Component: React.ComponentType<P>, componentName: string) {
  return function WrappedWithObjectEntriesDebug(props: P) {
    // Clean props before logging
    const cleanedProps = removeSensitiveData(props);
    
    // Log the props before rendering
    console.log(`Rendering ${componentName} with props:`, cleanedProps);
    
    // Check if any prop is null or undefined
    for (const key in props) {
      // Skip cookie-related props
      const lowerKey = key.toLowerCase();
      if (['cookie', 'cookies', 'authorization', 'token', 'password', 'secret'].some(k => lowerKey.includes(k))) {
        continue;
      }
      
      if (props[key] === null || props[key] === undefined) {
        console.warn(`${componentName} received ${props[key] === null ? 'null' : 'undefined'} for prop '${key}'`);
      }
    }
    
    // Render the component
    return <Component {...props} />;
  };
}