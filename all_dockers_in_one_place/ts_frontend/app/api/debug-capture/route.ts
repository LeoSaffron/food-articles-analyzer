import { NextRequest, NextResponse } from 'next/server';
import { withApiRoute } from '@/lib/api-route-wrapper';
import { serverLogger } from '@/lib/server-logger';

/**
 * Completely removes cookies and sensitive data from any object
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
 * Removes cookies and sensitive data from headers
 * @param headers The headers object to clean
 * @returns A new object with cookies and sensitive data removed
 */
function cleanHeaders(headers: Record<string, any>): Record<string, any> {
  const cleanedHeaders: Record<string, any> = {};
  
  for (const key in headers) {
    const lowerKey = key.toLowerCase();
    // Skip any cookie-related headers entirely
    if (!['cookie', 'authorization', 'token', 'password', 'secret'].some(k => lowerKey.includes(k))) {
      cleanedHeaders[key] = headers[key];
    }
  }
  
  return cleanedHeaders;
}

/**
 * Inspects the structure of a nested object without including all values
 * and completely removes any cookie-related data
 * @param obj The object to inspect
 * @returns A structure representation of the object with sensitive data removed
 */
function inspectNestedStructure(obj: any, depth: number = 0, maxDepth: number = 3): any {
  // Base case: null or undefined
  if (obj === null || obj === undefined) {
    return obj === null ? 'null' : 'undefined';
  }
  
  // Base case: primitive types
  if (typeof obj !== 'object') {
    return `${typeof obj}: ${String(obj).substring(0, 50)}${String(obj).length > 50 ? '...' : ''}`;
  }
  
  // Base case: max depth reached
  if (depth >= maxDepth) {
    return Array.isArray(obj) ? `Array(${obj.length})` : `Object(${Object.keys(obj).length} keys)`;
  }
  
  // Recursive case: array
  if (Array.isArray(obj)) {
    if (obj.length === 0) return 'Empty Array';
    if (obj.length > 5) {
      return `Array(${obj.length}): [${obj.slice(0, 3).map(item => inspectNestedStructure(item, depth + 1, maxDepth)).join(', ')}, ... ${obj.length - 3} more items]`;
    }
    return `Array(${obj.length}): [${obj.map(item => inspectNestedStructure(item, depth + 1, maxDepth)).join(', ')}]`;
  }
  
  // Recursive case: object
  const keys = Object.keys(obj).filter(key => {
    // Filter out any cookie-related keys
    const lowerKey = key.toLowerCase();
    return !['cookie', 'cookies', 'authorization', 'token', 'password', 'secret'].some(k => lowerKey.includes(k));
  });
  
  if (keys.length === 0) return 'Empty Object';
  
  const result: Record<string, any> = {};
  for (const key of keys) {
    if (keys.length > 10 && result.hasOwnProperty('...')) {
      result['...'] = `${keys.length - Object.keys(result).length + 1} more keys`;
      break;
    }
    result[key] = inspectNestedStructure(obj[key], depth + 1, maxDepth);
  }
  
  return result;
}

/**
 * Endpoint for capturing detailed data context before an error occurs
 * This is specifically designed to help debug client-side errors
 */
export const POST = withApiRoute(async (request: NextRequest) => {
  const requestId = request.headers.get('X-Request-ID') || `req-${Date.now()}`;
  
  try {
    // Parse the request body
    const body = await request.json();
    
    // Extract component name and data from the request
    const { componentName, componentProps, errorInfo, contextData } = body;
    
    // Remove any sensitive data including cookies
    const cleanedProps = removeSensitiveData(componentProps);
    const cleanedContext = removeSensitiveData(contextData);
    
    // Log detailed information about the component and its props
    await serverLogger.debug(
      `Debug context capture before error in ${componentName || 'unknown component'}`,
      {
        requestId,
        timestamp: new Date().toISOString(),
        componentName,
        errorInfo: removeSensitiveData(errorInfo),
        // Detailed inspection of component props
        propsInspection: inspectNestedStructure(cleanedProps),
        // Raw props for reference (with sensitive data removed)
        rawProps: cleanedProps,
        // Additional context data (with sensitive data removed)
        contextData: cleanedContext,
        // Browser and environment info
        environment: {
          userAgent: request.headers.get('user-agent'),
          referer: request.headers.get('referer'),
          host: request.headers.get('host')
        }
      }
    );
    
    return NextResponse.json({
      success: true,
      message: 'Debug context captured successfully',
      requestId,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    // Log the error
    await serverLogger.error(
      `Debug context capture error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      {
        requestId,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        timestamp: new Date().toISOString(),
        // Don't include raw body as it might contain cookies
        bodyType: typeof request.body
      }
    );
    
    // Return error response
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      requestId
    }, { status: 500 });
  }
});

/**
 * Endpoint for capturing detailed data about specific objects that might be causing errors
 */
export const PUT = withApiRoute(async (request: NextRequest) => {
  const requestId = request.headers.get('X-Request-ID') || `req-${Date.now()}`;
  
  try {
    // Parse the request body
    const body = await request.json();
    
    // Extract the object to inspect and metadata
    const { objectToInspect, location, operation, additionalContext } = body;
    
    // Remove any sensitive data including cookies
    const cleanedObject = removeSensitiveData(objectToInspect);
    const cleanedContext = removeSensitiveData(additionalContext);
    
    // Perform detailed inspection of the object
    const inspection = {
      type: typeof cleanedObject,
      isNull: cleanedObject === null,
      isUndefined: cleanedObject === undefined,
      isArray: Array.isArray(cleanedObject),
      isEmpty: cleanedObject === null || cleanedObject === undefined ? true : 
              (typeof cleanedObject === 'object' ? 
                (Array.isArray(cleanedObject) ? cleanedObject.length === 0 : Object.keys(cleanedObject).length === 0) : 
                false),
      keys: cleanedObject && typeof cleanedObject === 'object' ? Object.keys(cleanedObject) : [],
      structure: inspectNestedStructure(cleanedObject, 0, 5),  // Deeper inspection for debugging
      // For arrays, provide additional info
      arrayInfo: Array.isArray(cleanedObject) ? {
        length: cleanedObject.length,
        firstFewItems: cleanedObject.slice(0, 5).map(item => typeof item),
        containsNullOrUndefined: cleanedObject.some(item => item === null || item === undefined)
      } : null,
      // For objects, check for common issues
      objectIssues: cleanedObject && typeof cleanedObject === 'object' && !Array.isArray(cleanedObject) ? {
        hasNullPrototype: Object.getPrototypeOf(cleanedObject) === null,
        hasCircularReferences: false, // Hard to detect without custom logic
        hasNullOrUndefinedValues: Object.values(cleanedObject).some(val => val === null || val === undefined)
      } : null
    };
    
    // Log the inspection results
    await serverLogger.debug(
      `Object inspection at ${location || 'unknown location'}`,
      {
        requestId,
        timestamp: new Date().toISOString(),
        location,
        operation,
        inspection,
        additionalContext: cleanedContext,
        // Include raw object for reference if it's not too large (with sensitive data removed)
        rawObject: cleanedObject && typeof cleanedObject === 'object' ? 
                  (JSON.stringify(cleanedObject).length < 10000 ? cleanedObject : '[Object too large to include]') : 
                  cleanedObject
      }
    );
    
    return NextResponse.json({
      success: true,
      message: 'Object inspection completed',
      requestId,
      inspection: {
        type: inspection.type,
        isNull: inspection.isNull,
        isUndefined: inspection.isUndefined,
        isArray: inspection.isArray,
        isEmpty: inspection.isEmpty,
        keyCount: inspection.keys.length
      }
    });
  } catch (error) {
    // Log the error
    await serverLogger.error(
      `Object inspection error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      {
        requestId,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        timestamp: new Date().toISOString()
      }
    );
    
    // Return error response
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      requestId
    }, { status: 500 });
  }
});