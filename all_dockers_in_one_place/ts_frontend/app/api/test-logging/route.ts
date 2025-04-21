import { NextRequest, NextResponse } from 'next/server';
import { withApiRoute } from '@/lib/api-route-wrapper';
import { serverLogger } from '@/lib/server-logger';

/**
 * Removes cookies and sensitive data from headers
 * @param headers The headers object to clean
 * @returns A new object with cookies and sensitive data removed
 */
function cleanSensitiveHeaders(headers: Record<string, any>): Record<string, any> {
  // Remove cookies and sensitive auth data completely
  const keysToRemove = [
    'cookie', 'authorization', 'token', 'password', 'secret'
  ];
  
  const cleanedHeaders: Record<string, any> = {};
  
  // Include all headers except the sensitive ones
  for (const key in headers) {
    const lowerKey = key.toLowerCase();
    if (!keysToRemove.some(badKey => lowerKey.includes(badKey))) {
      cleanedHeaders[key] = headers[key];
    }
  }
  
  return cleanedHeaders;
}

/**
 * Logs API communication details
 * @param message The log message
 * @param details Additional details to log
 * @param requestId The request ID for correlation
 */
async function logApiCommunication(
  message: string, 
  details: Record<string, any>, 
  requestId: string
): Promise<void> {
  await serverLogger.info(
    `API Communication: ${message}`,
    {
      requestId,
      timestamp: new Date().toISOString(),
      ...details
    }
  );
}

/**
 * Test endpoint that demonstrates API communication logging
 */
export const GET = withApiRoute(async (request: NextRequest) => {
  const requestId = request.headers.get('X-Request-ID') || `req-${Date.now()}`;
  const url = new URL(request.url);
  const query = url.searchParams.get('q') || '';
  const shouldFail = url.searchParams.has('fail');
  
  try {
    // Clean headers - only remove cookies and sensitive data
    const headers = Object.fromEntries(request.headers.entries());
    const cleanedHeaders = cleanSensitiveHeaders(headers);
    
    // Log the incoming request
    await logApiCommunication('Request received', {
      endpoint: url.pathname,
      method: request.method,
      query: Object.fromEntries(url.searchParams.entries()),
      headers: cleanedHeaders
    }, requestId);
    
    // Simulate API processing steps
    await logApiCommunication('Processing request', {
      query,
      processingStage: 'initial',
      timestamp: new Date().toISOString()
    }, requestId);
    
    // Simulate database lookup
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Log database results - this is the kind of message you want to see
    await logApiCommunication('Recipe found in DB', {
      recipeId: 'recipe-123',
      recipeName: 'Chocolate Cake',
      source: 'database',
      queryTime: '285ms',
      cacheHit: true
    }, requestId);
    
    // Simulate processing the recipe data
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Log processing details
    await logApiCommunication('Recipe data processed', {
      recipeId: 'recipe-123',
      processingSteps: ['normalize', 'enrich', 'format'],
      processingTime: '198ms',
      ingredientsCount: 12,
      stepsCount: 8
    }, requestId);
    
    // Simulate error if requested
    if (shouldFail) {
      throw new Error('Failed to process recipe data');
    }
    
    // Prepare response data
    const responseData = {
      id: 'recipe-123',
      name: 'Chocolate Cake',
      ingredients: [
        '2 cups flour',
        '1 cup sugar',
        '3/4 cup cocoa powder',
        // More ingredients...
      ],
      steps: [
        'Preheat oven to 350°F',
        'Mix dry ingredients',
        // More steps...
      ],
      nutrition: {
        calories: 350,
        protein: '5g',
        fat: '12g'
      }
    };
    
    // Log the response being sent
    await logApiCommunication('Sending response', {
      responseSize: JSON.stringify(responseData).length,
      responseTime: '485ms',
      cacheStatus: 'stored',
      status: 200
    }, requestId);
    
    // Return the response
    return NextResponse.json({
      success: true,
      message: 'Recipe found and processed successfully',
      requestId,
      data: responseData
    });
  } catch (error) {
    // Log the error with full details
    await serverLogger.error(
      `API Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      {
        requestId,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        query,
        endpoint: url.pathname,
        timestamp: new Date().toISOString()
      }
    );
    
    // Return error response
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      requestId
    }, { status: 500 });
  } finally {
    // Always log request completion
    await logApiCommunication('Request completed', {
      endpoint: url.pathname,
      method: request.method,
      duration: `${Math.floor(Math.random() * 500)}ms`,
      status: shouldFail ? 500 : 200
    }, requestId);
  }
});

/**
 * Endpoint for streaming API responses with detailed logging
 */
export const POST = withApiRoute(async (request: NextRequest) => {
  const requestId = request.headers.get('X-Request-ID') || `req-${Date.now()}`;
  
  try {
    // Parse the request body
    const body = await request.json();
    const query = body.query || '';
    
    // Log the request received
    await logApiCommunication('Stream request received', {
      requestId,
      query,
      streamMode: true,
      headers: cleanSensitiveHeaders(Object.fromEntries(request.headers.entries()))
    }, requestId);
    
    // Create a stream response
    const stream = new ReadableStream({
      async start(controller) {
        // Log stream started
        await logApiCommunication('Stream started', {
          requestId,
          timestamp: new Date().toISOString()
        }, requestId);
        
        // Simulate database lookup
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Log database hit - this is the kind of message you want to see
        await logApiCommunication('Recipe found in DB', {
          recipeId: 'recipe-456',
          recipeName: 'Pasta Carbonara',
          source: 'database',
          queryTime: '185ms'
        }, requestId);
        
        // Send first chunk
        const chunk1 = JSON.stringify({
          type: 'partial',
          data: {
            id: 'recipe-456',
            name: 'Pasta Carbonara',
            ingredients: ['Pasta', 'Eggs', 'Bacon']
          }
        });
        controller.enqueue(new TextEncoder().encode(chunk1 + '\n'));
        
        // Log first chunk sent
        await logApiCommunication('Stream chunk sent', {
          requestId,
          chunkNumber: 1,
          chunkSize: chunk1.length,
          content: 'Basic recipe info'
        }, requestId);
        
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Log processing step - this is the kind of message you want to see
        await logApiCommunication('Processing recipe instructions', {
          recipeId: 'recipe-456',
          step: 'formatting',
          progress: '50%'
        }, requestId);
        
        // Send second chunk
        const chunk2 = JSON.stringify({
          type: 'partial',
          data: {
            steps: [
              'Boil pasta until al dente',
              'Fry bacon until crispy',
              'Mix eggs with cheese'
            ]
          }
        });
        controller.enqueue(new TextEncoder().encode(chunk2 + '\n'));
        
        // Log second chunk sent
        await logApiCommunication('Stream chunk sent', {
          requestId,
          chunkNumber: 2,
          chunkSize: chunk2.length,
          content: 'Recipe instructions'
        }, requestId);
        
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Log final processing - this is the kind of message you want to see
        await logApiCommunication('Finalizing recipe data', {
          recipeId: 'recipe-456',
          nutritionCalculated: true,
          tagsGenerated: ['italian', 'pasta', 'quick']
        }, requestId);
        
        // Send final chunk
        const chunk3 = JSON.stringify({
          type: 'final',
          data: {
            nutrition: {
              calories: 450,
              protein: '22g',
              fat: '18g'
            },
            tags: ['italian', 'pasta', 'quick']
          }
        });
        controller.enqueue(new TextEncoder().encode(chunk3 + '\n'));
        
        // Log final chunk sent
        await logApiCommunication('Stream completed', {
          requestId,
          totalChunks: 3,
          totalTime: '685ms'
        }, requestId);
        
        controller.close();
      }
    });
    
    return new Response(stream, {
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId
      }
    });
  } catch (error) {
    // Log the error
    await serverLogger.error(
      `Stream API Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
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

/**
 * Endpoint for capturing detailed request data before errors
 */
export const PATCH = withApiRoute(async (request: NextRequest) => {
  const requestId = request.headers.get('X-Request-ID') || `req-${Date.now()}`;
  
  try {
    // Parse the request body
    const body = await request.json();
    
    // Clean the body to remove any cookies or sensitive data
    const cleanBody = (obj: any): any => {
      if (!obj || typeof obj !== 'object') return obj;
      
      if (Array.isArray(obj)) {
        return obj.map(item => cleanBody(item));
      }
      
      const result: Record<string, any> = {};
      for (const [key, value] of Object.entries(obj)) {
        // Skip any cookie-related keys
        const lowerKey = key.toLowerCase();
        if (['cookie', 'cookies', 'authorization', 'token', 'password', 'secret'].some(k => lowerKey.includes(k))) {
          continue;
        }
        
        // Recursively clean nested objects
        result[key] = typeof value === 'object' && value !== null
          ? cleanBody(value)
          : value;
      }
      
      return result;
    };
    
    const cleanedBody = cleanBody(body);
    const cleanedHeaders = cleanSensitiveHeaders(Object.fromEntries(request.headers.entries()));
    
    // Log the entire request payload with detailed context
    await serverLogger.debug(
      `Pre-error context capture`,
      {
        requestId,
        timestamp: new Date().toISOString(),
        endpoint: request.url,
        method: request.method,
        headers: cleanedHeaders,
        payload: cleanedBody,
        // Add detailed inspection of potentially problematic data
        dataInspection: {
          payloadType: typeof cleanedBody,
          isNull: cleanedBody === null,
          isUndefined: cleanedBody === undefined,
          hasData: cleanedBody && Object.keys(cleanedBody).length > 0,
          dataKeys: cleanedBody ? Object.keys(cleanedBody) : [],
          nestedStructure: inspectNestedStructure(cleanedBody)
        }
      }
    );
    
    return NextResponse.json({
      success: true,
      message: 'Context data captured successfully',
      requestId,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    // Log the error
    await serverLogger.error(
      `Context capture error: ${error instanceof Error ? error.message : 'Unknown error'}`,
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