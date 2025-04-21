'use client';

import { useEffect } from 'react';
import { usePathname, useSearchParams } from 'next/navigation';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    // Log client-side error to server
    console.error('Client-side error:', error);
    
    // Capture browser information
    const browserInfo = {
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
      screenSize: {
        width: window.innerWidth,
        height: window.innerHeight,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
      },
      timestamp: new Date().toISOString(),
      referrer: document.referrer,
      url: window.location.href,
    };
    
    // Capture application state
    const appState = {
      pathname,
      query: Object.fromEntries(searchParams.entries()),
      localStorage: (() => {
        try {
          // Only capture non-sensitive localStorage items
          const safeItems = {};
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && !key.toLowerCase().includes('token') && !key.toLowerCase().includes('auth')) {
              safeItems[key] = localStorage.getItem(key);
            }
          }
          return safeItems;
        } catch (e) {
          return { error: 'Could not access localStorage' };
        }
      })(),
    };
    
    // Send comprehensive error data to server for logging
    fetch('/api/log/error', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Error-Source': 'client',
      },
      body: JSON.stringify({
        message: `Client error on ${pathname}`,
        error: {
          message: error.message,
          stack: error.stack,
          digest: error.digest,
          type: error.name,
          constructor: error.constructor?.name,
        },
        url: pathname,
        context: {
          browser: browserInfo,
          appState,
          performance: {
            navigation: JSON.parse(JSON.stringify(performance.timing || {})),
            memory: performance.memory ? {
              jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
              totalJSHeapSize: performance.memory.totalJSHeapSize,
              usedJSHeapSize: performance.memory.usedJSHeapSize,
            } : null,
          },
        },
      }),
    }).catch(console.error); // Don't let this throw
  }, [error, pathname, searchParams]);

  return (
    <div className="error-container p-6 max-w-4xl mx-auto my-8 bg-gray-50 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold text-red-600 mb-4">Something went wrong!</h2>
      <p className="mb-4">An error occurred while processing your request.</p>
      <p className="text-sm text-gray-500 mb-6">Error ID: {error.digest || 'unknown'}</p>
      <button
        onClick={reset}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        Try again
      </button>
    </div>
  );
}