'use client';

import { useEffect } from 'react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log global error to server with full context
    console.error('Global application error:', error);
    
    // Capture system information
    const systemInfo = {
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
      url: window.location.href,
      timestamp: new Date().toISOString(),
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
      screen: {
        width: window.screen.width,
        height: window.screen.height,
        colorDepth: window.screen.colorDepth,
        orientation: window.screen.orientation?.type || 'unknown',
      },
      connection: navigator.connection ? {
        effectiveType: navigator.connection.effectiveType,
        downlink: navigator.connection.downlink,
        rtt: navigator.connection.rtt,
        saveData: navigator.connection.saveData,
      } : null,
    };
    
    // Send comprehensive error data to server
    fetch('/api/log/error', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Error-Source': 'global',
      },
      body: JSON.stringify({
        message: `Global application error: ${error.message}`,
        error: {
          message: error.message,
          stack: error.stack,
          digest: error.digest,
          type: error.name,
          constructor: error.constructor?.name,
        },
        url: window.location.href,
        context: {
          system: systemInfo,
          performance: {
            timing: JSON.parse(JSON.stringify(performance.timing || {})),
            navigation: JSON.parse(JSON.stringify(performance.navigation || {})),
            memory: performance.memory ? {
              jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
              totalJSHeapSize: performance.memory.totalJSHeapSize,
              usedJSHeapSize: performance.memory.usedJSHeapSize,
            } : null,
          },
        },
      }),
    }).catch(console.error); // Don't let this throw
  }, [error]);

  return (
    <html>
      <body>
        <div style={{
          padding: '2rem',
          maxWidth: '64rem',
          margin: '0 auto',
          backgroundColor: '#f9fafb',
          borderRadius: '0.5rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#dc2626', marginBottom: '1rem' }}>
            Something went wrong!
          </h1>
          <p style={{ marginBottom: '1rem' }}>
            A critical error occurred in the application.
          </p>
          <p style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '1.5rem' }}>
            Error ID: {error.digest || 'unknown'}
          </p>
          <button
            onClick={reset}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#2563eb',
              color: 'white',
              borderRadius: '0.25rem',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}