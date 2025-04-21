'use client';

import { useEffect } from 'react';
import { usePathname, useSearchParams } from 'next/navigation';
import Link from 'next/link';

export default function NotFound() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  
  useEffect(() => {
    // Log the 404 error to the server
    fetch('/api/log/error', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Error-Source': 'not-found',
      },
      body: JSON.stringify({
        message: `404 Not Found: ${pathname}`,
        error: {
          type: '404',
          path: pathname,
          query: Object.fromEntries(searchParams.entries()),
        },
        url: pathname,
        context: {
          browser: {
            userAgent: navigator.userAgent,
            language: navigator.language,
            referrer: document.referrer,
          },
          timestamp: new Date().toISOString(),
        },
      }),
    }).catch(console.error);
  }, [pathname, searchParams]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center">
      <h1 className="text-4xl font-bold text-red-600 mb-4">404 - Page Not Found</h1>
      <p className="mb-6 text-lg">The page you are looking for does not exist.</p>
      <p className="mb-8 text-gray-600">
        URL: <code className="bg-gray-100 px-2 py-1 rounded">{pathname}</code>
      </p>
      <Link 
        href="/"
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        Return to Home
      </Link>
    </div>
  );
}