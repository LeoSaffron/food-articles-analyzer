type LogLevel = 'access' | 'debug' | 'error' | 'info' | 'warn'

// Store request ID for the current session
let currentRequestId: string | null = null;

class Logger {
  /**
   * Set the request ID for the current session
   */
  static setRequestId(requestId: string) {
    currentRequestId = requestId;
  }

  /**
   * Get the current request ID or generate a new one
   */
  static getRequestId(): string {
    if (!currentRequestId) {
      // Generate a simple client-side ID if we don't have one
      currentRequestId = Math.random().toString(36).substring(2, 15);
    }
    return currentRequestId;
  }

  /**
   * Send log to server
   */
  private static async log(level: LogLevel, message: string, data?: any) {
    if (level === 'debug' && process.env.ENABLE_DEBUG !== 'true') {
      return;
    }

    const requestId = this.getRequestId();
    const timestamp = new Date().toISOString();

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      const logData = { timestamp, level, message, requestId, ...(data && { data }) };
      console.log(JSON.stringify(logData));
    }

    try {
      const response = await fetch('/api/log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': requestId
        },
        body: JSON.stringify({ level, message, data, requestId }),
      });

      if (!response.ok) {
        console.error('Logging failed:', await response.text());
      }
    } catch (error) {
      console.error('Logging error:', error);
    }
  }

  static access(message: string, data?: any) {
    this.log('access', message, data);
  }

  static info(message: string, data?: any) {
    this.log('info', message, data);
  }

  static debug(message: string, data?: any) {
    this.log('debug', message, data);
  }

  static warn(message: string, data?: any) {
    this.log('warn', message, data);
  }

  static error(message: string, error?: Error, data?: any) {
    this.log('error', message, { 
      error: error?.message, 
      stack: error?.stack, 
      ...data 
    });
  }

  /**
   * Log performance metrics
   */
  static performance(name: string, duration: number, data?: any) {
    this.log('info', `Performance: ${name}`, { 
      performance: true,
      duration,
      ...data
    });
  }
}

export const logger = Logger;