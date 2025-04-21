const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const fs = require('fs');
const path = require('path');

// Define log paths
const LOG_DIR = path.join(process.cwd(), 'logs');
const ACCESS_LOG = path.join(LOG_DIR, 'access.log');
const ERROR_LOG = path.join(LOG_DIR, 'error.log');
const DEBUG_LOG = path.join(LOG_DIR, 'debug.log');

// Ensure log directory exists
if (!fs.existsSync(LOG_DIR)) {
  console.log(`Creating log directory: ${LOG_DIR}`);
  fs.mkdirSync(LOG_DIR, { recursive: true, mode: 0o755 });
}

// Ensure log files exist
[ACCESS_LOG, ERROR_LOG, DEBUG_LOG].forEach(logFile => {
  if (!fs.existsSync(logFile)) {
    fs.writeFileSync(logFile, '', { mode: 0o644 });
  }
});

// Format log entry as JSON
function formatLogEntry(method, url, ip, userAgent, referer, statusCode, duration) {
  const timestamp = new Date().toISOString();
  const logData = {
    timestamp,
    level: 'access',
    message: `${method} ${url}`,
    sourceIp: ip, // Add source IP at the top level for visibility
    data: {
      method,
      url,
      ip,
      sourceIp: ip, // Duplicate for consistency with middleware
      userAgent,
      referer,
      statusCode,
      duration
    }
  };
  
  return JSON.stringify(logData) + '\n';
}

// Format error log entry
function formatErrorEntry(message, error, req) {
  const timestamp = new Date().toISOString();
  
  // Extract request information if available
  let requestInfo = {};
  if (req) {
    const ip = req.headers['cf-connecting-ip'] || 
              (req.headers['x-forwarded-for'] ? req.headers['x-forwarded-for'].split(',')[0].trim() : null) || 
              req.headers['x-real-ip'] || 
              (req.socket ? req.socket.remoteAddress : null) || 
              'unknown';
    
    requestInfo = {
      method: req.method,
      url: req.url,
      ip,
      sourceIp: ip,
      userAgent: req.headers['user-agent'] || 'unknown',
      referer: req.headers['referer'] || 'direct'
    };
  }
  
  // Format error information
  const errorInfo = error instanceof Error ? {
    name: error.name,
    message: error.message,
    stack: error.stack
  } : (typeof error === 'object' ? error : { message: String(error) });
  
  const logData = {
    timestamp,
    level: 'error',
    message,
    ...(requestInfo.sourceIp && { sourceIp: requestInfo.sourceIp }),
    data: {
      ...requestInfo,
      error: errorInfo
    }
  };
  
  return JSON.stringify(logData) + '\n';
}

// Log request to file
function logRequest(req, res, startTime) {
  const duration = Date.now() - startTime;
  const { method, url } = req;
  
  // Extract IP address from various headers and fallbacks
  let ip = 'unknown';
  
  // Try to get IP from headers first (in order of reliability)
  if (req.headers['cf-connecting-ip']) {
    // Cloudflare
    ip = req.headers['cf-connecting-ip'];
  } else if (req.headers['x-forwarded-for']) {
    // Standard proxy header, may contain multiple IPs
    const forwardedIps = req.headers['x-forwarded-for'].split(',');
    ip = forwardedIps[0].trim(); // Get the first IP (client)
  } else if (req.headers['x-real-ip']) {
    // Nginx
    ip = req.headers['x-real-ip'];
  } else if (req.socket && req.socket.remoteAddress) {
    // Direct connection
    ip = req.socket.remoteAddress;
  }
  
  // Clean up IPv6 localhost format if needed
  if (ip === '::1') {
    ip = '127.0.0.1';
  }
  
  // Other request data
  const userAgent = req.headers['user-agent'] || 'unknown';
  const referer = req.headers['referer'] || 'direct';
  const statusCode = res.statusCode;
  
  const logEntry = formatLogEntry(method, url, ip, userAgent, referer, statusCode, duration);
  
  // Log to console with IP address
  console.log(`[ACCESS] ${method} ${url} ${statusCode} ${duration}ms from ${ip}`);
  
  // Write to log file
  try {
    fs.appendFileSync(ACCESS_LOG, logEntry, { encoding: 'utf8' });
  } catch (error) {
    console.error('Failed to write to access log:', error);
  }
}

// Log error to file
function logError(message, error, req = null) {
  // Format error log entry
  const logEntry = formatErrorEntry(message, error, req);
  
  // Log to console
  console.error(`[ERROR] ${message}`, error instanceof Error ? error.stack : error);
  
  // Write to error log file
  try {
    fs.appendFileSync(ERROR_LOG, logEntry, { encoding: 'utf8' });
  } catch (writeError) {
    console.error('Failed to write to error log:', writeError);
  }
  
  // Also log to debug log for more context
  try {
    fs.appendFileSync(DEBUG_LOG, logEntry, { encoding: 'utf8' });
  } catch (writeError) {
    console.error('Failed to write to debug log:', writeError);
  }
}

// Log debug information
function logDebug(message, data = null, req = null) {
  // Format debug log entry similar to error entry
  const timestamp = new Date().toISOString();
  
  // Extract request information if available
  let requestInfo = {};
  if (req) {
    const ip = req.headers['cf-connecting-ip'] || 
              (req.headers['x-forwarded-for'] ? req.headers['x-forwarded-for'].split(',')[0].trim() : null) || 
              req.headers['x-real-ip'] || 
              (req.socket ? req.socket.remoteAddress : null) || 
              'unknown';
    
    requestInfo = {
      method: req.method,
      url: req.url,
      ip,
      sourceIp: ip,
      userAgent: req.headers['user-agent'] || 'unknown',
      referer: req.headers['referer'] || 'direct'
    };
  }
  
  const logData = {
    timestamp,
    level: 'debug',
    message,
    ...(requestInfo.sourceIp && { sourceIp: requestInfo.sourceIp }),
    data: {
      ...requestInfo,
      ...(data && { details: data })
    }
  };
  
  const logEntry = JSON.stringify(logData) + '\n';
  
  // Log to console
  console.log(`[DEBUG] ${message}`, data);
  
  // Write to debug log file
  try {
    fs.appendFileSync(DEBUG_LOG, logEntry, { encoding: 'utf8' });
  } catch (writeError) {
    console.error('Failed to write to debug log:', writeError);
  }
}

// Create custom server
const dev = process.env.NODE_ENV !== 'production';
const hostname = process.env.HOSTNAME || 'localhost';
const port = parseInt(process.env.PORT || '3000', 10);

// Global error handlers
process.on('uncaughtException', (error) => {
  logError('Uncaught Exception', error);
  console.error('[FATAL] Uncaught Exception:', error);
  // In production, you might want to restart the process here
  // process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logError('Unhandled Promise Rejection', reason);
  console.error('[FATAL] Unhandled Promise Rejection:', reason);
});

// Initialize Next.js
const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  createServer(async (req, res) => {
    try {
      // Record start time for request duration
      const startTime = Date.now();
      
      // Track when response finishes to log the request
      res.on('finish', () => {
        logRequest(req, res, startTime);
      });
      
      // Parse URL
      const parsedUrl = parse(req.url, true);
      
      // Let Next.js handle the request
      await handle(req, res, parsedUrl);
    } catch (err) {
      // Log the error with detailed information
      logError(`Error handling request ${req.method} ${req.url}`, err, req);
      
      // Send error response
      res.statusCode = 500;
      res.setHeader('Content-Type', 'text/html');
      res.end(`
        <html>
          <head>
            <title>Server Error</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }
              .error-container { max-width: 800px; margin: 0 auto; background: #f8f8f8; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
              h1 { color: #e74c3c; }
              .error-id { color: #7f8c8d; font-size: 0.9em; margin-top: 15px; }
              .error-time { color: #7f8c8d; font-size: 0.9em; }
            </style>
          </head>
          <body>
            <div class="error-container">
              <h1>Application Error</h1>
              <p>A server-side exception has occurred while processing your request.</p>
              <p>The error has been logged and will be investigated.</p>
              <p class="error-id">Error ID: ${Date.now().toString(36)}-${Math.random().toString(36).substr(2, 5)}</p>
              <p class="error-time">Time: ${new Date().toISOString()}</p>
            </div>
          </body>
        </html>
      `);
    }
  }).listen(port, (err) => {
    if (err) {
      logError('Failed to start server', err);
      throw err;
    }
    console.log(`> Ready on http://${hostname}:${port}`);
    console.log(`> Request logging enabled to ${ACCESS_LOG}`);
    console.log(`> Error logging enabled to ${ERROR_LOG}`);
    console.log(`> Debug logging enabled to ${DEBUG_LOG}`);
    
    // Log server startup
    logDebug('Server started', {
      env: process.env.NODE_ENV || 'development',
      port,
      hostname,
      nodeVersion: process.version,
      platform: process.platform,
      memoryUsage: process.memoryUsage()
    });
  });
});