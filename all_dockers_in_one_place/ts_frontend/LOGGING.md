# Comprehensive Request Logging

## Overview

This project has been configured to log every HTTP request (GET, POST, etc.) in all server configurations, including development and production environments.

## How It Works

1. **Custom Server**: A custom server implementation (`server/index.js`) intercepts all HTTP requests before they reach Next.js and logs them to `logs/access.log`.

2. **Middleware**: The existing middleware (`middleware.ts`) provides additional request logging with more detailed information.

3. **Log Files**: All logs are stored in the `logs/` directory:
   - `access.log`: Records all HTTP requests
   - `error.log`: Records application errors
   - `debug.log`: Records debug information (when enabled)

## Using the Logging System

### Starting the Server with Logging

```bash
# Development with full request logging
npm run dev

# Production with full request logging
npm run start
```

### Viewing Logs in Real-time

```bash
# Watch access logs in real-time
npm run logs:tail
# or
npm run logs:tail:access

# Watch error logs in real-time
npm run logs:tail:error

# Watch debug logs in real-time
npm run logs:tail:debug
```

### Analyzing Logs

```bash
# General log analysis
npm run logs

# Focus on errors
npm run logs:errors

# Focus on slow requests (>500ms)
npm run logs:slow
```

### Testing the Logging System

```bash
# Test if logging is working properly
npm run test:logs

# Test direct file system logging
npm run test:direct-logs
```

## Log Format

Logs are stored in JSON format with the following structure:

```json
{
  "timestamp": "2023-06-01T12:34:56.789Z",
  "level": "access|error|debug",
  "message": "GET /api/example",  "sourceIp": "192.168.1.1",  "sourceIp": "192.168.1.1",
  "requestId": "unique-request-id",
  "data": {
    "method": "GET",
    "ip": "192.168.1.1",
    "sourceIp": "192.168.1.1",le",
    "ip": "192.168.1.1",
    "sourceIp": "192.168.1.1",
    "userAgent": "Mozilla/5.0...",
    "referer": "https://example.com",
    "statusCode": 200,
    "duration": 42
  }
The source IP address is included at both the top level (`sourceIp`) and within the `data` object for compatibility and ease of access.}
```

The source IP address is included at both the top level (`sourceIp`) and within the `data` object for compatibility and ease of access.

## Troubleshooting

### Logs Not Being Written

1. Check permissions on the `logs` directory:
   ```bash
   ls -la logs/
   ```

2. Ensure the directory and files are writable:
   ```bash
   chmod -R 755 logs/
   chmod 644 logs/*.log
   ```

3. Run the logging test script:
   ```bash
   npm run test:logs
   ```

### Missing Requests in Logs

If some requests are not being logged:

1. Ensure you're using the custom server:
   ```bash
   npm run dev  # Not next dev
   ```

2. Check if the request is being handled by a static file or is being cached.

3. Verify that the request is reaching your server by adding console logs.

## Advanced Configuration

### Enabling Debug Logs

To enable debug logs, set the `ENABLE_DEBUG` environment variable:

```bash
ENABLE_DEBUG=true npm run dev
```
## IP Address Detection

The system uses a comprehensive approach to detect the client's source IP address, checking multiple headers in the following order:

1. `cf-connecting-ip` - Used by Cloudflare
2. `x-forwarded-for` - Standard proxy header (first IP in the list is used)
3. `x-real-ip` - Used by Nginx and other proxies
4. Direct socket connection IP

This ensures accurate IP logging even when your application is behind proxies, load balancers, or CDNs.
### Log Rotation

Logs are automatically rotated when they reach 5MB in size. Up to 5 rotated log files are kept for each log type.