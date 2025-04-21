# Application Logging System

## Overview

This application uses a comprehensive logging system that provides structured logs for both client and server-side operations. The logging system includes features such as:

- Structured JSON logging
- Log rotation to prevent log files from growing too large
- Request ID tracking for tracing requests across the system
- Performance metrics logging
- Different log levels (access, info, debug, error, warn)
- Source IP address tracking for all requests
- Log analysis tools
- Edge runtime compatibility

## Log Files

Logs are stored in the `logs/` directory and include:

- `access.log` - HTTP requests and general access information
- `error.log` - Application errors and exceptions
- `debug.log` - Detailed debug information (only when enabled)

## Troubleshooting Logging

If logs aren't being written to files, you can run the test script to diagnose issues:

```bash
npm run test:logs
```

This will check if the logs directory exists, if log files are writable, and attempt to write test entries to the log files.

## Environment Variables

To enable debug logging, run the application with:

```bash
ENABLE_DEBUG=true npm run dev
```

## Edge Runtime Compatibility

The logging system is designed to work in both Node.js and Edge runtime environments:

- In Node.js environments, logs are written to files in the `logs/` directory
- In Edge runtime environments, logs are stored in memory and also output to the console
- The API route for client-side logging is configured to use Node.js runtime

To access in-memory logs from Edge runtime:

```typescript
import { getMemoryLogs } from '@/lib/server-logger';

// Get all access logs stored in memory
const accessLogs = getMemoryLogs('access');

// Get all error logs stored in memory
const errorLogs = getMemoryLogs('error');
```

## Log Analysis

The application includes a log analysis tool that can be used to search and filter logs:

```bash
# View all logs
npm run logs

# View only errors
npm run logs:errors

# View slow requests (>500ms)
npm run logs:slow

# Custom filtering
node scripts/logs/analyze.js --file=logs/access.log --since=1h --format=json
```

Available options:

- `--file=<filename>` - Specify log file to analyze (default: logs/access.log)
- `--level=<level>` - Filter by log level (access, error, debug, info, warn)
- `--requestId=<id>` - Filter by request ID
- `--since=<time>` - Show logs since time (e.g., 1h, 30m, 1d)
- `--errors` - Show only errors
- `--slow=<ms>` - Show requests slower than specified ms
- `--format=<format>` - Output format (json, table, pretty)

## Configuration

Logging behavior can be configured through environment variables:

- `ENABLE_DEBUG=true` - Enable debug logging
- `NODE_ENV=development` - Show logs in console as well as writing to files