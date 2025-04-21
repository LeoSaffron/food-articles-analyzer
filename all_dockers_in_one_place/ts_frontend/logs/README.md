# Logging System

This directory contains log files for the application.

## Log Files

- `access.log`: Records all HTTP requests to the application
- `error.log`: Records application errors
- `debug.log`: Records debug information (only when ENABLE_DEBUG=true)

## Viewing Logs

You can use the following npm scripts to view logs:

```bash
# Tail logs in real-time
npm run logs:tail         # Default: access.log
npm run logs:tail:access  # Access logs
npm run logs:tail:error   # Error logs
npm run logs:tail:debug   # Debug logs

# Analyze logs
npm run logs              # General log analysis
npm run logs:errors       # Focus on errors
npm run logs:slow         # Focus on slow requests (>500ms)
```

## Log Format

Logs are stored in JSON format with the following structure:

```json
{
  "timestamp": "2023-06-01T12:34:56.789Z",
  "level": "access|error|debug",
  "message": "Request or error message",
  "requestId": "unique-request-id",
  "data": {
    // Additional context data
  }
}
```

## Log Rotation

Logs are automatically rotated when they reach 5MB in size. Up to 5 rotated log files are kept for each log type.