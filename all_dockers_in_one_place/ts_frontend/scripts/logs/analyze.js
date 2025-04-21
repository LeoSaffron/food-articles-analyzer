#!/usr/bin/env node

/**
 * Log Analysis Script
 * 
 * Usage:
 *   node scripts/logs/analyze.js [options]
 * 
 * Options:
 *   --file=<filename>    Specify log file to analyze (default: logs/access.log)
 *   --level=<level>      Filter by log level (access, error, debug, info, warn)
 *   --requestId=<id>     Filter by request ID
 *   --since=<time>       Show logs since time (e.g., 1h, 30m, 1d)
 *   --errors             Show only errors
 *   --slow=<ms>          Show requests slower than specified ms
 *   --format=<format>    Output format (json, table, pretty)
 *   --help               Show this help
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Parse command line arguments
const args = process.argv.slice(2);
let options = {
  file: 'logs/access.log',
  level: null,
  requestId: null,
  since: null,
  errors: false,
  slow: null,
  format: 'pretty'
};

// Show help
if (args.includes('--help')) {
  console.log(`
Log Analysis Script

Usage:
  node scripts/logs/analyze.js [options]

Options:
  --file=<filename>    Specify log file to analyze (default: logs/access.log)
  --level=<level>      Filter by log level (access, error, debug, info, warn)
  --requestId=<id>     Filter by request ID
  --since=<time>       Show logs since time (e.g., 1h, 30m, 1d)
  --errors             Show only errors
  --slow=<ms>          Show requests slower than specified ms
  --format=<format>    Output format (json, table, pretty)
  --help               Show this help
`);
  process.exit(0);
}

// Parse arguments
args.forEach(arg => {
  if (arg === '--errors') {
    options.errors = true;
    return;
  }
  
  const match = arg.match(/--([^=]+)=(.+)/);
  if (match) {
    const [, key, value] = match;
    options[key] = value;
  }
});

// If --errors is specified, set file to error.log and level to error
if (options.errors) {
  options.file = 'logs/error.log';
  options.level = 'error';
}

// Calculate since time
let sinceTime = null;
if (options.since) {
  const now = new Date();
  const match = options.since.match(/^(\d+)([hmd])$/);
  if (match) {
    const [, amount, unit] = match;
    switch (unit) {
      case 'h':
        sinceTime = new Date(now - amount * 60 * 60 * 1000);
        break;
      case 'm':
        sinceTime = new Date(now - amount * 60 * 1000);
        break;
      case 'd':
        sinceTime = new Date(now - amount * 24 * 60 * 60 * 1000);
        break;
    }
  }
}

// Resolve file path
const logFile = path.resolve(process.cwd(), options.file);
if (!fs.existsSync(logFile)) {
  console.error(`Error: Log file not found: ${logFile}`);
  process.exit(1);
}

// Process the log file
async function processLogFile() {
  const fileStream = fs.createReadStream(logFile);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  const logs = [];

  for await (const line of rl) {
    if (!line.trim()) continue;
    
    try {
      const logEntry = JSON.parse(line);
      
      // Apply filters
      if (options.level && logEntry.level !== options.level) continue;
      if (options.requestId && logEntry.requestId !== options.requestId) continue;
      if (sinceTime && new Date(logEntry.timestamp) < sinceTime) continue;
      if (options.slow && (!logEntry.data || !logEntry.data.duration || logEntry.data.duration < parseInt(options.slow))) continue;
      
      logs.push(logEntry);
    } catch (error) {
      console.error(`Error parsing log line: ${line}`);
    }
  }

  return logs;
}

// Format and display logs
async function displayLogs() {
  const logs = await processLogFile();
  
  if (logs.length === 0) {
    console.log('No logs found matching the criteria.');
    return;
  }
  
  switch (options.format) {
    case 'json':
      console.log(JSON.stringify(logs, null, 2));
      break;
    
    case 'table':
      console.table(logs.map(log => ({
        timestamp: log.timestamp,
        level: log.level,
        message: log.message,
        requestId: log.requestId || 'N/A',
        duration: log.data?.duration || 'N/A'
      })));
      break;
    
    case 'pretty':
    default:
      logs.forEach(log => {
        const timestamp = new Date(log.timestamp).toLocaleString();
        const level = log.level.toUpperCase().padEnd(5);
        const requestId = log.requestId ? `[${log.requestId.substring(0, 8)}]` : '';
        const duration = log.data?.duration ? `(${log.data.duration}ms)` : '';
        
        console.log(`${timestamp} ${level} ${requestId} ${duration} ${log.message}`);
        
        if (log.data) {
          const { duration, _context, ...restData } = log.data;
          if (Object.keys(restData).length > 0) {
            console.log('  Data:', JSON.stringify(restData));
          }
          if (_context) {
            console.log('  Context:', JSON.stringify(_context));
          }
        }
        console.log();
      });
      break;
  }
  
  console.log(`Found ${logs.length} log entries.`);
}

displayLogs().catch(error => {
  console.error('Error processing logs:', error);
  process.exit(1);
});