#!/usr/bin/env node

/**
 * Direct logging test script
 * This bypasses the TypeScript and module system to test logging directly
 */

const directLogger = require('../lib/direct-logger');

console.log('Starting direct logging test...');

// Test access log
directLogger.writeLog('access', 'Direct test access log', { test: true, timestamp: new Date().toISOString() });

// Test error log
directLogger.writeLog('error', 'Direct test error log', { test: true, timestamp: new Date().toISOString() });

// Test debug log
directLogger.writeLog('debug', 'Direct test debug log', { test: true, timestamp: new Date().toISOString() });

console.log('Direct logging test complete.');
console.log(`Check the following files for log entries:`);
console.log(`- ${directLogger.ACCESS_LOG}`);
console.log(`- ${directLogger.ERROR_LOG}`);
console.log(`- ${directLogger.DEBUG_LOG}`);