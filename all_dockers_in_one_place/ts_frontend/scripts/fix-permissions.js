const fs = require('fs');
const path = require('path');

const LOG_DIR = path.join(process.cwd(), 'logs');

try {
  // Create logs directory if it doesn't exist
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true, mode: 0o755 });
    console.log('Created logs directory with correct permissions');
  }

  // Fix directory permissions
  fs.chmodSync(LOG_DIR, 0o755);
  console.log('Fixed logs directory permissions');

  // Fix file permissions
  const files = fs.readdirSync(LOG_DIR);
  files.forEach(file => {
    const filePath = path.join(LOG_DIR, file);
    fs.chmodSync(filePath, 0o644);
    console.log(`Fixed permissions for ${file}`);
  });

  console.log('All permissions fixed successfully');
} catch (error) {
  console.error('Error fixing permissions:', error);
  process.exit(1);
}