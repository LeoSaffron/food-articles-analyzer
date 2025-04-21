const fs = require('fs');
const path = require('path');

function checkEnvironment() {
  console.log('Checking environment...');
  
  // Check Node.js version
  console.log(`Node version: ${process.version}`);
  
  // Check if required directories exist
  const dirs = ['.next', 'node_modules'];
  dirs.forEach(dir => {
    if (fs.existsSync(dir)) {
      console.log(`✓ ${dir} directory exists`);
    } else {
      console.log(`✗ ${dir} directory missing`);
    }
  });
  
  // Check environment variables
  const requiredEnvVars = ['PORT', 'NODE_ENV', 'NEXT_DIST_DIR'];
  requiredEnvVars.forEach(envVar => {
    if (process.env[envVar]) {
      console.log(`✓ ${envVar} is set to: ${process.env[envVar]}`);
    } else {
      console.log(`✗ ${envVar} is not set`);
    }
  });
}

checkEnvironment();