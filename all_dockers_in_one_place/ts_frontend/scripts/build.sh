#!/bin/bash

# Exit on error
set -e

echo "Cleaning previous build..."
rm -rf .next
rm -rf node_modules

echo "Installing dependencies..."
npm install

echo "Creating logs directory..."
mkdir -p logs
chmod 755 logs

echo "Building application..."
NEXT_DIST_DIR=.next npm run build

if [ $? -eq 0 ]; then
    echo "Build successful"
else
    echo "Build failed"
    exit 1
fi