#!/bin/bash

# Stop any running containers
docker-compose down

# Start development container
docker-compose up web-dev