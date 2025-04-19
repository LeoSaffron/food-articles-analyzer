#!/bin/bash

# Stop any running containers
docker-compose down

# Start production container
docker-compose up web-prod