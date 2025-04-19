# Development stage
FROM node:18-alpine AS development

# Set working directory
WORKDIR /app

# Install dependencies first
COPY package.json ./

# Install dependencies and ensure next is in PATH
ENV PATH /app/node_modules/.bin:$PATH

# Install dependencies
RUN npm install

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 3000

# Start development server with hot reload
CMD ["npm", "run", "dev"]

# Production stage
FROM node:18-alpine AS production

# Set working directory
WORKDIR /app

# Add dependencies for building
RUN apk add --no-cache libc6-compat

# Install dependencies first
COPY package.json ./

# Install dependencies and ensure next is in PATH
ENV PATH /app/node_modules/.bin:$PATH

# Install dependencies
RUN npm install

# Copy the rest of the code
COPY . .

# Build the application
RUN npm run build

# Install only production dependencies
RUN npm prune --production

# Expose port
EXPOSE 3000

# Start production server
CMD ["npm", "start"]