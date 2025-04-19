# Recipe Analyzer Frontend

A Next.js application for analyzing recipes for plant-based compatibility.

## Prerequisites

- Node.js (v16 or higher)
- npm (v7 or higher)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd recipe-analyzer-frontend
```

2. Install dependencies:
```bash
npm install
```

## Configuration

The application can be configured using environment variables. Create a `.env.local` file in the project root or set environment variables directly.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEXT_PUBLIC_API_HOST | API server hostname | localhost |
| NEXT_PUBLIC_API_PORT | API server port | 8002 |
| ENABLE_DEBUG | Enable debug output | false |

### Configuration Methods

1. Using `.env.local` file:
```env
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud
NEXT_PUBLIC_API_PORT=44251
ENABLE_DEBUG=true
```

2. Using command line:
```bash
# Windows (PowerShell)
$env:NEXT_PUBLIC_API_HOST="recipesanalysis.cloud"; $env:NEXT_PUBLIC_API_PORT="44251"; npm run dev

# Windows (CMD)
set NEXT_PUBLIC_API_HOST=recipesanalysis.cloud && set NEXT_PUBLIC_API_PORT=44251 && npm run dev

# Linux/Mac
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud NEXT_PUBLIC_API_PORT=44251 npm run dev
```

## Running the Application

### Development Mode

1. Basic development server:
```bash
npm run dev
```

2. With custom API endpoint:
```bash
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud NEXT_PUBLIC_API_PORT=44251 npm run dev
```

3. With debug output enabled:
```bash
ENABLE_DEBUG=true npm run dev
```

4. With all options:
```bash
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud NEXT_PUBLIC_API_PORT=44251 ENABLE_DEBUG=true npm run dev
```

### Production Mode

1. Build the application:
```bash
npm run build
```

2. Start the production server:
```bash
npm start
```

3. With custom configuration:
```bash
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud NEXT_PUBLIC_API_PORT=44251 npm start
```

## Development

### Debug Mode

Debug mode provides additional information in the UI and console logs. To enable:

1. Using environment variable:
```bash
ENABLE_DEBUG=true npm run dev
```

2. For production:
```bash
ENABLE_DEBUG=true npm run build
ENABLE_DEBUG=true npm start
```

### API Configuration Examples

1. Local development API:
```bash
NEXT_PUBLIC_API_HOST=localhost NEXT_PUBLIC_API_PORT=8002 npm run dev
```

2. Production API:
```bash
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud NEXT_PUBLIC_API_PORT=44251 npm run dev
```

### Common Configurations

1. Local development setup:
```bash
NEXT_PUBLIC_API_HOST=localhost NEXT_PUBLIC_API_PORT=8002 ENABLE_DEBUG=true npm run dev
```

2. Production setup:
```bash
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud NEXT_PUBLIC_API_PORT=44251 npm run build
NEXT_PUBLIC_API_HOST=recipesanalysis.cloud NEXT_PUBLIC_API_PORT=44251 npm start
```

## Troubleshooting

### Common Issues

1. API Connection Failed
```
Error: Failed to fetch from API
```
Solution: Verify API host and port configuration:
```bash
# Test API connection
curl http://$NEXT_PUBLIC_API_HOST:$NEXT_PUBLIC_API_PORT/health
```

2. Debug Output Not Showing
```
Debug information not visible
```
Solution: Ensure debug mode is enabled:
```bash
ENABLE_DEBUG=true npm run dev
```

### Checking Configuration

To verify current configuration:

1. With debug mode:
```bash
ENABLE_DEBUG=true npm run dev
```
Debug information will be visible in the UI.

2. Check environment variables:
```bash
# Linux/Mac
echo $NEXT_PUBLIC_API_HOST
echo $NEXT_PUBLIC_API_PORT

# Windows (PowerShell)
echo $env:NEXT_PUBLIC_API_HOST
echo $env:NEXT_PUBLIC_API_PORT
```

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run linter

## Additional Notes

- The application requires a running instance of the recipe analysis API
- Debug mode is controlled server-side via ENABLE_DEBUG environment variable
- API configuration can be changed without rebuilding the application
- All configuration can be set via environment variables or .env files
