# Debugging Utilities

This directory contains utilities for debugging client-side errors, particularly focused on capturing context before errors occur.

## Important Note on Privacy

All debugging utilities in this package are designed to completely remove cookies and sensitive data from logs. This includes:

- Cookies are never logged in any circumstance
- Authorization tokens are removed
- Password data is removed
- Any key containing 'secret' is removed

## Available Utilities

### Client Debug

- `captureDebugContext`: Sends debugging context to the server before an error might occur
- `inspectPotentialErrorObject`: Inspects a specific object that might be causing errors
- `createSafeObjectEntries`: Creates a wrapped version of Object.entries that logs debugging information
- `withDebugLogging`: HOC to add debug logging before rendering a component

### Object Entries Debug

- `safeObjectEntries`: A safe version of Object.entries that won't throw on null/undefined
- `monkeyPatchObjectEntries`: Monkey patches the global Object.entries to use the safe version
- `withObjectEntriesDebug`: Wraps a component with debugging for Object.entries calls

## How to Use

### Debugging the "Cannot convert undefined or null to object" Error

If you're seeing the error "Cannot convert undefined or null to object" at Object.entries, you can use these utilities to help debug the issue:

1. **Option 1: Use the safe version directly**

   ```tsx
   import { safeObjectEntries } from '@/lib/debug-utils';
   
   // Instead of:
   // const entries = Object.entries(someObject);
   
   // Use:
   const entries = safeObjectEntries(someObject, 'ResultsDisplay:processData');
   ```

2. **Option 2: Monkey patch Object.entries globally**

   ```tsx
   import { monkeyPatchObjectEntries } from '@/lib/debug-utils';
   
   // In your app initialization code:
   const restoreOriginal = monkeyPatchObjectEntries();
   
   // Later, if you want to restore the original:
   // restoreOriginal();
   ```

3. **Option 3: Wrap the problematic component**

   ```tsx
   import { withObjectEntriesDebug } from '@/lib/debug-utils';
   
   const ResultsDisplayWithDebug = withObjectEntriesDebug(ResultsDisplay, 'ResultsDisplay');
   
   // Use ResultsDisplayWithDebug instead of ResultsDisplay
   ```

### Capturing Context Before Errors

```tsx
import { captureDebugContext } from '@/lib/debug-utils';

function MyComponent(props) {
  // Capture debug context at the beginning of the component
  captureDebugContext('MyComponent', props, {
    additionalInfo: 'Some extra context',
    currentState: someState
  });
  
  // Rest of your component...
}
```

### Using the Debug Error Boundary

```tsx
import { DebugErrorBoundary } from '@/components/debug-error-boundary';

function App() {
  return (
    <DebugErrorBoundary componentName="ResultsDisplay">
      <ResultsDisplay data={someData} />
    </DebugErrorBoundary>
  );
}
```

Or use the HOC version:

```tsx
import { withErrorBoundary } from '@/components/debug-error-boundary';

const ResultsDisplayWithErrorBoundary = withErrorBoundary(
  ResultsDisplay,
  'ResultsDisplay',
  (error) => <div>Something went wrong: {error.message}</div>
);
```

## API Endpoints

These utilities communicate with the following API endpoints:

- `POST /api/debug-capture`: Captures component context before errors
- `PUT /api/debug-capture`: Inspects specific objects that might cause errors
- `PATCH /api/test-logging`: General purpose debug context capture

The captured data will be available in your server logs with detailed information to help debug the issues, with all cookies and sensitive data completely removed.