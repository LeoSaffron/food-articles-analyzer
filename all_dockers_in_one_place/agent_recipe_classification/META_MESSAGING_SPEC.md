# 🎯 META Messaging System Specification

## Overview
The META messaging system provides UI/UX control instructions to receiving servers without displaying them to end users. This enables dynamic loading states and better user experience during long-running operations.

## Message Format
```
[META] KEY=VALUE|KEY=VALUE|KEY=VALUE
```

## Standard Keys

### LOADING_TYPE
Indicates the expected duration of the operation:
- `SHORT_WAIT` - Operations taking 30-60 seconds (ingredient analysis)
- `LONG_WAIT` - Operations taking 5-10+ minutes (LLM scraping)
- `COMPLETE` - Indicates an operation has finished

### UI_ACTION  
Specifies the recommended UI element to display:
- `SHOW_SPINNER` - For long operations requiring patience
- `SHOW_PROGRESS_DOTS` - For shorter operations with progress indication
- `HIDE_SPINNER` - Remove spinner when long operation completes
- `HIDE_PROGRESS_DOTS` - Remove progress indicator when short operation completes

### DEV_MESSAGE
Human-readable explanation for developers/debugging:
- Describes what operation is happening
- Includes expected timeframes
- Provides context for troubleshooting

## Implementation Locations

### 1. LLM Recipe Scraping (LONG_WAIT + COMPLETE)
**Trigger:** When dedicated library fails and LLM agent is needed
```
[INFO] Could not scrape with a dedicated library, proceeding with the agent
[META] LOADING_TYPE=LONG_WAIT|UI_ACTION=SHOW_SPINNER|DEV_MESSAGE=Starting LLM-based recipe scraping - this process can take 5-10 minutes as it requires AI analysis of the webpage content
... (LLM scraping happens) ...
[META] LOADING_TYPE=COMPLETE|UI_ACTION=HIDE_SPINNER|DEV_MESSAGE=LLM-based recipe scraping completed successfully
```

### 2. Ingredient Analysis (SHORT_WAIT + COMPLETE)
**Trigger:** After recipe extraction, before and after analyzing each ingredient
```
[INFO] Successfully Extracted ingredient list. Analyzing each ingredient...
[META] LOADING_TYPE=SHORT_WAIT|UI_ACTION=SHOW_PROGRESS_DOTS|DEV_MESSAGE=Running LLM analysis on each ingredient for plant-based classification - typically takes 30-60 seconds depending on ingredient count
... (analysis happens) ...
[META] LOADING_TYPE=COMPLETE|UI_ACTION=HIDE_PROGRESS_DOTS|DEV_MESSAGE=Ingredient analysis completed successfully
[INFO] Finished analysis.
```

## Client Implementation Guide

### Message Filtering
```javascript
// Filter messages for display
messages.forEach(message => {
  if (message.startsWith('[META]')) {
    handleMetaMessage(message);
  } else if (message.startsWith('[INFO]')) {
    displayToUser(message);
  }
});
```

### META Message Parsing
```javascript
function handleMetaMessage(message) {
  const content = message.replace('[META] ', '');
  const params = {};
  
  content.split('|').forEach(part => {
    const [key, value] = part.split('=', 2);
    params[key] = value;
  });
  
  // Control UI based on parameters
  switch(params.LOADING_TYPE) {
    case 'LONG_WAIT':
      showSpinner(params.DEV_MESSAGE);
      break;
    case 'SHORT_WAIT':  
      showProgressDots(params.DEV_MESSAGE);
      break;
  }
  
  // Log for debugging
  console.log('META:', params.DEV_MESSAGE);
}
```

### UI State Management
```javascript
function showSpinner(message) {
  // Show spinning loader
  // Display "This may take 5-10 minutes" message
  // Log technical details
}

function showProgressDots(message) {
  // Show animated dots or progress bar
  // Display "Analyzing ingredients..." message  
  // Show estimated completion time
}
```

## Testing Examples

### Test LONG_WAIT Message
```bash
curl -N "http://127.0.0.1:8002/check_recipe_stream?url=https://unsupported-site.com/recipe"
```
Expected: `[META] LOADING_TYPE=LONG_WAIT|UI_ACTION=SHOW_SPINNER|...`

### Test SHORT_WAIT Message  
```bash
curl -N "http://127.0.0.1:8002/check_recipe_stream?url=https://tasty.co/recipe/any-recipe"
```
Expected: `[META] LOADING_TYPE=SHORT_WAIT|UI_ACTION=SHOW_PROGRESS_DOTS|...`

## Future Extensions

### Additional LOADING_TYPES
- `INSTANT` - Immediate operations (< 5 seconds)
- `MEDIUM_WAIT` - Operations taking 1-3 minutes

### Additional UI_ACTIONS
- `SHOW_PROGRESS_BAR` - For operations with measurable progress
- `SHOW_ESTIMATED_TIME` - When precise timing is available
- `SHOW_STEP_INDICATOR` - For multi-step processes

### Enhanced Parameters
- `PROGRESS_CURRENT` - Current step number
- `PROGRESS_TOTAL` - Total steps
- `TIME_ESTIMATE` - Estimated completion time
- `OPERATION_ID` - Unique identifier for tracking

## Error Handling

### Malformed META Messages
- Client should gracefully ignore invalid META messages
- Log parsing errors for debugging
- Continue normal operation

### Missing META Messages
- Assume default SHORT_WAIT behavior
- Show generic loading indicator
- Don't block user interface

## Benefits

### User Experience
- ✅ Appropriate loading indicators for different operations
- ✅ Clear expectations about wait times
- ✅ No confusion about system responsiveness

### Developer Experience  
- ✅ Easy to parse structured format
- ✅ Clear separation of user/system messages
- ✅ Extensible for future requirements
- ✅ Debugging information included

### System Scalability
- ✅ Consistent message format across endpoints
- ✅ Easy to add new operation types
- ✅ No breaking changes to existing parsing
- ✅ Future-proof architecture