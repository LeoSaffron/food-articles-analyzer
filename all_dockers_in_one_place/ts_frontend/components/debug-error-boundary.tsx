import React, { Component, ErrorInfo, ReactNode } from 'react';
import { captureDebugContext } from '@/lib/debug-utils';

interface Props {
  children: ReactNode;
  componentName?: string;
  fallback?: ReactNode | ((error: Error, errorInfo: ErrorInfo) => ReactNode);
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Removes cookies and sensitive data from props
 */
function cleanProps(props: any): any {
  if (!props || typeof props !== 'object') {
    return props;
  }
  
  // Create a clean copy without children (React nodes)
  const cleanedProps = { ...props };
  delete cleanedProps.children;
  
  // Remove any cookie-related properties
  for (const key in cleanedProps) {
    const lowerKey = key.toLowerCase();
    if (['cookie', 'cookies', 'authorization', 'token', 'password', 'secret'].some(k => lowerKey.includes(k))) {
      delete cleanedProps[key];
    } else if (typeof cleanedProps[key] === 'object' && cleanedProps[key] !== null) {
      cleanedProps[key] = cleanProps(cleanedProps[key]);
    }
  }
  
  return cleanedProps;
}

/**
 * An error boundary component that captures and logs detailed information
 * when errors occur in its children components
 */
export class DebugErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Update state with error details
    this.setState({
      errorInfo
    });

    // Capture debug context with error information
    captureDebugContext(
      this.props.componentName || 'UnknownComponent',
      cleanProps(this.props),
      {
        error: {
          message: error.message,
          name: error.name,
          stack: error.stack
        },
        errorInfo: {
          componentStack: errorInfo.componentStack
        },
        location: window.location.href,
        timestamp: new Date().toISOString()
      }
    );

    // Log to console for local debugging
    console.error('Error caught by DebugErrorBoundary:', error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      // Render fallback UI if provided
      if (this.props.fallback) {
        if (typeof this.props.fallback === 'function' && this.state.error) {
          return this.props.fallback(this.state.error, this.state.errorInfo || { componentStack: '' });
        }
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <div className="error-boundary-fallback">
          <h2>Something went wrong.</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            <summary>Error Details</summary>
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo && this.state.errorInfo.componentStack}
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Higher-order component to wrap a component with a DebugErrorBoundary
 * @param Component The component to wrap
 * @param componentName The name of the component (for logging)
 * @param fallback Optional fallback UI
 */
export function withErrorBoundary<P>(
  Component: React.ComponentType<P>,
  componentName: string,
  fallback?: ReactNode | ((error: Error, errorInfo: ErrorInfo) => ReactNode)
) {
  return function WithErrorBoundary(props: P) {
    return (
      <DebugErrorBoundary componentName={componentName} fallback={fallback}>
        <Component {...props} />
      </DebugErrorBoundary>
    );
  };
}