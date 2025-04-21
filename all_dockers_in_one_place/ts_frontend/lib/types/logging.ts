export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  data?: any;
}

export interface LogConfig {
  dir: string;
  files: {
    access: string;
    debug: string;
    error: string;
  };
  format: {
    timestamp: boolean;
    color: boolean;
  };
}