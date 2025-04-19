interface ApiConfig {
  host: string;
  port: string;
}

export const CONFIG = {
  debug: process.env.ENABLE_DEBUG === 'true',
  api: {
    host: process.env.NEXT_PUBLIC_API_HOST || 'localhost',
    port: process.env.NEXT_PUBLIC_API_PORT || '8002'
  }
} as const;

export function getApiConfig(): ApiConfig {
  return {
    host: process.env.NEXT_PUBLIC_API_HOST || 'localhost',
    port: process.env.NEXT_PUBLIC_API_PORT || '8002'
  };
}