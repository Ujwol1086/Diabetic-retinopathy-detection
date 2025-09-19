// Environment configuration for Vite
export const config = {
  API_BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:5000',
  NODE_ENV: import.meta.env.MODE || 'development',
  DEV: import.meta.env.DEV,
  PROD: import.meta.env.PROD,
} as const
