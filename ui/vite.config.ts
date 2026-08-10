import { configDefaults, defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// The UI runs on its own dev server and talks to the FastAPI backend (default
// http://127.0.0.1:8000) over CORS. Override the target with VITE_API_BASE.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  test: {
    // Playwright owns e2e/. The two runners share the `*.spec.ts` convention but not a
    // runtime, so without this vitest picks up the browser specs and dies on the
    // @playwright/test import.
    exclude: [...configDefaults.exclude, 'e2e/**'],
  },
})
