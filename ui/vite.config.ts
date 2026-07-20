import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The UI runs on its own dev server and talks to the FastAPI backend (default
// http://127.0.0.1:8000) over CORS. Override the target with VITE_API_BASE.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
