import { defineConfig, devices } from '@playwright/test'

// Browser tests for the one thing no unit test can see: layout.
//
// The suite runs against the *built* bundle served by `vite preview`, with the API stubbed
// at the network layer (see e2e/fixtures.ts) - no FastAPI, no Qdrant, no model, no API key.
// Chromium only: the assertions are about CSS box sizing, which is not where engines
// disagree, and a three-browser matrix would triple the CI cost of one guard.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0, // these are deterministic; a retry would only hide a flake worth seeing
  reporter: process.env.CI ? 'line' : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      // A fixed viewport, because every assertion is a measurement. The default 1280x720
      // is also small enough that the stage's height binds the fit, which is the case the
      // regression lived in.
      use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 720 } },
    },
  ],
  webServer: {
    // Builds first so a local run cannot measure a stale dist/. In CI this repeats the
    // build step above it, which costs under a second.
    //
    // `--host 127.0.0.1` is not decoration: vite preview otherwise binds "localhost",
    // which resolves to ::1 only on some machines, and the IPv4 probe below then waits out
    // the full timeout against a server that is up and listening.
    command: 'npm run build && npm run preview -- --port 4173 --strictPort --host 127.0.0.1',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
