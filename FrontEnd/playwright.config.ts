import { defineConfig } from "@playwright/test";


const frontendPort = Number(process.env.PLAYWRIGHT_FRONTEND_PORT ?? 4173);
const backendPort = Number(process.env.PLAYWRIGHT_BACKEND_PORT ?? 8787);


export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: "node tests/fixtures/mock-stream-server.mjs",
      port: backendPort,
      reuseExistingServer: !process.env.CI,
      cwd: ".",
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4173",
      port: frontendPort,
      reuseExistingServer: !process.env.CI,
      cwd: ".",
      env: {
        VITE_API_BASE_URL: `http://127.0.0.1:${backendPort}`,
      },
    },
  ],
});
