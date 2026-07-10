import { defineConfig } from '@playwright/test';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Smoke e2e: serves the built frontend through the real fmesh2 server with
 * the core2 reference mesh. Requires `npm run build` first (the server ships
 * the static bundle) and a chromium via `npx playwright install chromium`.
 * Override the mesh with FMESH2_E2E_MESH, the port with FMESH2_E2E_PORT. */

const PORT = Number(process.env.FMESH2_E2E_PORT ?? 8321);
const MESH = process.env.FMESH2_E2E_MESH ?? path.resolve(__dirname, '../../core2');

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  use: { baseURL: `http://127.0.0.1:${PORT}` },
  webServer: {
    command: `micromamba run -n meshgen fmesh2 ${MESH} --port ${PORT} --no-browser`,
    url: `http://127.0.0.1:${PORT}/api/meshes`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
