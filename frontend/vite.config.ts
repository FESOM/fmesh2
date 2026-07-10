/// <reference types="vitest" />
import { defineConfig } from 'vite';

export default defineConfig({
  test: {
    // Playwright specs are run by `npm run e2e`, not vitest
    exclude: ['e2e/**', 'node_modules/**'],
  },
  build: {
    outDir: '../src/fmesh2/server/static',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8017',
    },
  },
});
