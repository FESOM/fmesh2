import { defineConfig } from 'vite';

export default defineConfig({
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
