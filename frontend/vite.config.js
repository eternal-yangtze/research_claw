import { defineConfig } from 'vite';

export default defineConfig({
  server: { proxy: { '/api': { target: process.env.RESEARCHCLAW_API_TARGET || 'http://127.0.0.1:8011', changeOrigin: true } } },
});
