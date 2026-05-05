import { defineConfig } from 'vite';

export default defineConfig(async () => {
  const { default: glsl } = await import('vite-plugin-glsl');
  return {
    plugins: [glsl()],
  };
});
