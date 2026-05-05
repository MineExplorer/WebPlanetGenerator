import { defineConfig, Plugin } from 'vite';

function removeCrossorigin(): Plugin {
  return {
    name: 'remove-crossorigin',
    apply: 'build',
    transformIndexHtml(html) {
      return html
        .replace(/<script([^>]*?) crossorigin([^>]*)>/g, '<script$1$2>')
        .replace(/<link([^>]*?) crossorigin([^>]*?)>/g, '<link$1$2>')
        .replace(/<script([^>]*?) type="module"([^>]*)>/g, '<script$1 defer$2>');
    },
  };
}

export default defineConfig(async () => {
  const { default: glsl } = await import('vite-plugin-glsl');
  return {
    plugins: [glsl(), removeCrossorigin()],
    base: './',
    build: {
      rollupOptions: {
        output: {
          format: 'iife',
        },
      },
    },
  };
});
