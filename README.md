# Web Planet Generator

A minimal template project that renders a procedural 3D planet using Three.js, Simplex noise and a small GUI for parameters. Built with Vite for fast local development.

Quick start

1. Install dependencies:

```bash
npm install
```

2. Start dev server:

```bash
npm run dev
```

3. Open the URL printed by Vite (usually http://localhost:5173)

Files

- `index.html` - app entry
- `src/main.js` - scene, renderer, GUI
- `src/planet.js` - planet generation and update logic
- `src/styles.css` - minimal styling

Notes

- Tweak parameters in the GUI to change noise, elevation and resolution.
- To produce a different planet, change the `seed` in the GUI and hit "regen".
