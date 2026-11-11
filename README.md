# Web Planet Generator

An interactive procedural planet generator that runs in the browser. Terrain, climate and biomes are generated from seeded Simplex noise, baked to textures on the GPU with GLSL shaders, and rendered with Three.js. A [lil-gui](https://lil-gui.georgealways.com/) panel controls all parameters.

## Tech stack

- [TypeScript](https://www.typescriptlang.org/) and [Vite](https://vitejs.dev/) (with `vite-plugin-glsl` for importing `.glsl` shaders)
- [Three.js](https://threejs.org/) — WebGL rendering, `OrbitControls` for camera
- [simplex-noise](https://www.npmjs.com/package/simplex-noise) — seeded noise
- [lil-gui](https://lil-gui.georgealways.com/) — parameter panel

## Features

- Procedural terrain: continents (multi-octave noise), high-frequency detail, sea level and shore width
- Climate model: temperature and moisture driven by latitude, elevation (lapse-rate cooling, orographic drying) and noise, with adjustable offsets
- Biome colouring, polar ice caps, and optional fancy ocean shading
- Map views: biome, topographic, precipitation and temperature
- Animated cloud layer with zonal wind bands
- Optional globe grid overlay, sun shadow, starfield background and FPS counter
- Custom height map: load an image to replace the noise-generated terrain
- Prefabs: export/import the current parameters as a JSON file
- Orbit camera (optionally following the planet's spin); press `Space` to pause/resume rotation

## Quick start

1. Install dependencies:

```bash
npm install
```

2. Start dev server:

```bash
npm run dev
```

3. Open the URL printed by Vite (usually http://localhost:5173)

4. Build for production (`npm run build`) and preview it locally (`npm run preview`).

## Notes

- Tweak parameters in the GUI (Shape, Noise, Climate, Seed, Appearance, Clouds, Prefabs, Height Map);
- To produce a different planet, change the `seed` in the Seed folder or press "🎲 Randomise seed".

## Screenshots
<img src="docs/screenshots/planet view.png" width="600" alt="Planet view">
<img src="docs/screenshots/heat map.png" width="600" alt="Temperature map">
