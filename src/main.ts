import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import GUI from 'lil-gui';
import { Planet, DEFAULT_PARAMS } from './planet.js';
import { PlanetParams } from './types/planetParams.js';

type PrefabParams = Omit<PlanetParams, 'topoMap' | 'moistureView' | 'tempView' | 'fancyOcean' | 'showIceCaps' | 'showGrid'>;

interface PlanetPrefabFile {
  version: 1;
  params: PrefabParams;
}

const EXCLUDED_PREFAB_KEYS = ['topoMap', 'moistureView', 'tempView', 'fancyOcean', 'showIceCaps', 'showGrid'] as const;

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------
const container = document.getElementById('app') as HTMLDivElement;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080d1a);

// Star field — all stars placed on a shell between 80 and 200 units from origin
{
  const STAR_COUNT = 1000;
  const MIN_R = 80, MAX_R = 200;
  const pos = new Float32Array(STAR_COUNT * 3);
  for (let i = 0; i < STAR_COUNT; i++) {
    // Uniform direction on sphere
    const u = Math.random(), v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi   = Math.acos(2 * v - 1);
    const r     = MIN_R + Math.random() * (MAX_R - MIN_R);
    pos[i * 3 + 0] = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    pos[i * 3 + 2] = r * Math.cos(phi);
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.25 })));
}

// ---------------------------------------------------------------------------
// Camera + Controls
// ---------------------------------------------------------------------------
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, DEFAULT_PARAMS.radius * 3.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function updateZoom(): void {
  const r = params.radius;
  controls.minDistance = r * 1.15;
  controls.maxDistance = r * 20;
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------
const ambientLight = new THREE.AmbientLight(0x334466, 0.8);
scene.add(ambientLight);
const sun = new THREE.DirectionalLight(0xfff4e0, 1.4);
sun.position.set(6, 4, 5);
scene.add(sun);

// ---------------------------------------------------------------------------
// Planet
// ---------------------------------------------------------------------------
const params: PlanetParams = { ...DEFAULT_PARAMS, seed: Math.random() };
updateZoom();

const planet = new Planet(params, renderer);
scene.add(planet.group);

function refreshGuiDisplays(root: GUI): void {
  root.controllersRecursive().forEach((controller) => controller.updateDisplay());
}

function buildPrefabParams(source: PlanetParams): PrefabParams {
  const prefab = { ...source } as Partial<PlanetParams>;
  for (const key of EXCLUDED_PREFAB_KEYS) delete prefab[key];
  return prefab as PrefabParams;
}

function readPrefabParams(payload: unknown): Partial<PrefabParams> | null {
  if (!payload || typeof payload !== 'object') return null;

  const maybeWrapped = 'params' in payload
    ? (payload as { params?: unknown }).params
    : payload;

  if (!maybeWrapped || typeof maybeWrapped !== 'object') return null;

  const candidate = maybeWrapped as Record<string, unknown>;
  const result: Partial<PrefabParams> = {};

  for (const key of Object.keys(DEFAULT_PARAMS) as Array<keyof PlanetParams>) {
    if (EXCLUDED_PREFAB_KEYS.includes(key as typeof EXCLUDED_PREFAB_KEYS[number])) continue;

    const value = candidate[key];
    if (value === undefined) continue;

    const defaultValue = DEFAULT_PARAMS[key];
    if (typeof defaultValue === 'number' && typeof value === 'number' && Number.isFinite(value)) {
      (result as Record<string, unknown>)[key] = value;
    }
    if (typeof defaultValue === 'boolean' && typeof value === 'boolean') {
      (result as Record<string, unknown>)[key] = value;
    }
  }

  return Object.keys(result).length > 0 ? result : null;
}

function downloadPrefab(): void {
  const prefab: PlanetPrefabFile = {
    version: 1,
    params: buildPrefabParams(params),
  };

  const blob = new Blob([JSON.stringify(prefab, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `planet-prefab-${Math.round(params.seed * 1000)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function applyImportedPrefab(prefab: Partial<PrefabParams>): void {
  Object.assign(params, prefab);
  rebuild();
  refreshGuiDisplays(gui);
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------
const gui = new GUI({ title: 'Planet Generator' });

const prefabInput = document.createElement('input');
prefabInput.type = 'file';
prefabInput.accept = '.json,application/json';
prefabInput.style.display = 'none';
document.body.appendChild(prefabInput);

const shapeFolder = gui.addFolder('Shape');
shapeFolder.add(params, 'radius',    0.5,  3.0,  0.05).onFinishChange(rebuild);
shapeFolder.add(params, 'elevation', 0,    0.5,  0.01).onFinishChange(rebuild);
const seaLevelCtrl = shapeFolder.add(params, 'seaLevel',   0.0,  0.85, 0.005 ).name('sea level'  ).onFinishChange(rebuild);
shapeFolder.add(params, 'shoreWidth', 0.0,  0.15, 0.002).name('shore width').onFinishChange(rebuild);

const noiseFolder = gui.addFolder('Noise');
noiseFolder.add(params, 'continentScale',   0.3, 4.0,  0.05).name('continent scale'  ).onFinishChange(rebuild);
noiseFolder.add(params, 'continentOctaves', 1,   8,    1   ).name('continent octaves').onFinishChange(rebuild);
noiseFolder.add(params, 'biomeScale',       0.5, 6.0,  0.05).name('climate noise scale').onFinishChange(rebuild);
noiseFolder.add(params, 'detailScale',      2.0, 20.0, 0.5 ).name('detail scale'     ).onFinishChange(rebuild);
noiseFolder.add(params, 'detailStrength',   0,   0.5,  0.01).name('detail strength'  ).onFinishChange(rebuild);
noiseFolder.add(params, 'moistureMapScale',     0.4, 2.0,  0.05).name('moisture noise scale'    ).onFinishChange(rebuild);
noiseFolder.close();

const climateFolder = gui.addFolder('Climate');
climateFolder.add(params, 'latitudeFactor',       0.0, 2.0,  0.01).name('latitude factor'  ).onFinishChange(rebuild);
climateFolder.add(params, 'heightFactor',         0.0, 2.0,  0.01).name('height factor'    ).onFinishChange(rebuild);
climateFolder.add(params, 'tempOffset',          -1.0, 1.0,  0.01).name('temperature offset'   ).onFinishChange(rebuild);
climateFolder.add(params, 'moistureOffset',      -0.5, 0.5,  0.01).name('moisture offset'       ).onFinishChange(rebuild);

const seedFolder  = gui.addFolder('Seed');
const seedCtrl    = seedFolder.add(params, 'seed').onFinishChange(rebuild);
seedFolder.add({ regen() {
  params.seed = Math.random();
  seedCtrl.updateDisplay();
  rebuild();
} }, 'regen').name('🎲 Randomise seed');

const lightingState = { sunShadow: true };

const mapState = { mode: 'biome' };
const appearanceFolder = gui.addFolder('Appearance');
appearanceFolder.add(mapState, 'mode', ['biome', 'topographic', 'precipitation', 'temperature'])
  .name('map')
  .onChange((v: string) => {
    params.topoMap      = v === 'topographic';
    params.moistureView = v === 'precipitation';
    params.tempView     = v === 'temperature';
    rebuild();
  });
appearanceFolder.add(params, 'fancyOcean').name('fancy ocean').onChange((v: boolean) => {
  planet.update(params);
  updateZoom();
});
appearanceFolder.add(params, 'showIceCaps').name('ice caps').onChange(rebuild);
appearanceFolder.add(lightingState, 'sunShadow').name('sun shadow').onChange((v: boolean) => {
  sun.intensity          = 1.4;
  ambientLight.intensity = 0.8;
});
appearanceFolder.add(params, 'showGrid').name('grid').onChange((v: boolean) => {
  planet.gridMesh.visible = v;
});

const cloudFolder = gui.addFolder('Clouds');
cloudFolder.add(params, 'showClouds').name('show clouds').onChange((v: boolean) => {
  planet.cloudMesh.visible = v;
});
cloudFolder.add(params, 'cloudHeight',  0.01, 0.12, 0.005).name('altitude').onFinishChange(rebuildClouds);
cloudFolder.add(params, 'cloudScale',   0.5,  6.0,  0.1  ).name('scale').onFinishChange(rebuildClouds);
cloudFolder.add(params, 'cloudDensity', 0.0,  0.6,  0.01 ).name('coverage').onFinishChange(rebuildClouds);
cloudFolder.add(params, 'cloudOpacity', 0.1,  1.0,  0.01 ).name('opacity').onFinishChange(rebuildClouds);
cloudFolder.add(params, 'cloudWindSpeed', 0.0, 0.15, 0.001).name('wind speed').onFinishChange(rebuildClouds);

const prefabFolder = gui.addFolder('Prefabs');
prefabFolder.add({ exportJson: downloadPrefab }, 'exportJson').name('💾 Export JSON');
prefabFolder.add({ importJson() { prefabInput.click(); } }, 'importJson').name('📥 Import JSON');
prefabFolder.close();

function rebuild(): void {
  planet.update(params);
  updateZoom();
}

function rebuildClouds(): void {
  planet.updateClouds(params);
}

// ---------------------------------------------------------------------------
// Custom height map
// ---------------------------------------------------------------------------
const fileInput = document.createElement('input');
fileInput.type   = 'file';
fileInput.accept = 'image/*';
fileInput.style.display = 'none';
document.body.appendChild(fileInput);

const heightMapState = { label: 'none loaded' };

const heightMapFolder = gui.addFolder('Height Map');
const heightMapNameCtrl = heightMapFolder.add(heightMapState, 'label').name('file').disable();
heightMapFolder.add({ load() { fileInput.click(); } }, 'load').name('📂 Load image…');
heightMapFolder.add({ clear() {
  planet.clearHeightMap();
  heightMapState.label = 'none loaded';
  heightMapNameCtrl.updateDisplay();
} }, 'clear').name('✕ Clear (use noise)');
heightMapFolder.close();

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  planet.loadHeightMap(file).then(() => {
    heightMapState.label = file.name;
    heightMapNameCtrl.updateDisplay();
    seaLevelCtrl.updateDisplay();
  });
  fileInput.value = '';
});

prefabInput.addEventListener('change', async () => {
  const file = prefabInput.files?.[0];
  if (!file) return;

  try {
    const parsed = JSON.parse(await file.text()) as unknown;
    const prefab = readPrefabParams(parsed);
    if (!prefab) {
      throw new Error('Invalid prefab JSON.');
    }
    applyImportedPrefab(prefab);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not import prefab JSON.';
    window.alert(message);
  } finally {
    prefabInput.value = '';
  }
});

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------------------
// Pause / resume rotation
// ---------------------------------------------------------------------------
let paused = false;
const hint = document.getElementById('hint') as HTMLDivElement;
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') {
    e.preventDefault();
    paused = !paused;
    hint.innerHTML = paused
      ? 'Paused — press <kbd>Space</kbd> to resume'
      : 'Press <kbd>Space</kbd> to toggle rotation';
  }
});

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
let lastTime = performance.now();
const fpsEl = document.getElementById('fps') as HTMLDivElement;
let fpsFrames = 0;
let fpsAccum  = 0;

function animate(): void {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt  = Math.min((now - lastTime) * 0.001, 0.05); // cap delta at 50 ms
  lastTime  = now;

  fpsFrames++;
  fpsAccum += dt;
  if (fpsAccum >= 0.5) {
    fpsEl.textContent = `${Math.round(fpsFrames / fpsAccum)} fps`;
    fpsFrames = 0;
    fpsAccum  = 0;
  }

  planet.group.rotation.y += paused ? 0 : 0.0008;
  planet.tickClouds(paused ? 0 : dt, camera.position.length());

  // When sun shadow is disabled, keep the light locked to the camera so every
  // visible surface is always fully lit — no dark side.
  if (!lightingState.sunShadow) {
    sun.position.copy(camera.position);
  } else {
    sun.position.set(6, 4, 5);
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();
