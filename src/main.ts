import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import GUI from 'lil-gui';
import { Planet, DEFAULT_PARAMS, PlanetParams } from './planet.js';

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

// Star field
{
  const starGeo = new THREE.BufferGeometry();
  const positions = new Float32Array(3000).map(() => (Math.random() - 0.5) * 400);
  starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
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
scene.add(new THREE.AmbientLight(0x334466, 0.8));
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

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------
const gui = new GUI({ title: 'Planet Generator' });

const shapeFolder = gui.addFolder('Shape');
shapeFolder.add(params, 'radius',    0.5,  3.0,  0.05).onFinishChange(rebuild);
shapeFolder.add(params, 'segments',  32,   256,   8  ).onFinishChange(rebuild);
shapeFolder.add(params, 'elevation', 0,    0.5,  0.01).onFinishChange(rebuild);
shapeFolder.add(params, 'seaLevel',  0.0,  0.85, 0.01).name('sea level').onFinishChange(rebuild);

const noiseFolder = gui.addFolder('Noise');
noiseFolder.add(params, 'continentScale',   0.3, 4.0,  0.05).name('continent scale'  ).onFinishChange(rebuild);
noiseFolder.add(params, 'continentOctaves', 1,   8,    1   ).name('continent octaves').onFinishChange(rebuild);
noiseFolder.add(params, 'biomeScale',       0.5, 6.0,  0.05).name('climate noise scale').onFinishChange(rebuild);
noiseFolder.add(params, 'detailScale',      2.0, 20.0, 0.5 ).name('detail scale'     ).onFinishChange(rebuild);
noiseFolder.add(params, 'detailStrength',   0,   0.5,  0.01).name('detail strength'  ).onFinishChange(rebuild);

const climateFolder = gui.addFolder('Climate');
climateFolder.add(params, 'tempOffset',     -1.0, 1.0, 0.01).name('temperature offset').onFinishChange(rebuild);
climateFolder.add(params, 'moistureOffset', -1.0, 1.0, 0.01).name('moisture offset'   ).onFinishChange(rebuild);

const seedFolder  = gui.addFolder('Seed');
const seedCtrl    = seedFolder.add(params, 'seed').onFinishChange(rebuild);
seedFolder.add({ regen() {
  params.seed = Math.random();
  seedCtrl.updateDisplay();
  rebuild();
} }, 'regen').name('🎲 Randomise seed');

const debugFolder = gui.addFolder('Debug');
debugFolder.add(params, 'debugTerrain').name('height map only').onFinishChange(rebuild);
debugFolder.close();

const cloudFolder = gui.addFolder('Clouds');
cloudFolder.add(params, 'showClouds').name('show clouds').onChange((v: boolean) => {
  planet.cloudMesh.visible = v;
});
cloudFolder.add(params, 'cloudHeight',  0.01, 0.12, 0.005).name('altitude').onFinishChange(rebuild);
cloudFolder.add(params, 'cloudScale',   0.5,  6.0,  0.1  ).name('scale').onFinishChange(rebuildClouds);
cloudFolder.add(params, 'cloudDensity', 0.0,  0.6,  0.01 ).name('coverage').onFinishChange(rebuildClouds);
cloudFolder.add(params, 'cloudOpacity', 0.1,  1.0,  0.01 ).name('opacity').onFinishChange(rebuildClouds);

function rebuild(): void {
  planet.update(params);
  updateZoom();
}

function rebuildClouds(): void {
  planet.update({ ...params });
}

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

function animate(): void {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt  = Math.min((now - lastTime) * 0.001, 0.05); // cap delta at 50 ms
  lastTime  = now;

  planet.group.rotation.y      += paused ? 0 : 0.0008;
  planet.cloudGroup.rotation.y += paused ? 0 : 0.00035;
  planet.tickClouds(paused ? 0 : dt);

  controls.update();
  renderer.render(scene, camera);
}
animate();
