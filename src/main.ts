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
camera.position.set(0, 0, 3.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance   = 1.3;
controls.maxDistance   = 20;

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

const planet = new Planet(params);
scene.add(planet.group);

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------
const gui = new GUI({ title: 'Planet Generator' });

const shapeFolder = gui.addFolder('Shape');
shapeFolder.add(params, 'radius',    0.4,  3.0,  0.05).onFinishChange(rebuild);
shapeFolder.add(params, 'segments',  32,   256,   8  ).onFinishChange(rebuild);
shapeFolder.add(params, 'elevation', 0,    0.5,  0.01).onFinishChange(rebuild);
shapeFolder.add(params, 'seaLevel',  0.0,  0.85, 0.01).name('sea level').onFinishChange(rebuild);

const noiseFolder = gui.addFolder('Noise');
noiseFolder.add(params, 'continentScale',   0.3, 4.0,  0.05).name('continent scale'  ).onFinishChange(rebuild);
noiseFolder.add(params, 'continentOctaves', 1,   8,    1   ).name('continent octaves').onFinishChange(rebuild);
noiseFolder.add(params, 'biomeScale',       0.5, 6.0,  0.05).name('biome scale'      ).onFinishChange(rebuild);
noiseFolder.add(params, 'detailScale',      2.0, 20.0, 0.5 ).name('detail scale'     ).onFinishChange(rebuild);
noiseFolder.add(params, 'detailStrength',   0,   0.5,  0.01).name('detail strength'  ).onFinishChange(rebuild);

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

function rebuild(): void {
  planet.update(params);
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
// Render loop
// ---------------------------------------------------------------------------
function animate(): void {
  requestAnimationFrame(animate);
  planet.group.rotation.y += 0.0008;
  controls.update();
  renderer.render(scene, camera);
}
animate();
