import * as THREE from 'three';
import SimplexNoise from 'simplex-noise';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlanetParams {
  radius: number;
  segments: number;
  seed: number;
  // Continent noise layer
  continentScale: number;
  continentOctaves: number;
  // Biome noise layers (temperature + moisture)
  biomeScale: number;
  // High-frequency detail layer
  detailScale: number;
  detailStrength: number;
  // Displacement amount for land above sea level
  elevation: number;
  // Sea level: fraction of normalised height [0..1] below which is ocean
  seaLevel: number;
  // Debug: show greyscale height + flat-blue sea instead of biomes
  debugTerrain: boolean;
  // Canvas texture resolution (independent of geometry segments)
  textureWidth: number;
  textureHeight: number;
}

interface SampleResult {
  /** Raw (uncalibrated) height value from continent + detail noise */
  h: number;
  /** Temperature noise value, roughly -1..1 */
  temp: number;
  /** Moisture noise value, roughly -1..1 */
  moisture: number;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

export const DEFAULT_PARAMS: PlanetParams = {
  radius: 1,
  segments: 128,
  seed: 42,
  continentScale: 1.2,
  continentOctaves: 5,
  biomeScale: 2.2,
  detailScale: 7.0,
  detailStrength: 0.18,
  elevation: 0.22,
  seaLevel: 0.45,
  debugTerrain: false,
  textureWidth: 2048,
  textureHeight: 1024,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Fractional Brownian Motion — returns value roughly in [-1, 1]. */
function fbm(
  noise: SimplexNoise,
  x: number, y: number, z: number,
  scale: number,
  octaves: number,
  persistence: number,
  lacunarity: number,
): number {
  let val = 0, amp = 1, freq = scale, total = 0;
  for (let i = 0; i < octaves; i++) {
    val   += noise.noise3D(x * freq, y * freq, z * freq) * amp;
    total += amp;
    amp   *= persistence;
    freq  *= lacunarity;
  }
  return val / total;
}

/** Linear interpolation of a single 0-255 colour channel — no heap allocation. */
function lerpC(a: number, b: number, t: number): number {
  return (a + (b - a) * t + 0.5) | 0;
}

/**
 * Map normalised height + biome values to a terrain colour.
 * Writes RGB directly into `data[idx..idx+2]` — no THREE.Color allocations.
 *
 * @param hN  Calibrated height in [0, 1]  (0 = deepest ocean floor, 1 = highest peak)
 * @param temp     Temperature noise roughly -1..1
 * @param moisture Moisture   noise roughly -1..1
 * @param seaLvl   Sea-level threshold in [0, 1]
 */
function biomeColor(
  data: Uint8ClampedArray, idx: number,
  hN: number, temp: number, moisture: number, seaLvl: number,
): void {
  // --- Ocean floor (texture only — ocean sphere mesh provides water appearance) ---
  if (hN < seaLvl) {
    const t = (1 - hN / seaLvl) * 0.75; // 0 = shore, 1 = deepest
    data[idx]     = lerpC(0xc8, 0x6a, t);
    data[idx + 1] = lerpC(0xb4, 0x5a, t);
    data[idx + 2] = lerpC(0x72, 0x30, t);
    return;
  }

  const beachTop = seaLvl + 0.04;

  // --- Beach / shore ---
  if (hN < beachTop) {
    data[idx] = 0xd4; data[idx + 1] = 0xbc; data[idx + 2] = 0x78;
    return;
  }

  // Remap land to [0, 1] with 0 = beach, 1 = highest peak
  const landH = (hN - beachTop) / Math.max(0.01, 1 - beachTop);

  // --- High altitude ---
  if (landH > 0.70) {
    const t = Math.min(1, (landH - 0.70) / 0.20); // rock → snow
    data[idx]     = lerpC(0x8a, 0xff, t);
    data[idx + 1] = lerpC(0x83, 0xff, t);
    data[idx + 2] = lerpC(0x7a, 0xff, t);
    return;
  }
  if (landH > 0.50) { // exposed rock
    data[idx] = 0x7a; data[idx + 1] = 0x7a; data[idx + 2] = 0x6a;
    return;
  }

  // --- Biome band (driven by temp + moisture) ---
  if (temp < -0.35) {
    const t = Math.min(1, (-temp - 0.35) / 0.30); // tundra → ice
    data[idx]     = lerpC(0xb8, 0xff, t);
    data[idx + 1] = lerpC(0xd4, 0xff, t);
    data[idx + 2] = lerpC(0xc8, 0xff, t);
    return;
  }
  if (temp > 0.30 && moisture < -0.20) { // desert
    data[idx] = 0xc8; data[idx + 1] = 0xa0; data[idx + 2] = 0x50;
  } else if (temp > 0.20 && moisture > 0.25) { // tropical forest
    data[idx] = 0x1a; data[idx + 1] = 0x6b; data[idx + 2] = 0x1a;
  } else if (temp < 0.00) { // taiga / conifer
    data[idx] = 0x3a; data[idx + 1] = 0x7a; data[idx + 2] = 0x42;
  } else if (moisture > 0.05) { // temperate forest
    data[idx] = 0x2d; data[idx + 1] = 0x8c; data[idx + 2] = 0x3a;
  } else { // grassland
    data[idx] = 0x7a; data[idx + 1] = 0xb8; data[idx + 2] = 0x48;
  }
}

/**
 * Debug view: pure greyscale land height + solid blue ocean.
 * Writes directly into `data[idx..idx+2]` — no heap allocation.
 */
function debugColor(data: Uint8ClampedArray, idx: number, hN: number, seaLvl: number): void {
  if (hN < seaLvl) {
    const depth = 1 - hN / seaLvl;
    data[idx]     = (0.05 * 255 + 0.5) | 0;
    data[idx + 1] = ((0.25 - depth * 0.15) * 255 + 0.5) | 0;
    data[idx + 2] = ((0.55 - depth * 0.20) * 255 + 0.5) | 0;
  } else {
    const v = ((hN - seaLvl) / Math.max(0.01, 1 - seaLvl) * 255 + 0.5) | 0;
    data[idx] = data[idx + 1] = data[idx + 2] = v;
  }
}

// ---------------------------------------------------------------------------
// Planet class
// ---------------------------------------------------------------------------

export class Planet {
  params: PlanetParams;

  /** Three.js group containing terrain mesh + ocean mesh. Add this to the scene. */
  group!: THREE.Group;
  mesh!: THREE.Mesh;
  oceanMesh!: THREE.Mesh;

  // Noise instances — one per layer so each layer is truly independent
  private _nContinent!: SimplexNoise;
  private _nTemp!: SimplexNoise;
  private _nMoisture!: SimplexNoise;
  private _nDetail!: SimplexNoise;

  /**
   * Calibrated min/max of the raw h noise field.
   * Measured by sampling ~2 000 points after every noise rebuild.
   * Used to remap h to a true [0, 1] range so seaLevel means what it says.
   */
  private _hMin = -1;
  private _hMax =  1;

  constructor(params: Partial<PlanetParams> = {}) {
    this.params = { ...DEFAULT_PARAMS, ...params };
    this._buildNoise();
    this._calibrate();
    this._createMesh();
  }

  // -------------------------------------------------------------------------
  // Noise
  // -------------------------------------------------------------------------

  private _buildNoise(): void {
    const s = this.params.seed.toString();
    this._nContinent = new SimplexNoise(s + '_cont');
    this._nTemp      = new SimplexNoise(s + '_temp');
    this._nMoisture  = new SimplexNoise(s + '_mois');
    this._nDetail    = new SimplexNoise(s + '_det');
  }

  /**
   * Sample ~2 000 Fibonacci-distributed points on the unit sphere to find
   * the actual min/max of the raw h field for this seed + noise config.
   * This lets _norm() map h to a true [0, 1] regardless of FBM amplitude.
   */
  private _calibrate(): void {
    let hMin = Infinity, hMax = -Infinity;
    const N = 2000;
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < N; i++) {
      const cosTheta = 1 - (i / (N - 1)) * 2;       // y component
      const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
      const phi      = goldenAngle * i;
      // Use Three.js SphereGeometry sign convention: x is negated
      const x = -Math.cos(phi) * sinTheta;
      const y = cosTheta;
      const z =  Math.sin(phi) * sinTheta;
      const { h } = this._sample(x, y, z);
      if (h < hMin) hMin = h;
      if (h > hMax) hMax = h;
    }
    this._hMin = hMin;
    this._hMax = hMax;
  }

  /**
   * Remap raw noise h → calibrated [0, 1].
   * 0 = absolute lowest point,  1 = absolute highest point on this planet.
   * seaLevel=0.45 therefore means "45 % of height range is below water".
   */
  private _norm(h: number): number {
    const range = this._hMax - this._hMin;
    if (range === 0) return 0.5;
    return Math.min(1, Math.max(0, (h - this._hMin) / range));
  }

  /** Sample all four noise layers at a unit-sphere point (x, y, z). */
  private _sample(x: number, y: number, z: number): SampleResult {
    const { continentScale, continentOctaves, biomeScale, detailScale, detailStrength } = this.params;
    const continent = fbm(this._nContinent, x, y, z, continentScale, continentOctaves, 0.50, 2.0);
    const detail    = fbm(this._nDetail,    x, y, z, detailScale,    4,                0.45, 2.1) * detailStrength;
    const h         = continent + detail;
    const temp      = fbm(this._nTemp,     x, y, z, biomeScale,       3, 0.55, 2.0);
    const moisture  = fbm(this._nMoisture, x, y, z, biomeScale * 0.8, 3, 0.50, 2.0);
    return { h, temp, moisture };
  }

  // -------------------------------------------------------------------------
  // Geometry
  // -------------------------------------------------------------------------

  /**
   * Displace sphere vertices:
   *  - Below seaLevel  → recessed inside the ocean sphere  (radius - depth * elevation * 0.5)
   *  - Above seaLevel  → raised above the ocean sphere     (ocean_r + gap + aboveSea * elevation)
   *
   * OCEAN_R = radius + 0.002  (the actual ocean mesh radius)
   * Land vertices start at OCEAN_R + 0.002 so even a barely-above-sea-level
   * beach still pokes out of the water sphere without z-fighting.
   */
  private _displaceGeometry(geo: THREE.SphereGeometry): void {
    const { seaLevel, elevation, radius } = this.params;
    const OCEAN_R   = radius + 0.002; // must match _buildOcean
    const LAND_BASE = OCEAN_R  + 0.002;
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const v   = new THREE.Vector3();

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).normalize();
      const { h } = this._sample(v.x, v.y, v.z);
      const hN    = this._norm(h); // calibrated [0, 1]

      let r: number;
      if (hN <= seaLevel) {
        // Sink ocean floor — deeper = more recessed
        const depth = 1 - hN / Math.max(0.0001, seaLevel); // 0 = at shore, 1 = deepest
        r = radius - depth * elevation * 0.5;
      } else {
        // Raise land proportionally above the ocean sphere
        const aboveSea = (hN - seaLevel) / Math.max(0.01, 1 - seaLevel);
        r = LAND_BASE + aboveSea * elevation;
      }

      v.multiplyScalar(r);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
  }

  // -------------------------------------------------------------------------
  // Texture
  // -------------------------------------------------------------------------

  /** Bake a full-resolution equirectangular canvas texture. */
  private _buildTexture(): THREE.CanvasTexture {
    const { textureWidth: W, textureHeight: H, seaLevel, debugTerrain } = this.params;
    const canvas = document.createElement('canvas');
    canvas.width  = W;
    canvas.height = H;
    const ctx  = canvas.getContext('2d')!;
    const img  = ctx.createImageData(W, H);
    const data = img.data;

    for (let py = 0; py < H; py++) {
      const phi    = (py / H) * Math.PI;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);
      for (let px = 0; px < W; px++) {
        const theta = (px / W) * 2 * Math.PI;
        // Match Three.js SphereGeometry vertex convention exactly:
        //   x = -cos(theta) * sin(phi)
        //   y =  cos(phi)
        //   z =  sin(theta) * sin(phi)
        const x = -sinPhi * Math.cos(theta);
        const y = cosPhi;
        const z = sinPhi * Math.sin(theta);

        const { h, temp, moisture } = this._sample(x, y, z);
        const hN  = this._norm(h); // calibrated — same scale as _displaceGeometry
        const idx = (py * W + px) * 4;
        if (debugTerrain) {
          debugColor(data, idx, hN, seaLevel);
        } else {
          biomeColor(data, idx, hN, temp, moisture, seaLevel);
        }
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  // -------------------------------------------------------------------------
  // Ocean sphere
  // -------------------------------------------------------------------------

  private _buildOcean(): void {
    const { radius, segments } = this.params;
    const oceanGeo = new THREE.SphereGeometry(
      radius + 0.002,          // OCEAN_R — must match OCEAN_R in _displaceGeometry
      Math.max(64, segments),
      Math.max(64, segments),
    );
    const oceanMat = new THREE.MeshPhysicalMaterial({
      color:             0x0a4a9a,
      emissive:          0x001133,
      emissiveIntensity: 0.15,
      transparent: true,
      opacity:     0.82,
      roughness:   0.05,
      metalness:   0.0,
      side:        THREE.FrontSide,
    });

    if (this.oceanMesh) {
      this.oceanMesh.geometry.dispose();
      (this.oceanMesh.material as THREE.Material).dispose();
      this.oceanMesh.geometry = oceanGeo;
      this.oceanMesh.material = oceanMat;
    } else {
      this.oceanMesh = new THREE.Mesh(oceanGeo, oceanMat);
    }
  }

  // -------------------------------------------------------------------------
  // Mesh assembly
  // -------------------------------------------------------------------------

  private _createMesh(): void {
    const { radius, segments } = this.params;
    const geo = new THREE.SphereGeometry(radius, segments, segments);
    this._displaceGeometry(geo);

    const tex = this._buildTexture();
    const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, metalness: 0.0 });

    if (this.mesh) {
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.MeshStandardMaterial).map?.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh.geometry = geo;
      this.mesh.material = mat;
    } else {
      this.mesh = new THREE.Mesh(geo, mat);
    }

    this._buildOcean();

    if (!this.group) {
      this.group = new THREE.Group();
      this.group.add(this.mesh);
      this.group.add(this.oceanMesh);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Rebuild the planet with (optionally updated) params. */
  update(params: Partial<PlanetParams>): void {
    Object.assign(this.params, params);
    this._buildNoise();
    this._calibrate();
    this._createMesh();
  }
}
