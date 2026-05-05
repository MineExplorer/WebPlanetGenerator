import * as THREE from 'three';
import SimplexNoise from 'simplex-noise';
import { CloudLayer } from './cloudLayer';
import { BAKE_FRAG, BAKE_VERT } from './shaders/planetSurface';
import { PlanetParams } from './types/planetParams';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  segments: 256,
  seed: 42,
  continentScale: 1.2,
  continentOctaves: 5,
  biomeScale: 3.5,
  tempOffset: 0,
  moistureOffset: 0,
  detailScale: 12.0,
  detailStrength: 0.18,
  elevation: 0.2,
  seaLevel: 0.54,
  shoreWidth: 0.02,
  topoMap: false,
  moistureView: false,
  tempView: false,
  textureWidth: 2048,
  textureHeight: 1024,
  cloudScale: 3,
  cloudDensity: 0.14,
  cloudOpacity: 0.90,
  cloudHeight: 0.05,
  showClouds: true,
  fancyOcean: true,
  showGrid: false,
  elevationMoisture: true,
  moistureMapScale: 1.2,
  moistureMapStrength: 1,
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

// ---------------------------------------------------------------------------
// Planet class
// ---------------------------------------------------------------------------

export class Planet {
  params: PlanetParams;

  /** Three.js group containing terrain mesh + ocean mesh. Add this to the scene. */
  group!: THREE.Group;
  mesh!: THREE.Mesh;
  oceanMesh!: THREE.Mesh;
  gridMesh!: THREE.LineSegments;

  /** Cloud layer — rotate `cloudGroup` separately to drift clouds. */
  private _cloudLayer!: CloudLayer;

  get cloudGroup(): THREE.Group { return this._cloudLayer.group; }
  get cloudMesh():  THREE.Mesh  { return this._cloudLayer.mesh; }

  private _renderer: THREE.WebGLRenderer;

  // Single shared noise instance — layers are separated by coordinate offsets,
  // matching the GPU shader exactly so geometry and texture stay aligned.
  private _noise!: SimplexNoise;
  private _offContinent!: THREE.Vector3;
  private _offDetail!: THREE.Vector3;
  private _offTemp!: THREE.Vector3;
  private _offMoisture!: THREE.Vector3;

  /**
   * Calibrated min/max of the raw h noise field.
   * Measured by sampling ~2 000 points after every noise rebuild.
   * Used to remap h to a true [0, 1] range so seaLevel means what it says.
   */
  private _hMin = -1;
  private _hMax =  1;

  // Reusable GPU bake objects — created once, reused on every rebuild
  private _bakeScene!:  THREE.Scene;
  private _bakeCam!:    THREE.OrthographicCamera;
  private _bakeMat!:    THREE.ShaderMaterial;
  private _bakeTarget!: THREE.WebGLRenderTarget; // colour
  private _dispTarget!: THREE.WebGLRenderTarget; // grayscale hN for displacementMap

  /** Optional user-supplied displacement texture (overrides procedural noise). */
  private _customDispTex: THREE.Texture | null = null;

  constructor(params: Partial<PlanetParams> = {}, renderer: THREE.WebGLRenderer) {
    this.params    = { ...DEFAULT_PARAMS, ...params };
    this._renderer = renderer;
    this._initBakePipeline();
    this.rebuild();
  }

  // -------------------------------------------------------------------------
  // Noise
  // -------------------------------------------------------------------------

  private _buildNoise(): void {
    this._noise        = new SimplexNoise(this.params.seed.toString());
    this._offContinent  = this._layerOffset(0);
    this._offDetail     = this._layerOffset(1);
    this._offTemp       = this._layerOffset(2);
    this._offMoisture   = this._layerOffset(3);
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

  /** Sample all four noise layers at a unit-sphere point (x, y, z). */
  private _sample(x: number, y: number, z: number): SampleResult {
    const { continentScale, continentOctaves, biomeScale, detailScale, detailStrength } = this.params;
    // Pre-add per-layer offsets so the CPU uses the same noise field as the GPU shader
    const continent = fbm(this._noise,
      x + this._offContinent.x, y + this._offContinent.y, z + this._offContinent.z,
      continentScale, continentOctaves, 0.50, 2.0);
    const detail = fbm(this._noise,
      x + this._offDetail.x, y + this._offDetail.y, z + this._offDetail.z,
      detailScale, 4, 0.45, 2.1) * detailStrength;
    const h = continent + detail;
    const temp = fbm(this._noise,
      x + this._offTemp.x, y + this._offTemp.y, z + this._offTemp.z,
      biomeScale, 3, 0.55, 2.0);
    const moisture = fbm(this._noise,
      x + this._offMoisture.x, y + this._offMoisture.y, z + this._offMoisture.z,
      biomeScale * 0.8, 3, 0.50, 2.0);
    return { h, temp, moisture };
  }

  // -------------------------------------------------------------------------
  // GPU texture baking
  // -------------------------------------------------------------------------

  /** Effective texture resolution — scales with radius so larger planets stay sharp. */
  private _texSize(): { w: number; h: number } {
    const { textureWidth, textureHeight, radius } = this.params;
    return {
      w: Math.max(1, Math.round(textureWidth  * radius)),
      h: Math.max(1, Math.round(textureHeight * radius)),
    };
  }

  /**
   * Create the reusable off-screen bake pipeline once.
   * A PlaneGeometry(2,2) covers clip space exactly, so the ortho camera +
   * render target give us a full rect render without any matrix transforms.
   */
  private _initBakePipeline(): void {
    const { w: W, h: H } = this._texSize();

    this._bakeScene  = new THREE.Scene();
    this._bakeCam    = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this._bakeTarget = new THREE.WebGLRenderTarget(W, H, {
      minFilter:  THREE.LinearFilter,
      magFilter:  THREE.LinearFilter,
      colorSpace: THREE.SRGBColorSpace,
    });
    // Displacement map must be linear (not sRGB) so heights are not gamma-corrected
    this._dispTarget = new THREE.WebGLRenderTarget(W, H, {
      minFilter:  THREE.LinearFilter,
      magFilter:  THREE.LinearFilter,
    });
    // RepeatWrapping on the displacement map ensures UV.x=1 wraps to UV.x=0,
    // so both sides of the SphereGeometry seam sample identical texels — no gap.
    this._dispTarget.texture.wrapS  = THREE.RepeatWrapping;
    this._bakeTarget.texture.wrapS  = THREE.RepeatWrapping;

    this._bakeMat = new THREE.ShaderMaterial({
      vertexShader:   BAKE_VERT,
      fragmentShader: BAKE_FRAG,
      uniforms: {
        uHMin:             { value: this._hMin },
        uHMax:             { value: this._hMax },
        uSeaLevel:         { value: this.params.seaLevel },
        uShoreWidth:       { value: this.params.shoreWidth },
        uContinentScale:   { value: this.params.continentScale },
        uContinentOctaves: { value: this.params.continentOctaves },
        uBiomeScale:       { value: this.params.biomeScale },
        uDetailScale:      { value: this.params.detailScale },
        uDetailStrength:   { value: this.params.detailStrength },
        uTopoMap:          { value: this.params.topoMap ? 1 : 0 },
        uMoistureView:     { value: this.params.moistureView ? 1 : 0 },
        uTempView:         { value: this.params.tempView ? 1 : 0 },
        uFancyOcean:       { value: this.params.fancyOcean ? 1 : 0 },
        uMode:             { value: 0 },
        uTempOffset:        { value: this.params.tempOffset },
        uMoistureOffset:    { value: this.params.moistureOffset },
        uElevationMoisture: { value: this.params.elevationMoisture ? 1 : 0 },
        uUseCustomMap:      { value: 0 },
        uCustomHeightTex:   { value: null },
        uOffContinent:        { value: this._layerOffset(0) },
        uOffDetail:           { value: this._layerOffset(1) },
        uOffTemp:             { value: this._layerOffset(2) },
        uOffMoisture:         { value: this._layerOffset(3) },
        uOffMoistureMap:      { value: this._layerOffset(4) },
        uMoistureMapScale:    { value: this.params.moistureMapScale },
        uMoistureMapStrength: { value: this.params.moistureMapStrength },
      },
    });

    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._bakeMat);
    this._bakeScene.add(quad);
  }

  /**
   * Compute a deterministic vec3 offset from the seed for a given layer index.
   * Adding this offset to the noise input coordinates "seeds" that layer.
   */
  private _layerOffset(layer: number): THREE.Vector3 {
    const s = this.params.seed;
    const hash = (n: number) => { const x = Math.sin(n) * 43758.5453123; return x - Math.floor(x); };
    return new THREE.Vector3(
      (hash(s * 127.1 + layer * 311.7) - 0.5) * 200,
      (hash(s * 269.5 + layer * 183.3) - 0.5) * 200,
      (hash(s * 113.9 + layer * 537.1) - 0.5) * 200,
    );
  }

  /** Render colour + displacement textures to their respective render targets. */
  private _bakeGPUTextures(): { colorTex: THREE.Texture; dispTex: THREE.Texture } {
    const u = this._bakeMat.uniforms;
    const p = this.params;
    u.uHMin.value             = this._hMin;
    u.uHMax.value             = this._hMax;
    u.uSeaLevel.value         = p.seaLevel;
    u.uShoreWidth.value       = p.shoreWidth;
    u.uContinentScale.value   = p.continentScale;
    u.uContinentOctaves.value = p.continentOctaves;
    u.uBiomeScale.value       = p.biomeScale;
    u.uDetailScale.value      = p.detailScale;
    u.uDetailStrength.value   = p.detailStrength;
    u.uTopoMap.value           = p.topoMap ? 1 : 0;
    u.uMoistureView.value      = p.moistureView ? 1 : 0;
    u.uTempView.value          = p.tempView ? 1 : 0;
    u.uFancyOcean.value        = p.fancyOcean ? 1 : 0;
    u.uTempOffset.value        = p.tempOffset;
    u.uMoistureOffset.value    = p.moistureOffset;
    u.uElevationMoisture.value = p.elevationMoisture ? 1 : 0;
    u.uOffContinent.value        = this._layerOffset(0);
    u.uOffDetail.value           = this._layerOffset(1);
    u.uOffTemp.value             = this._layerOffset(2);
    u.uOffMoisture.value         = this._layerOffset(3);
    u.uOffMoistureMap.value      = this._layerOffset(4);
    u.uMoistureMapScale.value    = p.moistureMapScale;
    u.uMoistureMapStrength.value = p.moistureMapStrength;

    // Custom height map: feed texture directly into the shader so both the colour
    // and displacement passes derive hN from the real map, not from noise.
    const hasCustom = !!this._customDispTex;
    u.uUseCustomMap.value    = hasCustom ? 1 : 0;
    u.uCustomHeightTex.value = hasCustom ? this._customDispTex : null;

    // Pass 1: colour
    u.uMode.value = 0;
    this._renderer.setRenderTarget(this._bakeTarget);
    this._renderer.render(this._bakeScene, this._bakeCam);

    // Pass 2: grayscale hN for displacementMap
    // Skipped when a custom height map is loaded — use it directly instead.
    let dispTex: THREE.Texture;
    if (this._customDispTex) {
      dispTex = this._customDispTex;
    } else {
      u.uMode.value = 1;
      this._renderer.setRenderTarget(this._dispTarget);
      this._renderer.render(this._bakeScene, this._bakeCam);
      dispTex = this._dispTarget.texture;
    }

    this._renderer.setRenderTarget(null);

    return { colorTex: this._bakeTarget.texture, dispTex };
  }

  // -------------------------------------------------------------------------
  // Ocean sphere
  // -------------------------------------------------------------------------

  private _buildOcean(): void {
    const { radius, segments, seaLevel, elevation, fancyOcean } = this.params;
    // Sits at exactly the height where hN = seaLevel in displacement
    const oceanRadius = radius + (seaLevel - 0.5) * elevation;
    const oceanGeo = new THREE.SphereGeometry(
      oceanRadius,
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
    this.oceanMesh.visible = fancyOcean;
  }

  // -------------------------------------------------------------------------
  // Cloud layer
  // -------------------------------------------------------------------------

  // -------------------------------------------------------------------------
  // Globe grid
  // -------------------------------------------------------------------------

  private _buildGrid(): void {
    const { radius, elevation, showGrid } = this.params;
    // Sit just above the maximum possible terrain height
    const r = radius + elevation * 0.5 + radius * 0.003;
    const STEP   = 15; // degrees
    const SEG    = 128; // segments per line
    const positions: number[] = [];

    // Latitude parallels (skip poles — they collapse to a point)
    for (let lat = -90 + STEP; lat < 90; lat += STEP) {
      const phi = (lat * Math.PI) / 180;
      const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi);
      const circleR = r * cosPhi;
      const y = r * sinPhi;
      for (let i = 0; i < SEG; i++) {
        const t0 = (i / SEG) * Math.PI * 2;
        const t1 = ((i + 1) / SEG) * Math.PI * 2;
        positions.push(
          circleR * Math.cos(t0), y, circleR * Math.sin(t0),
          circleR * Math.cos(t1), y, circleR * Math.sin(t1),
        );
      }
    }

    // Longitude meridians
    for (let lon = 0; lon < 360; lon += STEP) {
      const theta = (lon * Math.PI) / 180;
      const cosTheta = Math.cos(theta), sinTheta = Math.sin(theta);
      for (let i = 0; i < SEG; i++) {
        const phi0 = -Math.PI / 2 + (i / SEG) * Math.PI;
        const phi1 = -Math.PI / 2 + ((i + 1) / SEG) * Math.PI;
        positions.push(
          r * Math.cos(phi0) * cosTheta, r * Math.sin(phi0), r * Math.cos(phi0) * sinTheta,
          r * Math.cos(phi1) * cosTheta, r * Math.sin(phi1), r * Math.cos(phi1) * sinTheta,
        );
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
    });

    if (this.gridMesh) {
      this.gridMesh.geometry.dispose();
      (this.gridMesh.material as THREE.Material).dispose();
      this.gridMesh.geometry = geo;
      this.gridMesh.material = mat;
    } else {
      this.gridMesh = new THREE.LineSegments(geo, mat);
    }
    this.gridMesh.visible = showGrid;
  }

  private _buildClouds(): void {
    if (!this._cloudLayer) {
      this._cloudLayer = new CloudLayer(this.params, this._renderer);
    } else {
      this._cloudLayer.build(this.params);
    }
  }

  /**
   * Advance the cloud animation by `dt` seconds.
   * Call once per frame from your render loop.
   */
  /** Advance the cloud animation by `dt` seconds.
   * Pass `camDist` (camera distance from planet centre) for dynamic bake resolution.
   * Call once per frame from your render loop.
   */
  tickClouds(dt: number, camDist?: number): void {
    this._cloudLayer?.tick(dt, camDist);
  }

  // -------------------------------------------------------------------------
  // Mesh assembly
  // -------------------------------------------------------------------------

  private _createMesh(): void {
    const { radius, segments, elevation } = this.params;

    const texBuildStart = performance.now();
    const { colorTex, dispTex } = this._bakeGPUTextures();
    console.log(`Textures baked (GPU) in ${(performance.now() - texBuildStart).toFixed(2)} ms`);

    // GPU displacement map: both colour and height are from the same noise evaluation,
    // so their biome edges are pixel-perfect aligned — no CPU/GPU noise mismatch.
    const geo = new THREE.SphereGeometry(radius, segments, segments);
    const mat = new THREE.MeshStandardMaterial({
      map:               colorTex,
      displacementMap:   dispTex,
      displacementScale: elevation,
      // Centre displacement around radius: terrain spans [radius - 0.5·e, radius + 0.5·e].
      // Planet visual size = radius regardless of elevation value.
      displacementBias:  -0.5 * elevation,
      roughness:         0.9,
      metalness:         0.0,
    });

    if (this.mesh) {
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh.geometry = geo;
      this.mesh.material = mat;
    } else {
      this.mesh = new THREE.Mesh(geo, mat);
    }

    this._buildOcean();

    this._buildGrid();

    if (!this.group) {
      this.group = new THREE.Group();
      this.group.add(this.mesh);
      this.group.add(this.oceanMesh);
      this.group.add(this.gridMesh);
    }

    this._buildClouds();

    if (!this.group.children.includes(this._cloudLayer.group)) {
      this.group.add(this._cloudLayer.group);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Rebuild the planet with (optionally updated) params. */
  update(params: Partial<PlanetParams>): void {
    Object.assign(this.params, params);
    this.rebuild();
  }

  /**
   * Load an image file as the displacement (height) map.
   * The image should be a greyscale equirectangular map where white = highest
   * and black = lowest. PNG or 16-bit PNG gives the best results.
   * Calls rebuild() automatically when the texture is ready.
   */
  loadHeightMap(file: File): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      new THREE.TextureLoader().load(
        url,
        (tex) => {
          URL.revokeObjectURL(url);
          tex.wrapS = THREE.RepeatWrapping;
          tex.colorSpace = THREE.LinearSRGBColorSpace;
          if (this._customDispTex) this._customDispTex.dispose();
          this._customDispTex = tex;
          // Most greyscale heightmaps have 0 = deep ocean, values rising to 1 at peaks.
          // The user can still adjust via the sea level slider.
          this.params.seaLevel = 0.001;
          this.rebuild();
          resolve();
        },
        undefined,
        (err) => { URL.revokeObjectURL(url); reject(err); },
      );
    });
  }

  /** Remove the custom height map and revert to procedural noise. */
  clearHeightMap(): void {
    if (this._customDispTex) {
      this._customDispTex.dispose();
      this._customDispTex = null;
    }
    this.rebuild();
  }

  /** Returns true when a custom height map is currently active. */
  get hasCustomHeightMap(): boolean { return this._customDispTex !== null; }

  /** Update only the cloud layer (scale, density, opacity, height, visibility). */
  updateClouds(params: Partial<PlanetParams>): void {
    Object.assign(this.params, params);
    this._cloudLayer?.build(this.params);
  }

  rebuild(): void {
    const startTime = performance.now();
    this._buildNoise();
    const step1Time = performance.now();
    console.log(`Noise rebuilt in ${(step1Time - startTime).toFixed(2)} ms`);
    this._calibrate();
    const step2Time = performance.now();
    console.log(`Noise calibrated in ${(step2Time - step1Time).toFixed(2)} ms`);
    // Resize render targets if effective resolution changed (radius or texture params)
    const { w: W, h: H } = this._texSize();
    if (this._bakeTarget.width !== W || this._bakeTarget.height !== H) {
      this._bakeTarget.setSize(W, H);
      this._dispTarget.setSize(W, H);
    }
    this._createMesh();
    const endTime = performance.now();
    console.log(`Planet rebuilt in ${(endTime - startTime).toFixed(2)} ms`);
  }
}
