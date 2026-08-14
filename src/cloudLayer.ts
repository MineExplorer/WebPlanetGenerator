import * as THREE from 'three';
import { PlanetParams } from './types/planetParams';
import { CLOUD_BAKE_FRAG, CLOUD_BAKE_VERT, CLOUD_RENDER_FRAG, CLOUD_RENDER_VERT } from './shaders/clouds';

// ---------------------------------------------------------------------------
// Cloud shaders
// ---------------------------------------------------------------------------
//
// Strategy: the cloud noise is expensive (10 fBm calls/fragment/frame).
// Instead, we bake it to a small equirectangular render target every
// BAKE_INTERVAL seconds, then sample that texture from the cloud sphere.
// GPU bilinear filtering provides smooth interpolation between noise samples;
// apparent cloud drift comes from rotating the sphere mesh (already done in
// main.ts) so no per-frame noise evaluation is needed at all.
// ---------------------------------------------------------------------------

const BAKE_INTERVAL = 0.05; // seconds between cloud texture rebakes (~20fps bake rate)

// Texture resolution scales with zoom level.
// These are the [width, height] pairs at the near/far extremes.
const BAKE_RES_FAR  = [256, 128] as const; // far zoom-out
// Distance multiples of planet radius that map to the near/far extremes.
const DIST_NEAR = 1.2;  // <= this distance → max resolution
const DIST_FAR  = 12.0; // >= this distance → min resolution

function bakeResForDist(camDist: number, radius: number): [number, number] {
  const r = radius || 1;
  // Normalise to [0, 1] along the log scale between DIST_NEAR·r and DIST_FAR·r
  const t = Math.max(0, Math.min(1,
    (Math.log(camDist / r) - Math.log(DIST_NEAR)) /
    (Math.log(DIST_FAR)  - Math.log(DIST_NEAR))
  ));
  // Snap to power-of-two steps: 2048, 1024, 512, 256
  const w = t < 0.42 ? 2048 : t < 0.75 ? 1024 : t < 0.95 ? 512 : 256;
  return [w, w >> 1];
}

// ---------------------------------------------------------------------------
// CloudLayer class
// ---------------------------------------------------------------------------

/**
 * Self-contained cloud layer.
 *
 * Optimisation: instead of evaluating 10 fBm calls per visible fragment per
 * frame, the full noise is baked into a small equirectangular texture at a
 * limited rate (BAKE_INTERVAL seconds).  The cloud sphere samples that texture
 * — a single bilinear tap per fragment.  GPU bilinear filtering provides smooth
 * interpolation; apparent cloud drift is provided by rotating the cloud group
 * in the main render loop (already done in main.ts) at zero extra cost.
 *
 * Add `group` to your scene.  Call `tick(dt)` every frame.
 */
export class CloudLayer {
  /** Rotate this to drift clouds independently of the planet. */
  group: THREE.Group;
  mesh!: THREE.Mesh;

  private _renderer: THREE.WebGLRenderer;

  // Render material (cheap: one texture tap + limb darkening)
  private _renderMat!: THREE.ShaderMaterial;
  private _terrainHeightTex: THREE.Texture;
  private _time      = 0;
  private _lastBake  = -BAKE_INTERVAL; // force immediate first bake
  private _planetRadius = 1;
  private _bakeW     = BAKE_RES_FAR[0] as number;
  private _bakeH     = BAKE_RES_FAR[1] as number;

  // Bake pipeline (runs rarely)
  private _bakeScene!:  THREE.Scene;
  private _bakeCam!:    THREE.OrthographicCamera;
  private _bakeTarget!: THREE.WebGLRenderTarget;
  private _bakeMat!:    THREE.ShaderMaterial;

  constructor(params: PlanetParams, renderer: THREE.WebGLRenderer, terrainHeightTex: THREE.Texture) {
    this._renderer         = renderer;
    this._terrainHeightTex = terrainHeightTex;
    this.group = new THREE.Group();
    this._initBakePipeline();
    this.build(params);
  }

  // -------------------------------------------------------------------------

  private _initBakePipeline(): void {
    this._bakeScene  = new THREE.Scene();
    this._bakeCam    = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this._bakeTarget = new THREE.WebGLRenderTarget(BAKE_RES_FAR[0], BAKE_RES_FAR[1], {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS:     THREE.RepeatWrapping, // seamless horizontal wrap
    });
    this._bakeMat = new THREE.ShaderMaterial({
      vertexShader:   CLOUD_BAKE_VERT,
      fragmentShader: CLOUD_BAKE_FRAG,
      uniforms: {
        uTime:         { value: 0 },
        uCloudScale:   { value: 2.5 },
        uCloudDensity: { value: 0.28 },
        uCloudOpacity: { value: 0.90 },
      },
    });
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._bakeMat);
    this._bakeScene.add(quad);
  }

  private _runBake(): void {
    this._bakeMat.uniforms.uTime.value = this._time;
    this._renderer.setRenderTarget(this._bakeTarget);
    this._renderer.render(this._bakeScene, this._bakeCam);
    this._renderer.setRenderTarget(null);
  }

  // -------------------------------------------------------------------------

  /** (Re)build geometry and update shader uniforms from PlanetParams. */
  build(params: PlanetParams): void {
    const { radius, cloudHeight, cloudScale, cloudDensity, cloudOpacity, segments, showClouds } = params;
    this._planetRadius = radius;  // keep for distance-based resolution
    const cloudRadius = radius * (1 + cloudHeight);
    const segs = Math.min(Math.max(48, segments), 96);
    const geo  = new THREE.SphereGeometry(cloudRadius, segs, segs);

    // Update bake uniforms
    const bu = this._bakeMat.uniforms;
    bu.uCloudScale.value   = cloudScale;
    bu.uCloudDensity.value = cloudDensity;
    bu.uCloudOpacity.value = cloudOpacity;

    // Force a fresh bake immediately after parameter change
    this._lastBake = -BAKE_INTERVAL;

    if (!this._renderMat) {
      this._renderMat = new THREE.ShaderMaterial({
        vertexShader:   CLOUD_RENDER_VERT,
        fragmentShader: CLOUD_RENDER_FRAG,
        uniforms: {
          uCloudTex:          { value: this._bakeTarget.texture },
          uTerrainHeightTex:  { value: this._terrainHeightTex },
          uPlanetRadius:      { value: 1 },
          uTerrainElevation:  { value: 0 },
          uCloudHeight:       { value: 0 },
          uCloudRotationY:    { value: 0 },
        },
        transparent: true,
        depthWrite:  false,
        side:        THREE.FrontSide,
      });
    }

    const ru = this._renderMat.uniforms;
    ru.uTerrainHeightTex.value = this._terrainHeightTex;
    ru.uPlanetRadius.value     = radius;
    ru.uTerrainElevation.value = params.elevation;
    ru.uCloudHeight.value      = cloudHeight;

    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh.geometry = geo;
    } else {
      this.mesh = new THREE.Mesh(geo, this._renderMat);
      this.group.add(this.mesh);
    }

    this.mesh.visible = showClouds;
  }

  /**
   * Advance cloud animation by `dt` seconds. Call once per frame.
   * Pass `camDist` (camera distance from planet centre) to enable
   * dynamic bake resolution — higher res when close, lower when far away.
   */
  tick(dt: number, camDist?: number): void {
    this._time += dt;
    this._renderMat.uniforms.uCloudRotationY.value = this.group.rotation.y;

    // Resize bake target if zoom level crossed a resolution threshold
    if (camDist !== undefined) {
      const [w, h] = bakeResForDist(camDist, this._planetRadius);
      if (w !== this._bakeW || h !== this._bakeH) {
        this._bakeW = w;
        this._bakeH = h;
        this._bakeTarget.setSize(w, h);
        // Re-bake immediately at the new resolution even if animation is paused
        this._runBake();
        this._lastBake = this._time;
      }
    }

    if (this._time - this._lastBake >= BAKE_INTERVAL) {
      this._lastBake = this._time;
      this._runBake();
    }
  }

  dispose(): void {
    this.mesh?.geometry.dispose();
    this._renderMat?.dispose();
    this._bakeMat?.dispose();
    this._bakeTarget?.dispose();
  }
}
