import * as THREE from 'three';
import type { PlanetParams } from './planet';

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

// Full-screen quad vert — PlaneGeometry(2,2) spans clip space already.
const CLOUD_BAKE_VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// Bake fragment: full noise evaluation into an RGBA equirectangular texture.
const CLOUD_BAKE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform float uTime;
uniform float uCloudScale;
uniform float uCloudDensity;
uniform float uCloudOpacity;

// ---------------------------------------------------------------------------
// Simplex 3-D noise — Ashima Arts (MIT licence)
// ---------------------------------------------------------------------------
vec3 _mod289(vec3 x){return x-floor(x*(1./289.))*289.;}
vec4 _mod289(vec4 x){return x-floor(x*(1./289.))*289.;}
vec4 _perm(vec4 x){return _mod289(((x*34.)+10.)*x);}
vec4 _tiSqrt(vec4 r){return 1.79284291400159-.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1./6.,1./3.);
  const vec4 D=vec4(0.,.5,1.,2.);
  vec3 i=floor(v+dot(v,C.yyy));
  vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);
  vec3 l=1.-g;
  vec3 i1=min(g.xyz,l.zxy);
  vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;
  vec3 x2=x0-i2+C.yyy;
  vec3 x3=x0-D.yyy;
  i=_mod289(i);
  vec4 p=_perm(_perm(_perm(
    i.z+vec4(0.,i1.z,i2.z,1.))+
    i.y+vec4(0.,i1.y,i2.y,1.))+
    i.x+vec4(0.,i1.x,i2.x,1.));
  float n_=.142857142857;
  vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z);
  vec4 y_=floor(j-7.*x_);
  vec4 x=x_*ns.x+ns.yyyy;
  vec4 y=y_*ns.x+ns.yyyy;
  vec4 h=1.-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);
  vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.+1.;
  vec4 s1=floor(b1)*2.+1.;
  vec4 sh=-step(h,vec4(0.));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
  vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);
  vec3 p1=vec3(a0.zw,h.y);
  vec3 p2=vec3(a1.xy,h.z);
  vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=_tiSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.);
  m=m*m;
  return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

// fBm for clouds — 5 octaves max
float cfbm(vec3 p, int octaves) {
  float val=0.,amp=0.5,freq=1.,tot=0.;
  for(int i=0;i<5;i++){
    if(i>=octaves) break;
    val  +=snoise(p*freq)*amp;
    tot  +=amp;
    amp  *=0.50;
    freq *=2.07;
  }
  return val/max(tot,0.0001);
}

void main() {
  // Reconstruct unit-sphere direction from equirectangular UV
  float phi   = vUv.y * 3.14159265359;
  float theta = vUv.x * 6.28318530718;
  vec3 dir = vec3(-sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta));

  float lo = uCloudDensity - 0.10;
  float hi = uCloudDensity + 0.32;

  // Domain-warp for organic puffiness
  float ws = uCloudScale * 0.55;
  vec3 warp = vec3(
    cfbm(dir * ws + vec3( uTime*0.018,  0.000,       0.000       ), 2),
    cfbm(dir * ws + vec3( 0.000,        uTime*0.015, uTime*0.013 ), 2),
    cfbm(dir * ws + vec3( uTime*0.014,  uTime*0.019, 0.000       ), 2)
  );

  float cloud = cfbm(dir * uCloudScale + warp * 0.72, 4);
  float cover = smoothstep(lo, hi, cloud);

  // Colour: bright white core, soft blue-tinted wispy edges
  vec3 coreColor = vec3(1.00, 1.00, 1.00);
  vec3 edgeColor = vec3(0.75, 0.84, 0.97);
  vec3 finalColor = mix(edgeColor, coreColor, cover * cover);
  // Fake volumetric self-shadow
  finalColor *= 1.0 - 0.32 * smoothstep(0.45, 0.95, cover);

  float alpha = cover * cover * uCloudOpacity;

  gl_FragColor = vec4(finalColor, alpha);
}
`;

// ---------------------------------------------------------------------------
// Cloud render shader — one texture lookup per fragment, plus limb darkening
// ---------------------------------------------------------------------------

const CLOUD_RENDER_VERT = /* glsl */`
varying vec3 vLocalPos;
void main() {
  vLocalPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CLOUD_RENDER_FRAG = /* glsl */`
precision mediump float;
varying vec3 vLocalPos;

uniform sampler2D uCloudTex;

void main() {
  // Equirectangular UV from local sphere direction
  vec3 d  = normalize(vLocalPos);
  float u = atan(d.z, -d.x) / 6.28318530718 + 0.5;
  float v = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265359;
  vec4 t  = texture2D(uCloudTex, vec2(u, v));

  // Limb-darkening: fade alpha at grazing angles for a rounded appearance
  vec3 viewDir = normalize(cameraPosition - vLocalPos);
  float rim    = max(0.0, dot(normalize(vLocalPos), viewDir));
  float limb   = mix(0.55, 1.0, rim);

  gl_FragColor = vec4(t.rgb, t.a * limb);
}
`;

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

  constructor(params: PlanetParams, renderer: THREE.WebGLRenderer) {
    this._renderer = renderer;
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
          uCloudTex: { value: this._bakeTarget.texture },
        },
        transparent: true,
        depthWrite:  false,
        side:        THREE.FrontSide,
      });
    }

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
