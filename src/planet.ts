import * as THREE from 'three';
import SimplexNoise from 'simplex-noise';
import { CloudLayer } from './cloud';

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
  // Climate additive offsets applied after latitude function + noise  [-1 .. 1]
  tempOffset: number;
  moistureOffset: number;
  // High-frequency detail layer
  detailScale: number;
  detailStrength: number;
  // Displacement amount for land above sea level
  elevation: number;
  // Sea level: fraction of normalised height [0..1] below which is ocean
  seaLevel: number;
  // Debug: show greyscale height + flat-blue sea instead of biomes
  topoMap: boolean;
  // Canvas texture resolution (independent of geometry segments)
  textureWidth: number;
  textureHeight: number;
  // Cloud layer
  cloudScale: number;
  cloudDensity: number;
  cloudOpacity: number;
  cloudHeight: number;
  showClouds: boolean;
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
  segments: 256,
  seed: 42,
  continentScale: 1.2,
  continentOctaves: 5,
  biomeScale: 3.5,
  tempOffset: 0.25,
  moistureOffset: 0,
  detailScale: 12.0,
  detailStrength: 0.2,
  elevation: 0.18,
  seaLevel: 0.54,
  topoMap: false,
  textureWidth: 2048,
  textureHeight: 1024,
  cloudScale: 3,
  cloudDensity: 0.14,
  cloudOpacity: 0.90,
  cloudHeight: 0.05,
  showClouds: true,
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
// GLSL shaders for GPU texture baking
// ---------------------------------------------------------------------------

const BAKE_VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  // PlaneGeometry(2,2) already spans clip space — skip all matrix math
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const BAKE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform float uHMin;
uniform float uHMax;
uniform float uSeaLevel;
uniform float uContinentScale;
uniform int   uContinentOctaves;
uniform float uBiomeScale;
uniform float uDetailScale;
uniform float uDetailStrength;
uniform int   uTopoMap;
uniform int   uMode; // 0 = colour, 1 = displacement grayscale
// Climate additive offsets (sliders)
uniform float uTempOffset;
uniform float uMoistureOffset;

// Per-layer seed offsets (computed from seed on CPU, applied to noise input)
uniform vec3 uOffContinent;
uniform vec3 uOffDetail;
uniform vec3 uOffTemp;
uniform vec3 uOffMoisture;

// ---------------------------------------------------------------------------
// Simplex 3D noise — Ashima Arts (MIT licence)
// ---------------------------------------------------------------------------
vec3 mod289v3(vec3 x){return x-floor(x*(1./289.))*289.;}
vec4 mod289v4(vec4 x){return x-floor(x*(1./289.))*289.;}
vec4 permute(vec4 x){return mod289v4(((x*34.)+10.)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-.85373472095314*r;}
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
  i=mod289v3(i);
  vec4 p=permute(permute(permute(
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
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.);
  m=m*m;
  return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

float fbm(vec3 p, vec3 offset, float scale, int octaves, float persistence, float lacunarity) {
  float val = 0.0, amp = 1.0, freq = scale, total = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    val   += snoise((p + offset) * freq) * amp;
    total += amp;
    amp   *= persistence;
    freq  *= lacunarity;
  }
  return val / total;
}

// ---------------------------------------------------------------------------
// Biome colour logic  (mirrors planet.ts CPU version)
// ---------------------------------------------------------------------------
vec3 biomeColor(float hN, float temp, float moisture, float seaLvl, vec3 dir, vec3 offT, vec3 offM) {
  if (hN < seaLvl) {
    float t = (1.0 - hN / seaLvl) * 0.75;
    return mix(vec3(0.784, 0.706, 0.447), vec3(0.416, 0.353, 0.188), t);
  }
  float beachTop = seaLvl + 0.04;
  if (hN < beachTop) return vec3(0.831, 0.737, 0.471);

  float landH = (hN - beachTop) / max(0.01, 1.0 - beachTop);

  if (landH > 0.70) {
    float t = min(1.0, (landH - 0.70) / 0.20);
    return mix(vec3(0.541, 0.514, 0.478), vec3(1.0), t); // rock → snow
  }
  if (landH > 0.50) return vec3(0.478, 0.478, 0.416); // exposed rock

  // High-frequency warp to break up biome boundary regularity
  float warpT = snoise((dir + offT * 0.013) * 11.0) * 0.13
              + snoise((dir + offT * 0.007) * 23.0) * 0.05;
  float warpM = snoise((dir + offM * 0.013) * 10.0) * 0.13
              + snoise((dir + offM * 0.007) * 21.0) * 0.05;
  float bt = clamp(temp     + warpT, -1.0, 1.0);
  float bm = clamp(moisture + warpM, -1.0, 1.0);

  if (bt < -0.60) {
    float t = min(1.0, (-bt - 0.55) / 0.20);
    return mix(vec3(0.722, 0.831, 0.784), vec3(1.0), t); // tundra → ice/snow
  }
  if (bt < -0.40) return vec3(0.573, 0.533, 0.380); // tundra
  if (bt > 0.30 && bm < -0.20) return vec3(0.784, 0.627, 0.314); // desert
  if (bt > 0.20 && bm >  0.25) return vec3(0.102, 0.420, 0.102); // tropical
  if (bt < 0.00)                return vec3(0.227, 0.478, 0.259); // taiga
  if (bm > 0.05)                return vec3(0.176, 0.549, 0.227); // rainforest
  return vec3(0.478, 0.722, 0.282); // grassland
}

vec3 heightColor(float hN, float seaLvl) {
  // Ocean: deep navy → mid blue → shallow cyan-blue
  if (hN < seaLvl) {
    float t = hN / max(0.001, seaLvl); // 0 = deepest, 1 = shore
    vec3 deep    = vec3(0.031, 0.114, 0.310);
    vec3 mid     = vec3(0.063, 0.278, 0.529);
    vec3 shallow = vec3(0.173, 0.490, 0.663);
    if (t < 0.5) return mix(deep, mid, t * 2.0);
    return mix(mid, shallow, (t - 0.5) * 2.0);
  }

  // Beach strip
  float beachTop = seaLvl + 0.035;
  if (hN < beachTop) {
    float t = (hN - seaLvl) / 0.035;
    return mix(vec3(0.831, 0.776, 0.529), vec3(0.780, 0.718, 0.459), t);
  }

  float landH = (hN - beachTop) / max(0.001, 1.0 - beachTop); // 0 = coast, 1 = peak

  // Colour stops for geographic height map
  // 0.00 → lowland green
  // 0.20 → mid green
  // 0.40 → olive / yellow-green
  // 0.55 → light brown
  // 0.70 → medium brown
  // 0.82 → dark brown / rocky
  // 0.90 → grey rock
  // 1.00 → snow white
  const vec3 c0 = vec3(0.180, 0.502, 0.118); // lowland green
  const vec3 c1 = vec3(0.251, 0.588, 0.157); // mid green
  const vec3 c2 = vec3(0.600, 0.659, 0.220); // olive-yellow
  const vec3 c3 = vec3(0.702, 0.529, 0.243); // light brown
  const vec3 c4 = vec3(0.588, 0.388, 0.157); // medium brown
  const vec3 c5 = vec3(0.451, 0.314, 0.157); // dark brown
  const vec3 c6 = vec3(0.502, 0.478, 0.447); // grey rock
  const vec3 c7 = vec3(1.000, 1.000, 1.000); // snow

  if (landH < 0.20) return mix(c0, c1, landH / 0.20);
  if (landH < 0.40) return mix(c1, c2, (landH - 0.20) / 0.20);
  if (landH < 0.55) return mix(c2, c3, (landH - 0.40) / 0.15);
  if (landH < 0.70) return mix(c3, c4, (landH - 0.55) / 0.15);
  if (landH < 0.82) return mix(c4, c5, (landH - 0.70) / 0.12);
  if (landH < 0.90) return mix(c5, c6, (landH - 0.82) / 0.08);
  return mix(c6, c7, (landH - 0.90) / 0.10);
}

void main() {
  // Reconstruct unit-sphere direction from equirectangular UV
  // (matches Three.js SphereGeometry vertex convention exactly)
  float phi   = vUv.y * 3.14159265359;
  float theta = vUv.x * 6.28318530718;
  float sinPhi = sin(phi), cosPhi = cos(phi);
  vec3 dir = vec3(-sinPhi * cos(theta), cosPhi, sinPhi * sin(theta));

  float continent = fbm(dir, uOffContinent, uContinentScale, uContinentOctaves, 0.50, 2.0);
  float detail    = fbm(dir, uOffDetail,    uDetailScale,    4,                  0.45, 2.1) * uDetailStrength;
  float h         = continent + detail;

  // ---------------------------------------------------------------------------
  // Latitude-based climate  (geographic latitude: 0 = equator, PI/2 = poles)
  // ---------------------------------------------------------------------------
  // abs(cosPhi) = sin(geographic latitude), goes 0→equator, 1→poles
  float sinLat = abs(cosPhi);

  // Temperature: linear decline equator (+1) → poles (-1)
  float baseTemp = 1.0 - 2.0 * sinLat;

  // Moisture: cos(6·lat_rad) reproduces the wind-belt pattern —
  //   0°  (ITCZ)        → +1  max  (equatorial rainforest)
  //   30° (subtropical) → −1  min  (trade-wind deserts)
  //   60° (subpolar)    → +1  2nd max  (temperate forests)
  //   90° (polar)       → −1  min  (polar desert)
  float lat = asin(sinLat); // geographic latitude in radians [0, PI/2]
  float baseMoist = cos(6.0 * lat);

  // Noise perturbation on top of the base patterns
  float noiseTemp  = fbm(dir, uOffTemp,     uBiomeScale,       3, 0.55, 2.0);
  float noiseMoist = fbm(dir, uOffMoisture, uBiomeScale * 0.8, 3, 0.50, 2.0);

  // Additive slider offsets applied last
  float temp     = clamp(baseTemp  + noiseTemp  * 0.35 + uTempOffset,     -1.0, 1.0);
  float moisture = clamp(baseMoist + noiseMoist * 0.35 + uMoistureOffset, -1.0, 1.0);

  // Normalise h to [0,1] using CPU-calibrated range
  float range = max(0.0001, uHMax - uHMin);
  float hN    = clamp((h - uHMin) / range, 0.0, 1.0);

  vec3 col = uTopoMap != 0
    ? heightColor(hN, uSeaLevel)
    : biomeColor(hN, temp, moisture, uSeaLevel, dir, uOffTemp, uOffMoisture);

  // Modulate colour by height so terrain relief is visible in the texture.
  // Land: remap hN from [seaLevel,1] → [0.6,1.0] so highest peaks are brightest.
  // Ocean: remap from [0,seaLevel] → [0.5,0.85] for subtle depth variation.
  if (uMode == 0) {
    float heightMod;
    if (hN >= uSeaLevel) {
      heightMod = 0.60 + 0.40 * (hN - uSeaLevel) / max(0.001, 1.0 - uSeaLevel);
    } else {
      heightMod = 0.50 + 0.35 * (hN / max(0.001, uSeaLevel));
    }
    col *= heightMod;
  }

  // uMode 0 = colour output, 1 = grayscale hN for displacement map
  if (uMode == 1) {
    gl_FragColor = vec4(hN, hN, hN, 1.0);
  } else {
    gl_FragColor = vec4(col, 1.0);
  }
}
`;

// ---------------------------------------------------------------------------
// Planet class
// ---------------------------------------------------------------------------

export class Planet {
  params: PlanetParams;

  /** Three.js group containing terrain mesh + ocean mesh. Add this to the scene. */
  group!: THREE.Group;
  mesh!: THREE.Mesh;
  oceanMesh!: THREE.Mesh;

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
  private _bakeQuad!:   THREE.Mesh;
  private _bakeMat!:    THREE.ShaderMaterial;
  private _bakeTarget!: THREE.WebGLRenderTarget; // colour
  private _dispTarget!: THREE.WebGLRenderTarget; // grayscale hN for displacementMap

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
    this._noise       = new SimplexNoise(this.params.seed.toString());
    this._offContinent = this._layerOffset(0);
    this._offDetail    = this._layerOffset(1);
    this._offTemp      = this._layerOffset(2);
    this._offMoisture  = this._layerOffset(3);
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
        uContinentScale:   { value: this.params.continentScale },
        uContinentOctaves: { value: this.params.continentOctaves },
        uBiomeScale:       { value: this.params.biomeScale },
        uDetailScale:      { value: this.params.detailScale },
        uDetailStrength:   { value: this.params.detailStrength },
        uTopoMap:          { value: this.params.topoMap ? 1 : 0 },
        uMode:             { value: 0 },
        uTempOffset:       { value: this.params.tempOffset },
        uMoistureOffset:   { value: this.params.moistureOffset },
        uOffContinent:     { value: this._layerOffset(0) },
        uOffDetail:        { value: this._layerOffset(1) },
        uOffTemp:          { value: this._layerOffset(2) },
        uOffMoisture:      { value: this._layerOffset(3) },
      },
    });

    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._bakeMat);
    this._bakeScene.add(quad);
    this._bakeQuad = quad;
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
    u.uContinentScale.value   = p.continentScale;
    u.uContinentOctaves.value = p.continentOctaves;
    u.uBiomeScale.value       = p.biomeScale;
    u.uDetailScale.value      = p.detailScale;
    u.uDetailStrength.value   = p.detailStrength;
    u.uTopoMap.value           = p.topoMap ? 1 : 0;
    u.uTempOffset.value       = p.tempOffset;
    u.uMoistureOffset.value   = p.moistureOffset;
    u.uOffContinent.value     = this._layerOffset(0);
    u.uOffDetail.value        = this._layerOffset(1);
    u.uOffTemp.value          = this._layerOffset(2);
    u.uOffMoisture.value      = this._layerOffset(3);

    // Pass 1: colour
    u.uMode.value = 0;
    this._renderer.setRenderTarget(this._bakeTarget);
    this._renderer.render(this._bakeScene, this._bakeCam);

    // Pass 2: grayscale hN for displacementMap
    u.uMode.value = 1;
    this._renderer.setRenderTarget(this._dispTarget);
    this._renderer.render(this._bakeScene, this._bakeCam);

    this._renderer.setRenderTarget(null);

    return { colorTex: this._bakeTarget.texture, dispTex: this._dispTarget.texture };
  }

  // -------------------------------------------------------------------------
  // Ocean sphere
  // -------------------------------------------------------------------------

  private _buildOcean(): void {
    const { radius, segments, seaLevel, elevation } = this.params;
    // Sits at exactly the height where hN = seaLevel in _displaceGeometry
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
  }

  // -------------------------------------------------------------------------
  // Cloud layer
  // -------------------------------------------------------------------------

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

    if (!this.group) {
      this.group = new THREE.Group();
      this.group.add(this.mesh);
      this.group.add(this.oceanMesh);
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
