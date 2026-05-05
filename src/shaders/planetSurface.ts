// ---------------------------------------------------------------------------
// GLSL shaders for GPU texture baking
// ---------------------------------------------------------------------------

export const BAKE_VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  // PlaneGeometry(2,2) already spans clip space — skip all matrix math
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const BAKE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform float uHMin;
uniform float uHMax;
uniform float uSeaLevel;
uniform float uShoreWidth; // height-units width of beach strip above sea level
uniform float uContinentScale;
uniform int   uContinentOctaves;
uniform float uBiomeScale;
uniform float uDetailScale;
uniform float uDetailStrength;
uniform int   uTopoMap;
uniform int   uMoistureView;
uniform int   uTempView;
uniform int   uFancyOcean; // 1 = transparent water sphere, 0 = show seafloor
uniform int   uMode; // 0 = colour, 1 = displacement grayscale
// Climate additive offsets (sliders)
uniform float uTempOffset;
uniform float uMoistureOffset;

// Per-layer seed offsets (computed from seed on CPU, applied to noise input)
uniform vec3 uOffContinent;
uniform vec3 uOffDetail;
uniform vec3 uOffTemp;
uniform vec3 uOffMoisture;
uniform vec3 uOffMoistureMap;

// Large-scale moisture map
uniform float uMoistureMapScale;
uniform float uMoistureMapStrength;

// Custom height map — when uUseCustomMap=1 the texture replaces noise-derived hN
// for both the colour and displacement outputs.
uniform int       uUseCustomMap;
uniform sampler2D uCustomHeightTex;

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
// Conventional moisture-map palette
// Dry (0.0) → amber → yellow → yellow-green → green → dark green → deep forest (1.0)
// Conventional temperature palette  (temp in [-1, 1])
// -1 = polar ice → cold blue → cool cyan → mild green → warm yellow → hot orange → +1 scorching red
vec3 tempColor(float t) {
  float n = t * 0.5 + 0.5; // remap [-1,1] → [0,1]
  const vec3 c0 = vec3(0.878, 0.957, 1.000); // polar ice   near-white blue
  const vec3 c1 = vec3(0.196, 0.525, 0.878); // arctic      deep blue
  const vec3 c2 = vec3(0.149, 0.773, 0.851); // cold        cyan
  const vec3 c3 = vec3(0.310, 0.776, 0.388); // cool        green
  const vec3 c4 = vec3(0.949, 0.929, 0.212); // mild        yellow
  const vec3 c5 = vec3(0.957, 0.576, 0.102); // warm        orange
  const vec3 c6 = vec3(0.820, 0.102, 0.063); // hot         red
  if (n < 0.167) return mix(c0, c1, n / 0.167);
  if (n < 0.333) return mix(c1, c2, (n - 0.167) / 0.167);
  if (n < 0.500) return mix(c2, c3, (n - 0.333) / 0.167);
  if (n < 0.667) return mix(c3, c4, (n - 0.500) / 0.167);
  if (n < 0.833) return mix(c4, c5, (n - 0.667) / 0.167);
  return mix(c5, c6, (n - 0.833) / 0.167);
}

vec3 moistureColor(float m) {
  const vec3 c0 = vec3(0.710, 0.569, 0.200); // hyper-arid  amber
  const vec3 c1 = vec3(0.820, 0.749, 0.176); // arid        yellow
  const vec3 c2 = vec3(0.659, 0.753, 0.227); // semi-arid   yellow-green
  const vec3 c3 = vec3(0.353, 0.620, 0.251); // sub-humid   medium green
  const vec3 c4 = vec3(0.176, 0.478, 0.188); // humid       dark green
  const vec3 c5 = vec3(0.063, 0.306, 0.094); // very wet    deep forest
  const vec3 c6 = vec3(0.016, 0.173, 0.047); // rainforest  near-black forest
  if (m < 0.167) return mix(c0, c1, m / 0.167);
  if (m < 0.333) return mix(c1, c2, (m - 0.167) / 0.167);
  if (m < 0.500) return mix(c2, c3, (m - 0.333) / 0.167);
  if (m < 0.667) return mix(c3, c4, (m - 0.500) / 0.167);
  if (m < 0.833) return mix(c4, c5, (m - 0.667) / 0.167);
  return mix(c5, c6, (m - 0.833) / 0.167);
}

vec3 seafloorColor(float hN, float seaLvl) {
  float t = clamp(hN / max(0.001, seaLvl), 0.0, 1.0); // 0 = deepest, 1 = shore
  vec3 deep    = vec3(0.020, 0.071, 0.200); // dark deep ocean
  vec3 shallow = vec3(0.063, 0.278, 0.529); // lighter near shore
  return mix(deep, shallow, t);
}

vec3 biomeColor(float hN, float temp, float moisture, float seaLvl, vec3 dir, vec3 offT, vec3 offM) {
  if (hN < seaLvl) {
    float t = (1.0 - hN / seaLvl) * 0.75;
    return mix(vec3(0.784, 0.706, 0.447), vec3(0.416, 0.353, 0.188), t);
  }
  float beachTop = seaLvl + uShoreWidth;
  if (hN < beachTop) return vec3(0.831, 0.737, 0.471);

  float landH = (hN - beachTop) / max(0.01, 1.0 - beachTop);

  // High-frequency warp to break up biome boundary regularity
  float warpT = snoise((dir + offT * 0.013) * 11.0) * 0.13
              + snoise((dir + offT * 0.007) * 23.0) * 0.05;
  float warpM = snoise((dir + offM * 0.013) * 10.0) * 0.13
              + snoise((dir + offM * 0.007) * 21.0) * 0.05;
  float bt = clamp(temp     + warpT, -1.0, 1.0);
  float bm = clamp(moisture + warpM,  0.0, 1.0);

  // Elevation lapse rate: cubic curve keeps low-elevation areas unaffected
  // while aggressively cooling only true peaks (landH^3 is ~0 at foothills,
  // surges to 3.0 at the summit — driving any peak past the snow threshold).
  bt = clamp(bt - pow(landH, 3.0) * 3.0, -1.0, 1.0);

  // Elevation moisture decline: linear reduction to 20 % of original at landH=1.
  // Models orographic drying — high peaks are significantly more arid.
  bm = clamp(bm * (1.0 - 0.6 * landH), 0.0, 1.0);

  if (bt < -0.65) {
    float t = min(1.0, (-bt - 0.55) / 0.20);
    return mix(vec3(0.722, 0.831, 0.784), vec3(1.0), t); // tundra → ice/snow
  }
  if (bt < -0.40) return vec3(0.573, 0.533, 0.380); // tundra
  if (bt > 0.30 && bm < 0.40) return vec3(0.784, 0.627, 0.314); // desert
  if (bt > 0.20 && bm > 0.625) return vec3(0.102, 0.420, 0.102); // tropical
  if (bt < 0.00)               return vec3(0.227, 0.478, 0.259); // taiga
  if (bm > 0.525)              return vec3(0.176, 0.549, 0.227); // rainforest
  return vec3(0.478, 0.722, 0.282); // grassland
}

vec3 heightColor(float hN, float seaLvl) {
  if (hN < seaLvl) {
    float t = clamp(hN / max(0.001, seaLvl), 0.0, 1.0);
    vec3 deep    = vec3(0.020, 0.071, 0.200);
    vec3 shallow = vec3(0.063, 0.278, 0.529);
    return mix(deep, shallow, t);
  }

  // No shore strip in topo/height mode — land colour starts immediately at sea level.
  float beachTop = seaLvl;
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

  // Reduce detail noise underwater: use continent-only height as a proxy for depth
  // so high-frequency bumps don't roughen the seafloor / ocean surface.
  float range0      = max(0.0001, uHMax - uHMin);
  float hContApprox = clamp((continent - uHMin) / range0, 0.0, 1.0);
  float detailFactor = smoothstep(uSeaLevel - 0.05, uSeaLevel + 0.05, hContApprox);
  float h = continent + detail * mix(0.5, 1.0, detailFactor);

  // ---------------------------------------------------------------------------
  // Latitude-based climate  (geographic latitude: 0 = equator, PI/2 = poles)
  // ---------------------------------------------------------------------------
  // abs(cosPhi) = sin(geographic latitude), goes 0→equator, 1→poles
  float sinLat = abs(cosPhi);

  float lat = asin(sinLat); // geographic latitude in radians [0, PI/2]

  // Temperature: linear decline equator (+1) → poles (-1), crossing 0 at 45°.
  // lat / (PI/4) maps [0, PI/2] → [0, 2], so baseTemp = 1 - that.
  float baseTemp = clamp(1.0 - lat / (3.14159265359 * 0.25), -1.0, 1.0);

  // Moisture [0,1]: cos(6·lat_rad) gives wind-belt pattern remapped to [0,1].
  //   0°  (ITCZ)        → 1.0  max  (equatorial rainforest)
  //   30° (subtropical) → 0.0  min  (trade-wind deserts)
  //   60° (subpolar)    → 1.0  2nd max  (temperate forests)
  //   90° (polar)       → 0.0  min  (polar desert)
  float baseMoist = cos(6.0 * lat) * 0.5 + 0.5;

  // Continent-scale moisture map: very low-frequency noise that creates
  // large wet/dry zones (rainforest belts, desert continents) independent
  // of latitude. Output is [-1,1]; positive = wetter, negative = drier.
  float moistureMap = fbm(dir, uOffMoistureMap, uMoistureMapScale, 4, 0.50, 2.0);
  baseMoist = clamp(baseMoist + moistureMap * uMoistureMapStrength, 0.0, 1.0);

  // Noise perturbation on top of the base patterns
  float noiseTemp  = fbm(dir, uOffTemp,     uBiomeScale,       3, 0.55, 2.0);
  float noiseMoist = fbm(dir, uOffMoisture, uBiomeScale * 0.8, 3, 0.50, 2.0);

  // Additive slider offsets applied last
  float temp     = clamp(baseTemp  + noiseTemp  * 0.35  + uTempOffset,     -1.0, 1.0);
  float moisture = clamp(baseMoist + noiseMoist * 0.175 + uMoistureOffset,  0.0, 1.0);

  // Normalise h to [0,1] using CPU-calibrated range
  float range = max(0.0001, uHMax - uHMin);
  float hN    = clamp((h - uHMin) / range, 0.0, 1.0);

  // Custom height map: override hN with the raw texel value [0,1].
  // Both the colour/biome pass and the displacement pass use this same hN,
  // so they stay perfectly aligned.
  if (uUseCustomMap != 0) {
    hN = texture2D(uCustomHeightTex, vUv).r;
  }

  // When fancy ocean is off, clamp displacement to seaLevel so underwater
  // terrain is raised flush with the waterline (no underwater holes).
  float hNDisp = (uFancyOcean == 0) ? max(hN, uSeaLevel) : hN;

  vec3 col;
  if (uTempView != 0) {
    // Flat ocean, temperature palette on land
    if (hN < uSeaLevel) {
      col = vec3(0.118, 0.306, 0.651); // flat ocean blue
    } else {
      col = tempColor(temp);
    }
  } else if (uMoistureView != 0) {
    // Flat ocean, moisture palette on land — no height modulation so values read cleanly
    if (hN < uSeaLevel) {
      col = vec3(0.118, 0.306, 0.651); // flat ocean blue
    } else {
      col = moistureColor(moisture);
    }
  } else if (uTopoMap != 0) {
    col = heightColor(hN, uSeaLevel);
  } else if (uFancyOcean == 0 && hN < uSeaLevel) {
    col = seafloorColor(hN, uSeaLevel);
  } else {
    col = biomeColor(hN, temp, moisture, uSeaLevel, dir, uOffTemp, uOffMoisture);
  }

  // Modulate colour by height so terrain relief is visible in the texture.
  // Skipped for moisture view (flat values communicate information more clearly).
  // Land: remap hN from [seaLevel,1] → [0.6,1.0] so highest peaks are brightest.
  // Ocean: remap from [0,seaLevel] → [0.5,0.85] for subtle depth variation.
  if (uMode == 0 && uMoistureView == 0 && uTempView == 0) {
    float heightMod;
    if (hN >= uSeaLevel) {
      heightMod = 0.60 + 0.40 * (hN - uSeaLevel) / max(0.001, 1.0 - uSeaLevel);
    } else {
      heightMod = 0.50 + 0.35 * (hN / max(0.001, uSeaLevel));
    }
    col *= heightMod;
  }

  // uMode 0 = colour output, 1 = grayscale hNDisp for displacement map
  if (uMode == 1) {
    gl_FragColor = vec4(hNDisp, hNDisp, hNDisp, 1.0);
  } else {
    gl_FragColor = vec4(col, 1.0);
  }
}
`;