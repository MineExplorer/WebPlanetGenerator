precision highp float;
varying vec2 vUv;

uniform float uTime;
uniform float uCloudScale;
uniform float uCloudDensity;
uniform float uCloudOpacity;
uniform float uWindPhase;

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

vec3 dirAt(float phi, float theta) {
  return vec3(-sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta));
}

// Eastward drift rate per circulation band. Constant within a band, so each
// band advects as a rigid rotation — the only transform that never distorts
// the noise field. Bands are crossfaded rather than their rates interpolated,
// which is what would reintroduce shear.
const float RATE_EASTERLY = -0.55; // trades and polar easterlies
const float RATE_WESTERLY =  1.00; // mid-latitude jet

// Domain offsets keep neighbouring bands decorrelated so a crossfade reads as
// a dissolve instead of a double image.
const vec3 SEED_TROPIC = vec3(  0.0,  0.0,  0.0);
const vec3 SEED_MIDLAT = vec3( 17.3,  5.1,  9.7);
const vec3 SEED_POLAR  = vec3(-11.2, 23.4, -6.8);

float cloudField(float phi, float theta, float rate, vec3 seed) {
  vec3 dir = dirAt(phi, theta - uWindPhase * rate);

  // Domain-warp for organic puffiness
  vec3 wp = dir * (uCloudScale * 0.55) + seed;
  vec3 warp = vec3(
    cfbm(wp + vec3( uTime*0.018,  0.000,       0.000       ), 2),
    cfbm(wp + vec3( 0.000,        uTime*0.015, uTime*0.013 ), 2),
    cfbm(wp + vec3( uTime*0.014,  uTime*0.019, 0.000       ), 2)
  );

  return cfbm(dir * uCloudScale + seed + warp * 0.72, 4);
}

void main() {
  float phi   = vUv.y * 3.14159265359;
  float theta = vUv.x * 6.28318530718;
  float lat   = degrees(asin(abs(cos(phi))));

  // Pick the two bands straddling this latitude and how far between them we are.
  bool  upper = lat >= 45.0;
  float rateA = upper ? RATE_WESTERLY : RATE_EASTERLY;
  float rateB = upper ? RATE_EASTERLY : RATE_WESTERLY;
  vec3  seedA = upper ? SEED_MIDLAT   : SEED_TROPIC;
  vec3  seedB = upper ? SEED_POLAR    : SEED_MIDLAT;
  float k     = upper ? smoothstep(55., 65., lat) : smoothstep(25., 35., lat);

  // No pole fade needed: a rigid Y-rotation has the pole as a fixed point.
  float cloud = cloudField(phi, theta, rateA, seedA);
  if (k > 0.0) {
    cloud = mix(cloud, cloudField(phi, theta, rateB, seedB), k);
  }

  float lo = uCloudDensity - 0.10;
  float hi = uCloudDensity + 0.32;
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
