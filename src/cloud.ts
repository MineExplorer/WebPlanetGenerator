import * as THREE from 'three';
import type { PlanetParams } from './planet';

// ---------------------------------------------------------------------------
// Cloud shaders
// ---------------------------------------------------------------------------

const CLOUD_VERT = /* glsl */`
varying vec3 vLocalPos;
void main() {
  vLocalPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CLOUD_FRAG = /* glsl */`
precision highp float;
varying vec3 vLocalPos;

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
  vec3 dir = normalize(vLocalPos);

  float lo    = uCloudDensity - 0.10;
  float hi    = uCloudDensity + 0.32;

  // --- Domain-warp the sample point for organic puffiness (2 octaves each) ---
  float ws = uCloudScale * 0.55;
  vec3 warp = vec3(
    cfbm(dir * ws + vec3( uTime*0.013,  0.000,        0.000       ), 2),
    cfbm(dir * ws + vec3( 0.000,        uTime*0.011,  uTime*0.009 ), 2),
    cfbm(dir * ws + vec3( uTime*0.010,  uTime*0.014,  0.000       ), 2)
  );

  float cloud = cfbm(dir * uCloudScale + warp * 0.72, 4);

  // --- Coverage threshold with smooth edges ---
  float cover = smoothstep(lo, hi, cloud);

  // --- Colour: bright white core, soft blue-tinted wispy edges ---
  vec3 coreColor = vec3(1.00, 1.00, 1.00);
  vec3 edgeColor = vec3(0.75, 0.84, 0.97);
  vec3 baseColor = mix(edgeColor, coreColor, cover * cover);

  // --- Fake volumetric self-shadow: dense parts have darker bellies ---
  float shadow = 1.0 - 0.32 * smoothstep(0.45, 0.95, cover);

  // --- Limb-darkening hint: fade alpha at grazing angles for roundness ---
  vec3 viewDir = normalize(cameraPosition - vLocalPos);
  float rim    = max(0.0, dot(dir, viewDir));
  float limb   = mix(0.55, 1.0, rim);

  vec3  finalColor = baseColor * shadow;
  float alpha      = cover * cover * uCloudOpacity * limb;

  gl_FragColor = vec4(finalColor, alpha);
}
`;

// ---------------------------------------------------------------------------
// CloudLayer class
// ---------------------------------------------------------------------------

/**
 * Self-contained cloud layer.  Build once, then call `tick(dt)` every frame.
 * The `group` property should be added to the scene (or to the planet group).
 */
export class CloudLayer {
  /** Rotate this to drift clouds independently of the planet. */
  group: THREE.Group;
  mesh!: THREE.Mesh;

  private _mat!: THREE.ShaderMaterial;
  private _time = 0;

  constructor(params: PlanetParams) {
    this.group = new THREE.Group();
    this.build(params);
  }

  /** (Re)build geometry and update shader uniforms from PlanetParams. */
  build(params: PlanetParams): void {
    const { radius, cloudHeight, cloudScale, cloudDensity, cloudOpacity, segments, showClouds } = params;
    const cloudRadius = radius * (1 + cloudHeight);
    const segs = Math.min(Math.max(48, segments), 96);
    const geo  = new THREE.SphereGeometry(cloudRadius, segs, segs);

    if (!this._mat) {
      this._mat = new THREE.ShaderMaterial({
        vertexShader:   CLOUD_VERT,
        fragmentShader: CLOUD_FRAG,
        uniforms: {
          uTime:         { value: this._time },
          uCloudScale:   { value: cloudScale },
          uCloudDensity: { value: cloudDensity },
          uCloudOpacity: { value: cloudOpacity },
        },
        transparent: true,
        depthWrite:  false,
        side:        THREE.FrontSide,
      });
    } else {
      this._mat.uniforms.uCloudScale.value   = cloudScale;
      this._mat.uniforms.uCloudDensity.value = cloudDensity;
      this._mat.uniforms.uCloudOpacity.value = cloudOpacity;
    }

    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh.geometry = geo;
    } else {
      this.mesh = new THREE.Mesh(geo, this._mat);
      this.group.add(this.mesh);
    }

    this.mesh.visible = showClouds;
  }

  /** Advance cloud animation by `dt` seconds. Call once per frame. */
  tick(dt: number): void {
    this._time += dt;
    if (this._mat) {
      this._mat.uniforms.uTime.value = this._time;
    }
  }

  dispose(): void {
    this.mesh?.geometry.dispose();
    this._mat?.dispose();
  }
}
