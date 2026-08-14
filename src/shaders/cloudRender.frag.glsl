precision mediump float;
varying vec3 vLocalPos;

uniform sampler2D uCloudTex;
uniform sampler2D uTerrainHeightTex;
uniform float uPlanetRadius;
uniform float uTerrainElevation;
uniform float uCloudHeight;
uniform float uCloudRotationY;

void main() {
  // Equirectangular UV from local sphere direction
  vec3 d  = normalize(vLocalPos);
  float u = atan(d.z, -d.x) / 6.28318530718 + 0.5;
  float v = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265359;
  vec4 t  = texture2D(uCloudTex, vec2(u, v));

  float c = cos(uCloudRotationY);
  float s = sin(uCloudRotationY);
  vec3 terrainDir = vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
  float terrainU = fract(atan(terrainDir.z, -terrainDir.x) / 6.28318530718);
  float terrainV = 1.0 - acos(clamp(terrainDir.y, -1.0, 1.0)) / 3.14159265359;
  float terrainHeight = texture2D(uTerrainHeightTex, vec2(terrainU, terrainV)).r;
  float terrainRadius = uPlanetRadius + uTerrainElevation * (terrainHeight - 0.5);
  float cloudRadius = uPlanetRadius * (1.0 + uCloudHeight);
  float terrainFade = max(0.001, min(uPlanetRadius * 0.01, uPlanetRadius * uCloudHeight * 0.5));
  float terrainMask = 1.0 - smoothstep(cloudRadius - terrainFade, cloudRadius, terrainRadius);

  // Limb-darkening: fade alpha at grazing angles for a rounded appearance
  vec3 viewDir = normalize(cameraPosition - vLocalPos);
  float rim    = max(0.0, dot(normalize(vLocalPos), viewDir));
  float limb   = mix(0.55, 1.0, rim);

  gl_FragColor = vec4(t.rgb, t.a * limb * terrainMask);
}
