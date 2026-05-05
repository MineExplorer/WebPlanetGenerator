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
