varying vec2 vUv;
void main() {
  vUv = uv;
  // PlaneGeometry(2,2) already spans clip space — skip all matrix math
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
