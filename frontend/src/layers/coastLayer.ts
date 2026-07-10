import * as THREE from 'three';
import type { CoastBuffers } from '../types';

export function coastLayer(
  buffers: CoastBuffers,
  uMorph: { value: number },
): THREE.LineSegments {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(buffers.pos, 3));
  geo.setAttribute('position2', new THREE.BufferAttribute(buffers.pos2, 3));
  const line = new THREE.LineSegments(
    geo,
    new THREE.ShaderMaterial({
      uniforms: { uMorph },
      vertexShader: /* glsl */ `
        attribute vec3 position2; uniform float uMorph;
        void main() {
          gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
        }`,
      fragmentShader: `void main() { gl_FragColor = vec4(1.0, 0.66, 0.40, 0.9); }`,
      transparent: true,
    }),
  );
  line.frustumCulled = false;
  line.renderOrder = 3;
  return line;
}
