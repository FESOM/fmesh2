import * as THREE from 'three';
import { fetchHfun } from '../api';
import { VIRIDIS_GLSL } from '../colormap';

/** Equirect drape of the composed hfun field over globe and map.
 * A lat/lon grid mesh carries both sphere and plane positions and morphs
 * with the same uniform as the mesh layer; the field is a float texture
 * sampled with viridis in the fragment shader. */
export class HfunOverlay {
  readonly mesh: THREE.Mesh;
  readonly uniforms: {
    uMorph: { value: number };
    uOpacity: { value: number };
    uRange: { value: THREE.Vector2 };
    uMap: { value: THREE.DataTexture | null };
  };
  range: [number, number] = [0, 1];

  constructor(morph: { value: number }, segLat = 90, segLon = 180) {
    this.uniforms = {
      uMorph: morph,
      uOpacity: { value: 0.75 },
      uRange: { value: new THREE.Vector2(0, 1) },
      uMap: { value: null },
    };

    const nv = (segLat + 1) * (segLon + 1);
    const pos = new Float32Array(nv * 3);
    const pos2 = new Float32Array(nv * 3);
    const uv = new Float32Array(nv * 2);
    let k = 0;
    for (let j = 0; j <= segLat; j++) {
      const lat = -90 + (180 * j) / segLat;
      const latR = (lat * Math.PI) / 180;
      for (let i = 0; i <= segLon; i++) {
        const lon = -180 + (360 * i) / segLon;
        const lonR = (lon * Math.PI) / 180;
        const cl = Math.cos(latR);
        const r = 1.0015; // just above the mesh surface
        pos[3 * k] = r * cl * Math.cos(lonR);
        pos[3 * k + 1] = r * Math.sin(latR);
        pos[3 * k + 2] = -r * cl * Math.sin(lonR);
        pos2[3 * k] = lonR;
        pos2[3 * k + 1] = latR;
        pos2[3 * k + 2] = 0.002;
        uv[2 * k] = i / segLon;
        uv[2 * k + 1] = j / segLat;
        k++;
      }
    }
    const idx = new Uint32Array(segLat * segLon * 6);
    let t = 0;
    for (let j = 0; j < segLat; j++) {
      for (let i = 0; i < segLon; i++) {
        const a = j * (segLon + 1) + i;
        const b = a + segLon + 1;
        idx[t++] = a; idx[t++] = a + 1; idx[t++] = b;
        idx[t++] = a + 1; idx[t++] = b + 1; idx[t++] = b;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('position2', new THREE.BufferAttribute(pos2, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));

    const mat = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: /* glsl */ `
        attribute vec3 position2;
        uniform float uMorph;
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D uMap; uniform vec2 uRange; uniform float uOpacity;
        varying vec2 vUv;
        ${VIRIDIS_GLSL}
        void main() {
          float v = texture2D(uMap, vUv).r;
          if (v < 0.0) discard;  // sentinel for undefined cells
          float t = clamp((v - uRange.x) / max(uRange.y - uRange.x, 1e-6), 0.0, 1.0);
          gl_FragColor = vec4(viridis(t), uOpacity);
        }`,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    this.mesh.renderOrder = 2;
  }

  /** Fetch the current field and update the texture; returns the range. */
  async refresh(): Promise<[number, number]> {
    const grid = await fetchHfun();
    const clean = new Float32Array(grid.values.length);
    for (let i = 0; i < clean.length; i++) {
      const v = grid.values[i];
      clean[i] = Number.isFinite(v) ? v : -1; // sentinel: shader discards
    }
    const tex = new THREE.DataTexture(
      clean, grid.nLon, grid.nLat, THREE.RedFormat, THREE.FloatType,
    );
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    this.uniforms.uMap.value?.dispose();
    this.uniforms.uMap.value = tex;
    this.uniforms.uRange.value.set(grid.min, grid.max);
    this.range = [grid.min, grid.max];
    return this.range;
  }

  setOpacity(v: number): void {
    this.uniforms.uOpacity.value = v;
  }
}
