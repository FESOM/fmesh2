import * as THREE from 'three';
import { VIRIDIS_GLSL } from '../colormap';
import type { MeshBuffers, MeshMeta } from '../types';

export type ScalarField = 'res' | 'dep';

const MORPH_VERT = /* glsl */ `
  attribute vec3 position2;
  uniform float uMorph;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
  }`;

export class MeshLayer {
  readonly group = new THREE.Group();
  readonly fill: THREE.Mesh;
  readonly wire: THREE.LineSegments;
  readonly points: THREE.Points;
  readonly uniforms: {
    uMorph: { value: number };
    uField: { value: number };
    uRange: { value: THREE.Vector2 };
  };
  field: ScalarField = 'res';

  constructor(readonly meta: MeshMeta, buffers: MeshBuffers) {
    const posAttr = new THREE.BufferAttribute(buffers.pos, 3);
    const pos2Attr = new THREE.BufferAttribute(buffers.pos2, 3);

    this.uniforms = {
      uMorph: { value: 0 },
      uField: { value: 0 },
      uRange: { value: new THREE.Vector2(meta.res_range[0], meta.res_range[1]) },
    };

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', posAttr);
    geo.setAttribute('position2', pos2Attr);
    geo.setAttribute('res', new THREE.BufferAttribute(buffers.res, 1));
    geo.setAttribute('dep', new THREE.BufferAttribute(buffers.dep, 1));
    geo.setIndex(new THREE.BufferAttribute(buffers.tri, 1));

    const fillMat = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: /* glsl */ `
        attribute vec3 position2; attribute float res; attribute float dep;
        uniform float uMorph; uniform int uField; uniform vec2 uRange;
        varying float vT; varying vec3 vN;
        void main() {
          float s = uField == 0 ? res : dep;
          vT = clamp((s - uRange.x) / (uRange.y - uRange.x), 0.0, 1.0);
          vN = position;  // unit sphere: model-space normal == position
          gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
        }`,
      fragmentShader: /* glsl */ `
        varying float vT; varying vec3 vN;
        uniform float uMorph;
        ${VIRIDIS_GLSL}
        void main() {
          float sphereShade = 0.72 + 0.28 * max(dot(normalize(vN), normalize(cameraPosition)), 0.0);
          gl_FragColor = vec4(viridis(vT) * mix(sphereShade, 1.0, uMorph), 1.0);
        }`,
      side: THREE.FrontSide,
      polygonOffset: true,
      polygonOffsetFactor: 1.5,
      polygonOffsetUnits: 1.5,
    });
    this.fill = new THREE.Mesh(geo, fillMat);
    this.fill.frustumCulled = false;

    const edgeGeo = new THREE.BufferGeometry();
    edgeGeo.setAttribute('position', posAttr);
    edgeGeo.setAttribute('position2', pos2Attr);
    edgeGeo.setIndex(new THREE.BufferAttribute(buffers.edge, 1));
    this.wire = new THREE.LineSegments(
      edgeGeo,
      new THREE.ShaderMaterial({
        uniforms: { uMorph: this.uniforms.uMorph },
        vertexShader: MORPH_VERT,
        fragmentShader: `void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 0.42); }`,
        transparent: true,
      }),
    );
    this.wire.frustumCulled = false;
    this.wire.visible = false;

    const ptsGeo = new THREE.BufferGeometry();
    ptsGeo.setAttribute('position', posAttr);
    ptsGeo.setAttribute('position2', pos2Attr);
    this.points = new THREE.Points(
      ptsGeo,
      new THREE.ShaderMaterial({
        uniforms: { uMorph: this.uniforms.uMorph },
        vertexShader: MORPH_VERT.replace(
          'gl_Position =',
          'gl_PointSize = 2.0; gl_Position =',
        ),
        fragmentShader: `void main() { gl_FragColor = vec4(1.0, 1.0, 1.0, 0.8); }`,
        transparent: true,
      }),
    );
    this.points.frustumCulled = false;
    this.points.visible = false;

    this.group.add(this.fill, this.wire, this.points);
  }

  setField(field: ScalarField): void {
    this.field = field;
    this.uniforms.uField.value = field === 'res' ? 0 : 1;
    const range = field === 'res' ? this.meta.res_range : this.meta.dep_range;
    this.uniforms.uRange.value.set(range[0], range[1]);
  }
}
