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
  /** Highlighted nodes of the edit selection; visible only in edit mode. */
  readonly selPoints: THREE.Points;
  private readonly posAttr: THREE.BufferAttribute;
  private readonly pos2Attr: THREE.BufferAttribute;
  private readonly selAttr: THREE.BufferAttribute;
  private readonly depAttr: THREE.BufferAttribute;
  readonly uniforms: {
    uMorph: { value: number };
    uField: { value: number };
    uRange: { value: THREE.Vector2 };
  };
  field: ScalarField = 'res';

  constructor(
    readonly meta: MeshMeta,
    buffers: MeshBuffers,
    morph: { value: number } = { value: 0 },
  ) {
    const posAttr = new THREE.BufferAttribute(buffers.pos, 3);
    const pos2Attr = new THREE.BufferAttribute(buffers.pos2, 3);
    const selAttr = new THREE.BufferAttribute(new Float32Array(meta.verts), 1);
    this.posAttr = posAttr;
    this.pos2Attr = pos2Attr;
    this.selAttr = selAttr;

    // `morph` may be shared with coast/overlay/draw layers so an A/B mesh
    // swap keeps every layer morphing in lockstep
    this.uniforms = {
      uMorph: morph,
      uField: { value: 0 },
      uRange: { value: new THREE.Vector2(meta.res_range[0], meta.res_range[1]) },
    };

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', posAttr);
    geo.setAttribute('position2', pos2Attr);
    geo.setAttribute('res', new THREE.BufferAttribute(buffers.res, 1));
    this.depAttr = new THREE.BufferAttribute(buffers.dep, 1);
    geo.setAttribute('dep', this.depAttr);
    geo.setAttribute('sel', selAttr);
    geo.setIndex(new THREE.BufferAttribute(buffers.tri, 1));

    const fillMat = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: /* glsl */ `
        attribute vec3 position2; attribute float res; attribute float dep; attribute float sel;
        uniform float uMorph; uniform int uField; uniform vec2 uRange;
        varying float vT; varying vec3 vN; varying float vSel;
        void main() {
          float s = uField == 0 ? res : dep;
          vT = clamp((s - uRange.x) / (uRange.y - uRange.x), 0.0, 1.0);
          vN = position;  // unit sphere: model-space normal == position
          vSel = sel;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
        }`,
      fragmentShader: /* glsl */ `
        varying float vT; varying vec3 vN; varying float vSel;
        uniform float uMorph;
        ${VIRIDIS_GLSL}
        void main() {
          float sphereShade = 0.72 + 0.28 * max(dot(normalize(vN), normalize(cameraPosition)), 0.0);
          vec3 col = viridis(vT) * mix(sphereShade, 1.0, uMorph);
          col = mix(col, vec3(1.0, 0.55, 0.15), 0.55 * clamp(vSel, 0.0, 1.0));
          gl_FragColor = vec4(col, 1.0);
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

    const selGeo = new THREE.BufferGeometry();
    selGeo.setAttribute('position', posAttr);
    selGeo.setAttribute('position2', pos2Attr);
    selGeo.setAttribute('sel', selAttr);
    this.selPoints = new THREE.Points(
      selGeo,
      new THREE.ShaderMaterial({
        uniforms: { uMorph: this.uniforms.uMorph },
        vertexShader: /* glsl */ `
          attribute vec3 position2; attribute float sel;
          uniform float uMorph;
          varying float vSel;
          void main() {
            vSel = sel;
            gl_PointSize = 7.0;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
          }`,
        fragmentShader: /* glsl */ `
          varying float vSel;
          void main() {
            if (vSel < 0.5 || length(gl_PointCoord - 0.5) > 0.5) discard;
            gl_FragColor = vec4(1.0, 0.62, 0.2, 1.0);
          }`,
        transparent: true,
      }),
    );
    this.selPoints.frustumCulled = false;
    this.selPoints.visible = false;

    this.group.add(this.fill, this.wire, this.points, this.selPoints);
  }

  /** Set the selection flag of the given vertex indices (partial upload). */
  setSelectionFlags(verts: number[], value: 0 | 1): void {
    const arr = this.selAttr.array as Float32Array;
    for (const v of verts) arr[v] = value;
    for (const v of verts) this.selAttr.addUpdateRange(v, 1);
    this.selAttr.needsUpdate = true;
  }

  clearSelectionFlags(): void {
    (this.selAttr.array as Float32Array).fill(0);
    this.selAttr.clearUpdateRanges();
    this.selAttr.needsUpdate = true;
  }

  /** Write new positions for vertex indices (both projections) and upload
   * only the touched ranges — the live path for drags and edit commits.
   * The attributes are shared with the wire/points/picking geometries, so
   * every view updates in lockstep. */
  updatePositions(
    verts: number[],
    pos: ArrayLike<number>[],
    pos2: ArrayLike<number>[],
  ): void {
    const p = this.posAttr.array as Float32Array;
    const q = this.pos2Attr.array as Float32Array;
    for (let i = 0; i < verts.length; i++) {
      const o = verts[i] * 3;
      p[o] = pos[i][0];
      p[o + 1] = pos[i][1];
      p[o + 2] = pos[i][2];
      q[o] = pos2[i][0];
      q[o + 1] = pos2[i][1];
      q[o + 2] = pos2[i][2];
      this.posAttr.addUpdateRange(o, 3);
      this.pos2Attr.addUpdateRange(o, 3);
    }
    this.posAttr.needsUpdate = true;
    this.pos2Attr.needsUpdate = true;
  }

  /** Write topo-resampled depths for vertex indices (edit commits). */
  updateDepths(verts: number[], dep: ArrayLike<number>): void {
    const d = this.depAttr.array as Float32Array;
    for (let i = 0; i < verts.length; i++) {
      d[verts[i]] = dep[i];
      this.depAttr.addUpdateRange(verts[i], 1);
    }
    this.depAttr.needsUpdate = true;
  }

  setField(field: ScalarField): void {
    this.field = field;
    this.uniforms.uField.value = field === 'res' ? 0 : 1;
    const range = field === 'res' ? this.meta.res_range : this.meta.dep_range;
    this.uniforms.uRange.value.set(range[0], range[1]);
  }

  /** Free GPU resources (geometries/materials) after an A/B swap. */
  dispose(): void {
    for (const obj of [this.fill, this.wire, this.points, this.selPoints]) {
      obj.geometry.dispose();
      (obj.material as THREE.Material).dispose();
    }
  }
}
