import * as THREE from 'three';
import {
  closeThresholdKm,
  lonLatToPlane,
  lonLatToSphere,
  planeToLonLat,
  shouldClosePolygon,
  sphereToLonLat,
  type LonLat,
} from '../geo';

const LINE_VERT = /* glsl */ `
  attribute vec3 position2;
  uniform float uMorph;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
  }`;

export interface DrawContext {
  canvas: HTMLElement;
  camera: THREE.PerspectiveCamera;
  scene: THREE.Scene;
  morph: { value: number };
  getMode: () => 0 | 1;
  onFinish: (vertices: LonLat[]) => Promise<void>;
  onState?: (active: boolean, nVerts: number) => void;
}

/** Click-to-place polygon drawing on globe or map, with draggable handles.
 * Click near the first vertex (or press Enter) to close; Backspace removes
 * the last vertex; Escape cancels. */
export class DrawMode {
  active = false;
  private verts: LonLat[] = [];
  private readonly group = new THREE.Group();
  private line: THREE.Line;
  private handles: THREE.Points;
  private readonly raycaster = new THREE.Raycaster();
  private readonly sphere = new THREE.Sphere(new THREE.Vector3(), 1);
  private readonly plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  private dragIdx = -1;
  private dragged = false;

  constructor(private readonly ctx: DrawContext) {
    const uniforms = { uMorph: ctx.morph };
    this.line = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.ShaderMaterial({
        uniforms,
        vertexShader: LINE_VERT,
        fragmentShader: `void main() { gl_FragColor = vec4(1.0, 0.62, 0.25, 0.95); }`,
        transparent: true,
        depthTest: false,
      }),
    );
    this.handles = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.ShaderMaterial({
        uniforms,
        vertexShader: LINE_VERT.replace('gl_Position =', 'gl_PointSize = 9.0; gl_Position ='),
        fragmentShader: `void main() {
          if (length(gl_PointCoord - 0.5) > 0.5) discard;
          gl_FragColor = vec4(1.0, 0.78, 0.35, 1.0);
        }`,
        transparent: true,
        depthTest: false,
      }),
    );
    this.line.frustumCulled = false;
    this.handles.frustumCulled = false;
    this.line.renderOrder = 5;
    this.handles.renderOrder = 6;
    this.group.add(this.line, this.handles);
    this.group.visible = false;
    ctx.scene.add(this.group);

    ctx.canvas.addEventListener('pointerdown', this.onPointerDown, true);
    ctx.canvas.addEventListener('pointermove', this.onPointerMove, true);
    ctx.canvas.addEventListener('pointerup', this.onPointerUp, true);
    ctx.canvas.addEventListener('click', this.onClick, true);
    addEventListener('keydown', this.onKey);
  }

  toggle(): boolean {
    this.active = !this.active;
    if (!this.active) this.reset();
    this.group.visible = this.active;
    this.ctx.canvas.style.cursor = this.active ? 'crosshair' : '';
    this.notify();
    return this.active;
  }

  private notify(): void {
    this.ctx.onState?.(this.active, this.verts.length);
  }

  private reset(): void {
    this.verts = [];
    this.rebuild();
  }

  /** Raycast the pointer onto the globe (mode 0) or map plane (mode 1). */
  pickLonLat(clientX: number, clientY: number): LonLat | null {
    const rect = this.ctx.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.ctx.camera);
    const hit = new THREE.Vector3();
    if (this.ctx.getMode() === 0) {
      if (!this.raycaster.ray.intersectSphere(this.sphere, hit)) return null;
      return sphereToLonLat(hit.x, hit.y, hit.z);
    }
    if (!this.raycaster.ray.intersectPlane(this.plane, hit)) return null;
    if (Math.abs(hit.x) > Math.PI || Math.abs(hit.y) > Math.PI / 2) return null;
    return planeToLonLat(hit.x, hit.y);
  }

  private handleIndexAt(clientX: number, clientY: number): number {
    const rect = this.ctx.canvas.getBoundingClientRect();
    const v = new THREE.Vector3();
    for (let i = 0; i < this.verts.length; i++) {
      const p =
        this.ctx.getMode() === 0
          ? lonLatToSphere(this.verts[i].lon, this.verts[i].lat)
          : lonLatToPlane(this.verts[i].lon, this.verts[i].lat);
      v.set(p[0], p[1], p[2]).project(this.ctx.camera);
      const sx = ((v.x + 1) / 2) * rect.width + rect.left;
      const sy = ((1 - v.y) / 2) * rect.height + rect.top;
      if (Math.hypot(sx - clientX, sy - clientY) < 10) return i;
    }
    return -1;
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.active) return;
    this.dragIdx = this.handleIndexAt(e.clientX, e.clientY);
    this.dragged = false;
    if (this.dragIdx >= 0) e.stopPropagation(); // keep OrbitControls out of handle drags
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.active || this.dragIdx < 0) return;
    const ll = this.pickLonLat(e.clientX, e.clientY);
    if (!ll) return;
    this.verts[this.dragIdx] = ll;
    this.dragged = true;
    this.rebuild();
    e.stopPropagation();
  };

  private onPointerUp = (): void => {
    this.dragIdx = -1;
  };

  private onClick = (e: MouseEvent): void => {
    if (!this.active) return;
    e.stopPropagation();
    if (this.dragged) {
      this.dragged = false;
      return;
    }
    const ll = this.pickLonLat(e.clientX, e.clientY);
    if (!ll) return;
    const camDist =
      this.ctx.getMode() === 0
        ? Math.max(this.ctx.camera.position.length() - 1, 1e-4)
        : Math.max(this.ctx.camera.position.z, 1e-4);
    if (shouldClosePolygon(this.verts, ll, closeThresholdKm(camDist))) {
      void this.finish();
      return;
    }
    this.verts.push(ll);
    this.rebuild();
    this.notify();
  };

  private onKey = (e: KeyboardEvent): void => {
    if (!this.active || (e.target as HTMLElement)?.closest('input,textarea,select')) return;
    if (e.key === 'Escape') this.toggle();
    else if (e.key === 'Backspace') {
      this.verts.pop();
      this.rebuild();
      this.notify();
    } else if (e.key === 'Enter' && this.verts.length >= 3) void this.finish();
  };

  private async finish(): Promise<void> {
    const verts = this.verts;
    this.toggle(); // leaves draw mode and clears the preview
    await this.ctx.onFinish(verts);
  }

  private rebuild(): void {
    const n = this.verts.length;
    const closed = n >= 3 ? n + 1 : n;
    const pos = new Float32Array(closed * 3);
    const pos2 = new Float32Array(closed * 3);
    for (let i = 0; i < closed; i++) {
      const { lon, lat } = this.verts[i % n];
      const s = lonLatToSphere(lon, lat);
      const p = lonLatToPlane(lon, lat);
      pos.set([s[0] * 1.002, s[1] * 1.002, s[2] * 1.002], i * 3);
      pos2.set([p[0], p[1], 0.003], i * 3);
    }
    for (const obj of [this.line, this.handles]) {
      const geo = obj.geometry;
      const count = obj === this.handles ? n : closed;
      geo.setAttribute('position', new THREE.BufferAttribute(pos.slice(0, count * 3), 3));
      geo.setAttribute('position2', new THREE.BufferAttribute(pos2.slice(0, count * 3), 3));
      geo.computeBoundingSphere();
    }
  }
}
