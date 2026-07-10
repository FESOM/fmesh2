import * as THREE from 'three';

/** id -> RGB bytes (id + 1, so that cleared black decodes to NONE). */
export function encodeId(id: number): [number, number, number] {
  const v = id + 1;
  return [v & 255, (v >> 8) & 255, (v >> 16) & 255];
}

export function decodeId(r: number, g: number, b: number): number {
  return r + g * 256 + b * 65536 - 1;
}

export const PICK_NONE = -1;

/** Side of the square pick window read around the cursor, in device px. */
export const PICK_WINDOW = 33;

/**
 * Scan an RGBA pick-window readback for the id nearest to the window center.
 * Solves the sparse-node problem: when zoomed in, nodes are hundreds of px
 * apart, so "the exact pixel under the cursor" almost never hits one.
 */
export function nearestIdInWindow(pixels: Uint8Array, w: number): number {
  const c = (w - 1) / 2;
  let best = PICK_NONE;
  let bestD2 = Infinity;
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const id = decodeId(pixels[o], pixels[o + 1], pixels[o + 2]);
      if (id === PICK_NONE) continue;
      const d2 = (x - c) * (x - c) + (y - c) * (y - c);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = id;
      }
    }
  }
  return best;
}

const PICK_VERT = /* glsl */ `
  attribute vec3 position2; attribute float vid;
  uniform float uMorph;
  varying float vVid;
  void main() {
    vVid = vid;
    gl_PointSize = 9.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
  }`;

const PICK_FRAG = /* glsl */ `
  varying float vVid;
  void main() {
    float id = vVid + 1.0;
    gl_FragColor = vec4(
      mod(id, 256.0) / 255.0,
      mod(floor(id / 256.0), 256.0) / 255.0,
      mod(floor(id / 65536.0), 256.0) / 255.0,
      1.0);
  }`;

/** Offscreen ID render pass: O(1) node hit-testing at any mesh size. */
export class PickingPass {
  readonly scene = new THREE.Scene();
  private readonly target = new THREE.WebGLRenderTarget(PICK_WINDOW, PICK_WINDOW);
  private readonly pixels = new Uint8Array(PICK_WINDOW * PICK_WINDOW * 4);
  private readonly occluder: THREE.Mesh;

  constructor(
    posAttr: THREE.BufferAttribute,
    pos2Attr: THREE.BufferAttribute,
    private readonly uMorph: { value: number },
  ) {
    const n = posAttr.count;
    const vid = new Float32Array(n);
    for (let i = 0; i < n; i++) vid[i] = i;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', posAttr);
    geo.setAttribute('position2', pos2Attr);
    geo.setAttribute('vid', new THREE.BufferAttribute(vid, 1));
    const pts = new THREE.Points(
      geo,
      new THREE.ShaderMaterial({
        uniforms: { uMorph: this.uMorph },
        vertexShader: PICK_VERT,
        fragmentShader: PICK_FRAG,
      }),
    );
    pts.frustumCulled = false;

    // hides far-side nodes on the globe; decodes to PICK_NONE (black)
    this.occluder = new THREE.Mesh(
      new THREE.SphereGeometry(0.9985, 64, 32),
      new THREE.MeshBasicMaterial({ color: 0x000000 }),
    );
    this.scene.add(this.occluder, pts);
  }

  /** Pick near CSS pixel (x, y): nearest node id in the window, or PICK_NONE. */
  pick(renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera, x: number, y: number): number {
    this.occluder.visible = this.uMorph.value < 0.25;
    const dpr = renderer.getPixelRatio();
    const size = renderer.getSize(new THREE.Vector2());
    camera.setViewOffset(
      size.x * dpr,
      size.y * dpr,
      Math.round(x * dpr - PICK_WINDOW / 2),
      Math.round(y * dpr - PICK_WINDOW / 2),
      PICK_WINDOW,
      PICK_WINDOW,
    );
    const prevTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(this.target);
    renderer.render(this.scene, camera);
    renderer.setRenderTarget(prevTarget);
    camera.clearViewOffset();
    renderer.readRenderTargetPixels(
      this.target, 0, 0, PICK_WINDOW, PICK_WINDOW, this.pixels,
    );
    return nearestIdInWindow(this.pixels, PICK_WINDOW);
  }
}
