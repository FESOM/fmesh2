import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { initColorbar, updateColorbar } from './colorbar';
import { Inspector } from './inspect';
import { coastLayer } from './layers/coastLayer';
import { MeshLayer } from './layers/meshLayer';
import { PickingPass } from './picking';
import type { CoastBuffers, MeshBuffers, MeshMeta } from './types';
import { controlsConfig, nearFar, rotateSpeed, type ViewMode } from './viewMode';

export interface ViewerInput {
  id: string;
  meta: MeshMeta;
  buffers: MeshBuffers;
  coast: CoastBuffers | null;
  fetchSec: number;
}

const $ = (id: string) => document.getElementById(id)!;

export function startViewer(input: ViewerInput): void {
  const { meta, buffers, coast } = input;

  const renderer = new THREE.WebGLRenderer({
    canvas: $('canvas') as HTMLCanvasElement,
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 10);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.zoomSpeed = 1.4;
  controls.zoomToCursor = true;

  // dark inner sphere: occludes the far side, land holes read as landmass
  const inner = new THREE.Mesh(
    new THREE.SphereGeometry(0.9985, 128, 64),
    new THREE.MeshBasicMaterial({ color: 0x161a22 }),
  );
  scene.add(inner);

  const mesh = new MeshLayer(meta, buffers);
  scene.add(mesh.group);

  let coastLine: THREE.LineSegments | null = null;
  if (coast) {
    coastLine = coastLayer(coast, mesh.uniforms.uMorph);
    scene.add(coastLine);
  }

  // ---------- picking / inspect ----------
  const picking = new PickingPass(
    mesh.fill.geometry.getAttribute('position') as THREE.BufferAttribute,
    mesh.fill.geometry.getAttribute('position2') as THREE.BufferAttribute,
    mesh.uniforms.uMorph,
  );
  const inspector = new Inspector(input.id, buffers, mesh.uniforms.uMorph);
  scene.add(inspector.marker);

  let pointerX = -1;
  let pointerY = -1;
  let pointerMoved = false;
  let downX = 0;
  let downY = 0;
  renderer.domElement.addEventListener('pointermove', (e) => {
    pointerX = e.clientX;
    pointerY = e.clientY;
    pointerMoved = true;
  });
  renderer.domElement.addEventListener('pointerdown', (e) => {
    downX = e.clientX;
    downY = e.clientY;
  });
  renderer.domElement.addEventListener('click', (e) => {
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return; // was a drag
    void inspector.pin(picking.pick(renderer, camera, e.clientX, e.clientY));
  });

  // ---------- HUD ----------
  $('load').style.display = 'none';
  $('hud').style.display = 'block';
  initColorbar();
  const showColorbar = () => {
    const range = mesh.field === 'res' ? meta.res_range : meta.dep_range;
    updateColorbar(
      mesh.field === 'res' ? 'resolution [km]' : 'depth [m]',
      range[0],
      range[1],
      mesh.field === 'res' ? 1 : 0,
    );
  };
  showColorbar();
  $('counts').innerHTML =
    `<b>${meta.name}</b> &nbsp; ${(meta.nodes / 1e6).toFixed(2)}M nodes · ` +
    `${(meta.triangles / 1e6).toFixed(2)}M tri · ${(meta.edges / 1e6).toFixed(2)}M edges<br>` +
    `<span class="dim">${(Object.values(meta.bytes).reduce((a, b) => a + b, 0) / 1e6).toFixed(0)} MB buffers</span>`;
  $('loadtime').textContent = `fetch ${input.fetchSec.toFixed(1)}s`;

  // ---------- view mode ----------
  let mode: ViewMode = 0;
  let morphTarget = 0;
  function setMode(m: ViewMode): void {
    mode = m;
    morphTarget = m;
    const cfg = controlsConfig(m);
    controls.enableRotate = cfg.enableRotate;
    controls.mouseButtons.LEFT = cfg.leftIsPan ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    controls.screenSpacePanning = true;
    controls.minDistance = cfg.minDistance;
    controls.maxDistance = cfg.maxDistance;
    controls.target.set(0, 0, 0);
    camera.position.set(...cfg.cameraPos);
    if (m === 1) {
      controls.autoRotate = false;
      $('bSpin').classList.remove('on');
    }
    $('bProj').textContent = cfg.label;
    $('hint').textContent = cfg.hint;
  }
  setMode(0);

  // ---------- actions ----------
  const actions: Record<string, () => void> = {
    w: () => {
      mesh.wire.visible = !mesh.wire.visible;
      $('bWire').classList.toggle('on', mesh.wire.visible);
    },
    c: () => {
      mesh.setField(mesh.field === 'res' ? 'dep' : 'res');
      $('bField').textContent = `[C]olor: ${mesh.field === 'res' ? 'resolution' : 'depth'}`;
      showColorbar();
    },
    p: () => {
      mesh.points.visible = !mesh.points.visible;
      $('bPoints').classList.toggle('on', mesh.points.visible);
    },
    l: () => {
      if (!coastLine) return;
      coastLine.visible = !coastLine.visible;
      $('bCoast').classList.toggle('on', coastLine.visible);
    },
    r: () => {
      if (mode === 1) return;
      controls.autoRotate = !controls.autoRotate;
      $('bSpin').classList.toggle('on', controls.autoRotate);
    },
    g: () => setMode(mode === 0 ? 1 : 0),
    d: () => {
      const lo = renderer.getPixelRatio() > 1;
      renderer.setPixelRatio(lo ? 1 : Math.min(devicePixelRatio, 2));
      $('bDens').textContent = `[D] res ×${renderer.getPixelRatio()}`;
    },
  };
  addEventListener('keydown', (e) => {
    if (!(e.target as HTMLElement)?.closest('input,textarea')) actions[e.key.toLowerCase()]?.();
  });
  const buttons: Record<string, string> = {
    bWire: 'w', bField: 'c', bProj: 'g', bPoints: 'p', bCoast: 'l', bSpin: 'r', bDens: 'd',
  };
  for (const [btn, key] of Object.entries(buttons)) {
    $(btn).addEventListener('click', () => actions[key]());
  }

  function resize(): void {
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
  }
  addEventListener('resize', resize);
  resize();

  // ---------- render loop ----------
  const times: number[] = [];
  let lastHud = 0;
  let prevT = 0;
  let lastPick = 0;
  renderer.setAnimationLoop((t) => {
    const dt = prevT ? (t - prevT) / 1000 : 0.016;
    prevT = t;

    if (pointerMoved && t - lastPick > 120) {
      pointerMoved = false;
      lastPick = t;
      inspector.hover(picking.pick(renderer, camera, pointerX, pointerY));
    }

    const mv = mesh.uniforms.uMorph.value;
    if (mv !== morphTarget) {
      const next = mv + (morphTarget - mv) * Math.min(1, dt * 5);
      mesh.uniforms.uMorph.value = Math.abs(next - morphTarget) < 0.002 ? morphTarget : next;
    }
    inner.visible = mesh.uniforms.uMorph.value < 0.25;

    const d =
      mode === 0
        ? Math.max(camera.position.length() - 1, 1e-4)
        : Math.max(controls.getDistance(), 1e-3);
    controls.rotateSpeed = rotateSpeed(d);
    const nf = nearFar(d);
    camera.near = nf.near;
    camera.far = nf.far;
    camera.updateProjectionMatrix();
    controls.update();
    renderer.render(scene, camera);

    times.push(t);
    while (times.length > 90) times.shift();
    if (t - lastHud > 250 && times.length > 10) {
      const fdt = (times[times.length - 1] - times[0]) / (times.length - 1);
      $('fps').textContent = `${(1000 / fdt).toFixed(0)} fps`;
      $('ms').textContent = `${fdt.toFixed(1)} ms/frame`;
      lastHud = t;
    }
  });
}
