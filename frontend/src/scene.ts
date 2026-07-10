import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { loadMeshBuffers } from './api';
import { initColorbar, updateColorbar } from './colorbar';
import { Inspector } from './inspect';
import { coastLayer } from './layers/coastLayer';
import { HfunOverlay } from './layers/hfunOverlay';
import { MeshLayer } from './layers/meshLayer';
import { DrawMode } from './modes/draw';
import { RefineMode } from './modes/refine';
import { initLayerPanel } from './panels/layers';
import { PickingPass } from './picking';
import type { CoastBuffers, MeshBuffers, MeshMeta } from './types';
import {
  controlsConfig,
  MIN_MAP_HEIGHT,
  nearFar,
  radialClampFactor,
  rotateSpeed,
  type ViewMode,
} from './viewMode';

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

  // one morph uniform shared by every layer, surviving A/B mesh swaps
  const morph = { value: 0 };
  let mesh = new MeshLayer(meta, buffers, morph);
  scene.add(mesh.group);

  let coastLine: THREE.LineSegments | null = null;
  if (coast) {
    coastLine = coastLayer(coast, morph);
    scene.add(coastLine);
  }

  // ---------- picking / inspect ----------
  let picking = new PickingPass(
    mesh.fill.geometry.getAttribute('position') as THREE.BufferAttribute,
    mesh.fill.geometry.getAttribute('position2') as THREE.BufferAttribute,
    morph,
  );
  let inspector = new Inspector(input.id, buffers, morph);
  scene.add(inspector.marker);

  let pointerX = -1;
  let pointerY = -1;
  let pointerMoved = false;
  let downX = 0;
  let downY = 0;
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (e.buttons !== 0) return; // don't hover-pick mid-drag
    pointerX = e.clientX;
    pointerY = e.clientY;
    pointerMoved = true;
  });
  renderer.domElement.addEventListener('pointerdown', (e) => {
    downX = e.clientX;
    downY = e.clientY;
  });
  renderer.domElement.addEventListener('click', (e) => {
    if (draw.active || refine.active) return; // polygon modes own clicks
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return; // was a drag
    void inspector.pin(picking.pick(renderer, camera, e.clientX, e.clientY));
  });

  // ---------- project panel / hfun preview / draw mode (M2) ----------
  const overlay = new HfunOverlay(morph);
  scene.add(overlay.mesh);
  const refreshOverlay = () => {
    if (!overlay.mesh.visible) return;
    overlay
      .refresh()
      .then(([lo, hi]) => updateColorbar('hfun [km]', lo, hi, 1))
      .catch(() => {
        overlay.mesh.visible = false;
        $('bHfun').classList.remove('on');
      });
  };
  const panel = initLayerPanel({ onHfunStale: refreshOverlay });
  const draw: DrawMode = new DrawMode({
    canvas: renderer.domElement,
    camera,
    scene,
    morph,
    getMode: () => mode,
    onFinish: async (verts) => {
      await panel.addRegionLayer(verts);
      refreshOverlay();
    },
    onState: (active, nVerts) => {
      $('bDraw').classList.toggle('on', active);
      if (active) {
        $('hint').textContent =
          nVerts < 3
            ? 'draw: click to add vertices · Esc cancels'
            : 'draw: click first vertex or Enter to close · drag handles to adjust';
      } else {
        $('hint').textContent = controlsConfig(mode).hint;
      }
    },
  });

  // ---------- refine mode + A/B compare (M3) ----------
  type Slot = { id: string; meta: MeshMeta; buffers: MeshBuffers };
  let slotA: Slot = { id: input.id, meta, buffers };
  let slotB: Slot | null = null;
  let showing: 'A' | 'B' = 'A';

  function installMesh(id: string, m: MeshMeta, b: MeshBuffers): void {
    const wire = mesh.wire.visible;
    const pts = mesh.points.visible;
    const field = mesh.field;
    scene.remove(mesh.group);
    scene.remove(inspector.marker);
    mesh.dispose();
    mesh = new MeshLayer(m, b, morph);
    mesh.wire.visible = wire;
    mesh.points.visible = pts;
    mesh.setField(field);
    scene.add(mesh.group);
    picking = new PickingPass(
      mesh.fill.geometry.getAttribute('position') as THREE.BufferAttribute,
      mesh.fill.geometry.getAttribute('position2') as THREE.BufferAttribute,
      morph,
    );
    inspector = new Inspector(id, b, morph);
    scene.add(inspector.marker);
    updateCounts(m);
    showColorbar();
  }

  function swapAB(): void {
    if (!slotB) return;
    showing = showing === 'A' ? 'B' : 'A';
    const s = showing === 'A' ? slotA : slotB;
    installMesh(s.id, s.meta, s.buffers);
    refine.setSlots(slotA, slotB, showing);
  }

  const refine: RefineMode = new RefineMode({
    canvas: renderer.domElement,
    camera,
    scene,
    morph,
    getMode: () => mode,
    currentMesh: () => {
      const s = showing === 'A' ? slotA : slotB!;
      return { id: s.id, meta: s.meta };
    },
    onRefined: async (m) => {
      const bufs = await loadMeshBuffers(m.id);
      slotA = showing === 'A' ? slotA : slotB!; // A := the mesh that was refined
      slotB = { id: m.id, meta: m.meta, buffers: bufs };
      showing = 'B';
      installMesh(m.id, m.meta, bufs);
      refine.setSlots(slotA, slotB, showing);
    },
    onSwapRequest: () => swapAB(),
    onHint: (text) => {
      $('hint').textContent = text ?? controlsConfig(mode).hint;
    },
  });

  // ---------- HUD ----------
  $('load').style.display = 'none';
  $('hud').style.display = 'block';
  initColorbar();
  function showColorbar(): void {
    const m = mesh.meta;
    const range = mesh.field === 'res' ? m.res_range : m.dep_range;
    updateColorbar(
      mesh.field === 'res' ? 'resolution [km]' : 'depth [m]',
      range[0],
      range[1],
      mesh.field === 'res' ? 1 : 0,
    );
  }
  function updateCounts(m: MeshMeta): void {
    $('counts').innerHTML =
      `<b>${m.name}</b> &nbsp; ${(m.nodes / 1e6).toFixed(2)}M nodes · ` +
      `${(m.triangles / 1e6).toFixed(2)}M tri · ${(m.edges / 1e6).toFixed(2)}M edges<br>` +
      `<span class="dim">${(Object.values(m.bytes).reduce((a, b) => a + b, 0) / 1e6).toFixed(0)} MB buffers</span>`;
  }
  showColorbar();
  updateCounts(meta);
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
    h: () => {
      overlay.mesh.visible = !overlay.mesh.visible;
      $('bHfun').classList.toggle('on', overlay.mesh.visible);
      if (overlay.mesh.visible) refreshOverlay();
      else showColorbar();
    },
    x: () => {
      if (refine.active) return; // one polygon mode at a time
      if (!panel.hasProject() && !draw.active) return; // needs an open project
      draw.toggle();
    },
    v: () => {
      if (draw.active) return; // one polygon mode at a time
      refine.toggle();
    },
    b: () => swapAB(),
  };
  addEventListener('keydown', (e) => {
    if (!(e.target as HTMLElement)?.closest('input,textarea')) actions[e.key.toLowerCase()]?.();
  });
  const buttons: Record<string, string> = {
    bWire: 'w', bField: 'c', bProj: 'g', bPoints: 'p', bCoast: 'l', bSpin: 'r', bDens: 'd',
    bHfun: 'h', bDraw: 'x',
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

    const mv = morph.value;
    if (mv !== morphTarget) {
      const next = mv + (morphTarget - mv) * Math.min(1, dt * 5);
      morph.value = Math.abs(next - morphTarget) < 0.002 ? morphTarget : next;
    }
    inner.visible = morph.value < 0.25;

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

    // hard surface floor, measured from the globe center (the OrbitControls
    // target drifts under zoomToCursor, so its own minDistance can't do this)
    if (mode === 0) {
      camera.position.multiplyScalar(radialClampFactor(camera.position.length()));
    } else if (camera.position.z < MIN_MAP_HEIGHT) {
      camera.position.z = MIN_MAP_HEIGHT;
    }

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
