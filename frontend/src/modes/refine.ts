import type * as THREE from 'three';
import { getJob, getMeshMeta, refineMesh, type JobSnapshot } from '../api';
import { unwrapRing, type LonLat } from '../geo';
import type { MeshMeta } from '../types';
import { DrawMode } from './draw';

export interface MeshInfo {
  id: string;
  meta: MeshMeta;
}

export interface RefineContext {
  canvas: HTMLElement;
  camera: THREE.PerspectiveCamera;
  scene: THREE.Scene;
  morph: { value: number };
  getMode: () => 0 | 1;
  currentMesh: () => MeshInfo;
  /** Load the refined mesh (buffers etc.) and show it as the B slot. */
  onRefined: (mesh: MeshInfo) => Promise<void>;
  /** A/B hotkey handler lives in the scene (it owns the slots). */
  onSwapRequest: () => void;
  onHint: (text: string | null) => void;
}

/** Refine mode: the Draw polygon UX on a loaded mesh + target inputs.
 * Finishing the polygon POSTs a refine job, streams its progress into the
 * panel, and hands the resulting mesh to the scene as the B slot. */
export class RefineMode {
  private readonly draw: DrawMode;
  private readonly status: HTMLElement;
  private readonly abLine: HTMLElement;
  private readonly targetInput: HTMLInputElement;
  private readonly transInput: HTMLInputElement;
  private polling: number | undefined;
  private pollBusy = false; // interval ticks can overlap their async work
  private settled = false;
  private lastRegion: string | null = null;

  constructor(private readonly ctx: RefineContext) {
    const panel = document.createElement('div');
    panel.id = 'refine';
    panel.className = 'panel';
    panel.innerHTML = `
      <div><b>refine</b> <button id="bRefineDraw">[V] draw refine region</button></div>
      <div id="refform">
        <label>target km <input id="refTarget" type="number" step="any" value="25"></label>
        <label>transition km <input id="refTrans" type="number" step="any" placeholder="auto"></label>
      </div>
      <div id="refStatus" class="dim">draw a polygon on the mesh to refine it</div>
      <div id="refAB"></div>`;
    document.body.appendChild(panel);
    this.status = panel.querySelector('#refStatus')!;
    this.abLine = panel.querySelector('#refAB')!;
    this.targetInput = panel.querySelector('#refTarget')!;
    this.transInput = panel.querySelector('#refTrans')!;
    panel.querySelector('#bRefineDraw')!.addEventListener('click', () => this.toggle());

    this.draw = new DrawMode({
      canvas: ctx.canvas,
      camera: ctx.camera,
      scene: ctx.scene,
      morph: ctx.morph,
      getMode: ctx.getMode,
      onFinish: (verts) => this.submit(verts),
      onState: (active, nVerts) => {
        document.getElementById('bRefineDraw')!.classList.toggle('on', active);
        if (active) {
          this.ctx.onHint(
            nVerts < 3
              ? 'refine: click to outline the region · Esc cancels'
              : 'refine: click first vertex or Enter to remesh · drag handles to adjust',
          );
        } else {
          this.ctx.onHint(null);
        }
      },
    });
  }

  get active(): boolean {
    return this.draw.active;
  }

  toggle(): void {
    this.draw.toggle();
  }

  /** Render the A/B readout; called by the scene whenever slots change. */
  setSlots(a: MeshInfo, b: MeshInfo | null, showing: 'A' | 'B'): void {
    if (!b) {
      this.abLine.innerHTML = '';
      return;
    }
    const dn = b.meta.nodes - a.meta.nodes;
    const dt = b.meta.triangles - a.meta.triangles;
    const sign = (v: number) => (v >= 0 ? `+${v.toLocaleString()}` : v.toLocaleString());
    this.abLine.innerHTML =
      `<div style="margin-top:5px"><button id="bSwapAB">[B] showing: ${showing}</button></div>` +
      `A ${a.meta.nodes.toLocaleString()}n/${a.meta.triangles.toLocaleString()}t · ` +
      `B ${b.meta.nodes.toLocaleString()}n/${b.meta.triangles.toLocaleString()}t ` +
      `(Δ ${sign(dn)}n, ${sign(dt)}t)` +
      (this.lastRegion ? `<div class="dim">region ${this.lastRegion}</div>` : '');
    this.abLine.querySelector('#bSwapAB')!.addEventListener('click', this.ctx.onSwapRequest);
  }

  private async submit(verts: LonLat[]): Promise<void> {
    const target = Number(this.targetInput.value);
    if (!Number.isFinite(target) || target <= 0) {
      this.status.textContent = 'set a positive target resolution first';
      return;
    }
    const trans = Number(this.transInput.value);
    const ring = unwrapRing(verts);
    const lons = verts.map((v) => v.lon);
    const lats = verts.map((v) => v.lat);
    this.lastRegion =
      `[${Math.min(...lons).toFixed(1)}..${Math.max(...lons).toFixed(1)}]×` +
      `[${Math.min(...lats).toFixed(1)}..${Math.max(...lats).toFixed(1)}]`;

    let jobId: string;
    try {
      ({ job_id: jobId } = await refineMesh(this.ctx.currentMesh().id, {
        polygon: ring.map((v) => [v.lon, v.lat]),
        target_km: target,
        transition_km: Number.isFinite(trans) && trans > 0 ? trans : null,
      }));
    } catch (err) {
      this.status.textContent = String(err);
      return;
    }
    this.status.textContent = 'refining…';
    clearInterval(this.polling);
    this.settled = false;
    this.pollBusy = false;
    this.polling = window.setInterval(() => void this.poll(jobId), 400);
  }

  private async poll(jobId: string): Promise<void> {
    // one poll in flight at a time, and exactly one terminal handling —
    // otherwise a second racing poll would run onRefined twice
    if (this.pollBusy || this.settled) return;
    this.pollBusy = true;
    try {
      await this.pollOnce(jobId);
    } finally {
      this.pollBusy = false;
    }
  }

  private async pollOnce(jobId: string): Promise<void> {
    let snap: JobSnapshot;
    try {
      snap = await getJob(jobId);
    } catch (err) {
      clearInterval(this.polling);
      this.settled = true;
      this.status.textContent = String(err);
      return;
    }
    if (snap.status === 'running') {
      this.status.textContent = `refining: ${snap.stage} ${(snap.fraction * 100).toFixed(0)}%`;
      return;
    }
    clearInterval(this.polling);
    this.settled = true;
    if (snap.status !== 'done' || !snap.result?.mesh_id) {
      this.status.textContent = `refine ${snap.status}: ${snap.error ?? 'no result'}`;
      return;
    }
    const rep = snap.result.report;
    this.status.innerHTML =
      `done: <b>${rep?.n_new_interior_nodes.toLocaleString()}</b> new nodes` +
      (rep?.interior_resolution_km
        ? `, median ${rep.interior_resolution_km.toFixed(1)} km`
        : '') +
      (rep && !rep.watertight_seam ? ' <span class="warn">⚠ seam not watertight</span>' : '') +
      ` · <a href="/api/mesh/${snap.result.mesh_id}/mapping" download>mapping.csv</a>`;
    try {
      const meta = await getMeshMeta(snap.result.mesh_id);
      await this.ctx.onRefined({ id: snap.result.mesh_id, meta });
    } catch (err) {
      this.status.textContent = String(err);
    }
  }
}
