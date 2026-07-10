import * as THREE from 'three';
import {
  EditRejected,
  editMove,
  editOpen,
  editRedo,
  editSave,
  editSelectBrush,
  editSelectLasso,
  editSmooth,
  editUndo,
  type EditJournalState,
  type EditOpResult,
  type VertexUpdate,
} from '../api';
import {
  applyDragKm,
  dragDeltaKm,
  lonLatToPlane,
  lonLatToSphere,
  unwrapRing,
  type LonLat,
} from '../geo';
import type { MeshLayer } from '../layers/meshLayer';
import type { PickingPass } from '../picking';
import { Selection } from '../selection';
import type { MeshBuffers } from '../types';
import { DrawMode } from './draw';

export interface EditContext {
  canvas: HTMLElement;
  camera: THREE.PerspectiveCamera;
  scene: THREE.Scene;
  morph: { value: number };
  getMode: () => 0 | 1;
  renderer: THREE.WebGLRenderer;
  getPicking: () => PickingPass;
  getLayer: () => MeshLayer;
  getBuffers: () => MeshBuffers;
  currentMeshId: () => string;
  onHint: (text: string | null) => void;
}

type Tool = 'select' | 'lasso' | 'brush';

const HINTS: Record<Tool, string> = {
  select: 'edit: click toggles a node · drag a selected node to move the selection',
  lasso: 'edit lasso: click to outline · Enter/first vertex closes · Esc cancels',
  brush: 'edit brush: drag to select · hold Alt to deselect',
};

/** Edit mode (M4): select (click / lasso / brush), tangent-plane drag with
 * live buffer preview, smooth, journaled undo/redo, save-as. The backend
 * op is authoritative — after each commit the server's vertex update is
 * re-applied, and rejected ops revert the preview and tint the offending
 * elements. */
export class EditMode {
  active = false;
  private tool: Tool = 'select';
  private selection: Selection | null = null;
  private journal: EditJournalState = { can_undo: false, can_redo: false, n_ops: 0 };
  private busy = false;

  private dragging = false;
  private dragMoved = false;
  private downX = 0;
  private downY = 0;
  private dragAnchor: LonLat | null = null;
  private readonly dragStart = new Map<number, LonLat>();
  private readonly dragCurrent = new Map<number, LonLat>();

  private brushing = false;
  private brushErase = false;
  private brushBusy = false;
  private lastBrush = 0;

  private readonly lasso: DrawMode;
  private tint: THREE.Mesh | null = null;
  private tintTimer: number | undefined;

  private readonly tools: HTMLElement;
  private readonly status: HTMLElement;
  private readonly selLine: HTMLElement;
  private readonly brushInput: HTMLInputElement;
  private readonly iterInput: HTMLInputElement;
  private readonly strInput: HTMLInputElement;
  private readonly outInput: HTMLInputElement;

  constructor(private readonly ctx: EditContext) {
    const panel = document.createElement('div');
    panel.id = 'edit';
    panel.className = 'panel';
    panel.innerHTML = `
      <div><b>edit</b> <button id="bEdit">[E] edit nodes</button></div>
      <div id="editTools" style="display:none">
        <div>
          <button id="bToolSel" class="on">click</button><button id="bToolLasso">lasso</button><button id="bToolBrush">brush</button>
          <label>brush km <input id="editBrushR" type="number" step="any" value="150"></label>
        </div>
        <div id="editSelCount" class="dim">0 nodes selected</div>
        <div>
          <button id="bSmoothSel">smooth selection</button>
          <label>iter <input id="editIter" type="number" min="1" value="2"></label>
          <label>strength <input id="editStr" type="number" step="0.1" min="0.1" max="1" value="0.5"></label>
        </div>
        <div>
          <button id="bUndoOp" disabled>[⌘Z] undo</button><button id="bRedoOp" disabled>[⇧⌘Z] redo</button><button id="bClearSel">clear</button>
        </div>
        <div>
          <label>save as <input id="editOut" type="text" placeholder="/absolute/output/dir"></label>
          <button id="bSaveAs">save</button>
        </div>
        <div id="editStatus" class="dim">enter edit mode to start</div>
      </div>`;
    document.body.appendChild(panel);
    this.tools = panel.querySelector('#editTools')!;
    this.status = panel.querySelector('#editStatus')!;
    this.selLine = panel.querySelector('#editSelCount')!;
    this.brushInput = panel.querySelector('#editBrushR')!;
    this.iterInput = panel.querySelector('#editIter')!;
    this.strInput = panel.querySelector('#editStr')!;
    this.outInput = panel.querySelector('#editOut')!;
    panel.querySelector('#bEdit')!.addEventListener('click', () => this.toggle());
    panel.querySelector('#bToolSel')!.addEventListener('click', () => this.setTool('select'));
    panel.querySelector('#bToolLasso')!.addEventListener('click', () => this.setTool('lasso'));
    panel.querySelector('#bToolBrush')!.addEventListener('click', () => this.setTool('brush'));
    panel.querySelector('#bSmoothSel')!.addEventListener('click', () => void this.smooth());
    panel.querySelector('#bUndoOp')!.addEventListener('click', () => void this.undo());
    panel.querySelector('#bRedoOp')!.addEventListener('click', () => void this.redo());
    panel.querySelector('#bClearSel')!.addEventListener('click', () => this.clearSelection());
    panel.querySelector('#bSaveAs')!.addEventListener('click', () => void this.save());

    // registered before the lasso DrawMode so this handler sees lasso.active
    // BEFORE the DrawMode's own Escape handling closes it
    addEventListener('keydown', this.onKey);
    this.lasso = new DrawMode({
      canvas: ctx.canvas,
      camera: ctx.camera,
      scene: ctx.scene,
      morph: ctx.morph,
      getMode: ctx.getMode,
      onFinish: (verts) => this.finishLasso(verts),
      onState: (active) => {
        document.getElementById('bToolLasso')!.classList.toggle('on', active);
        if (!active && this.active && this.tool === 'lasso') {
          this.tool = 'select';
          this.refreshToolButtons();
          this.ctx.onHint(HINTS.select);
        }
      },
    });

    ctx.canvas.addEventListener('pointerdown', this.onPointerDown, true);
    ctx.canvas.addEventListener('pointermove', this.onPointerMove, true);
    ctx.canvas.addEventListener('pointerup', this.onPointerUp, true);
    ctx.canvas.addEventListener('click', this.onClick, true);
  }

  toggle(): void {
    if (this.active) this.exit();
    else void this.enter();
  }

  /** The scene calls this on A/B swaps: sessions/selections are per mesh. */
  onMeshInstalled(): void {
    if (this.active) this.exit();
  }

  private async enter(): Promise<void> {
    this.active = true;
    const layer = this.ctx.getLayer();
    this.selection = new Selection(layer.meta.nodes, this.ctx.getBuffers().dupsrc);
    layer.selPoints.visible = true;
    this.tools.style.display = 'block';
    document.getElementById('bEdit')!.classList.add('on');
    this.setTool('select');
    this.refreshSelCount();
    this.setStatus('opening edit session…');
    try {
      const res = await editOpen(this.ctx.currentMeshId());
      this.journal = res.journal;
      this.refreshJournalButtons();
      this.setStatus(
        `session open · ${res.n_nodes.toLocaleString()} nodes` +
          (res.journal.n_ops ? ` · ${res.journal.n_ops} ops` : ''),
      );
    } catch (err) {
      this.setStatus(String(err));
    }
  }

  private exit(): void {
    this.active = false;
    if (this.lasso.active) this.lasso.toggle();
    const layer = this.ctx.getLayer();
    layer.clearSelectionFlags();
    layer.selPoints.visible = false;
    this.selection = null;
    this.dragging = false;
    this.brushing = false;
    this.clearTint();
    this.tools.style.display = 'none';
    document.getElementById('bEdit')!.classList.remove('on');
    this.ctx.canvas.style.cursor = '';
    this.ctx.onHint(null);
  }

  private setTool(tool: Tool): void {
    if (!this.active) return;
    if (this.tool === 'lasso' && tool !== 'lasso' && this.lasso.active) this.lasso.toggle();
    this.tool = tool;
    if (tool === 'lasso' && !this.lasso.active) this.lasso.toggle();
    this.refreshToolButtons();
    this.ctx.canvas.style.cursor = tool === 'select' ? '' : 'crosshair';
    this.ctx.onHint(HINTS[tool]);
  }

  private refreshToolButtons(): void {
    document.getElementById('bToolSel')!.classList.toggle('on', this.tool === 'select');
    document.getElementById('bToolLasso')!.classList.toggle('on', this.tool === 'lasso');
    document.getElementById('bToolBrush')!.classList.toggle('on', this.tool === 'brush');
  }

  private refreshJournalButtons(): void {
    (document.getElementById('bUndoOp') as HTMLButtonElement).disabled = !this.journal.can_undo;
    (document.getElementById('bRedoOp') as HTMLButtonElement).disabled = !this.journal.can_redo;
  }

  private refreshSelCount(): void {
    this.selLine.textContent = `${this.selection?.size ?? 0} nodes selected`;
  }

  private setStatus(text: string, warn = false): void {
    this.status.textContent = text;
    this.status.className = warn ? 'warn' : 'dim';
  }

  // ------------------------------------------------------------- selection

  private pickNode(x: number, y: number): number {
    const vid = this.ctx.getPicking().pick(this.ctx.renderer, this.ctx.camera, x, y);
    return vid < 0 ? -1 : this.selection!.nodeOfVertex(vid);
  }

  private nodeLonLat(node: number): LonLat {
    const b = this.ctx.getBuffers();
    return {
      lon: (b.pos2[node * 3] * 180) / Math.PI,
      lat: (b.pos2[node * 3 + 1] * 180) / Math.PI,
    };
  }

  private addNodes(ids: number[], erase = false): void {
    if (!this.selection) return;
    const verts = erase ? this.selection.remove(ids) : this.selection.add(ids);
    if (verts.length) this.ctx.getLayer().setSelectionFlags(verts, erase ? 0 : 1);
    this.refreshSelCount();
  }

  private clearSelection(): void {
    if (!this.selection) return;
    const verts = this.selection.clear();
    if (verts.length) this.ctx.getLayer().setSelectionFlags(verts, 0);
    this.refreshSelCount();
  }

  private async finishLasso(verts: LonLat[]): Promise<void> {
    const ring = unwrapRing(verts).map((v) => [v.lon, v.lat] as [number, number]);
    try {
      const res = await editSelectLasso(this.ctx.currentMeshId(), ring);
      this.addNodes(res.ids);
      this.setStatus(`lasso selected ${res.ids.length.toLocaleString()} nodes`);
    } catch (err) {
      this.setStatus(String(err), true);
    }
  }

  private async applyBrush(x: number, y: number): Promise<void> {
    if (this.brushBusy) return;
    const ll = this.lasso.pickLonLat(x, y);
    if (!ll) return;
    const radius = Number(this.brushInput.value);
    if (!Number.isFinite(radius) || radius <= 0) return;
    this.brushBusy = true;
    try {
      const res = await editSelectBrush(this.ctx.currentMeshId(), [ll.lon, ll.lat], radius);
      this.addNodes(res.ids, this.brushErase);
    } catch (err) {
      this.setStatus(String(err), true);
    } finally {
      this.brushBusy = false;
    }
  }

  // ------------------------------------------------------------------ drag

  private capture(e: PointerEvent): void {
    try {
      (this.ctx.canvas as Element).setPointerCapture(e.pointerId);
    } catch {
      /* pointer already gone */
    }
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.active || this.lasso.active || e.button !== 0) return;
    this.dragMoved = false;
    this.downX = e.clientX;
    this.downY = e.clientY;
    if (this.tool === 'brush') {
      this.brushing = true;
      this.brushErase = e.altKey;
      this.lastBrush = 0;
      this.capture(e);
      void this.applyBrush(e.clientX, e.clientY);
      e.stopPropagation();
      return;
    }
    if (this.tool !== 'select' || !this.selection?.size) return;
    const node = this.pickNode(e.clientX, e.clientY);
    if (node < 0 || !this.selection.has(node)) return;
    const anchor = this.lasso.pickLonLat(e.clientX, e.clientY);
    if (!anchor) return;
    this.dragging = true;
    this.dragAnchor = anchor;
    this.dragStart.clear();
    this.dragCurrent.clear();
    for (const id of this.selection.ids()) this.dragStart.set(id, this.nodeLonLat(id));
    this.capture(e);
    e.stopPropagation(); // keep OrbitControls out of the node drag
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.active) return;
    if (this.brushing) {
      if (e.timeStamp - this.lastBrush > 120) {
        this.lastBrush = e.timeStamp;
        void this.applyBrush(e.clientX, e.clientY);
      }
      e.stopPropagation();
      return;
    }
    if (!this.dragging || !this.dragAnchor) return;
    const cur = this.lasso.pickLonLat(e.clientX, e.clientY);
    if (!cur) return;
    const delta = dragDeltaKm(this.dragAnchor, cur);
    this.previewPositions((start) => applyDragKm(start, delta.eastKm, delta.northKm));
    this.dragMoved = true;
    e.stopPropagation();
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (this.brushing) {
      this.brushing = false;
      e.stopPropagation();
      return;
    }
    if (!this.dragging) return;
    this.dragging = false;
    e.stopPropagation();
    if (this.dragMoved) void this.commitDrag();
  };

  private onClick = (e: MouseEvent): void => {
    if (!this.active || this.lasso.active) return;
    e.stopPropagation(); // the inspector must not pin while editing
    if (this.dragMoved) {
      this.dragMoved = false;
      return;
    }
    if (this.tool !== 'select') return;
    // an orbit gesture ends in a click too — only true clicks toggle nodes
    if (Math.hypot(e.clientX - this.downX, e.clientY - this.downY) > 4) return;
    const node = this.pickNode(e.clientX, e.clientY);
    if (node < 0 || !this.selection) return;
    const { selected, verts } = this.selection.toggle(node);
    this.ctx.getLayer().setSelectionFlags(verts, selected ? 1 : 0);
    this.refreshSelCount();
  };

  /** Recompute every dragged node's position and upload it to the GPU. */
  private previewPositions(place: (start: LonLat) => LonLat): void {
    if (!this.selection) return;
    const verts: number[] = [];
    const pos: [number, number, number][] = [];
    const pos2: [number, number, number][] = [];
    for (const [node, start] of this.dragStart) {
      const p = place(start);
      this.dragCurrent.set(node, p);
      const s = lonLatToSphere(p.lon, p.lat);
      const pl = lonLatToPlane(p.lon, p.lat);
      for (const v of this.selection.vertsOf(node)) {
        verts.push(v);
        pos.push(s);
        pos2.push(v < this.selection.nodes ? pl : [pl[0] + 2 * Math.PI, pl[1], 0]);
      }
    }
    this.ctx.getLayer().updatePositions(verts, pos, pos2);
  }

  private async commitDrag(): Promise<void> {
    const ids = [...this.dragCurrent.keys()];
    const lonlat = ids.map((n) => {
      const p = this.dragCurrent.get(n)!;
      return [p.lon, p.lat] as [number, number];
    });
    try {
      const res = await editMove(this.ctx.currentMeshId(), ids, lonlat);
      this.applyResult('moved', res);
    } catch (err) {
      this.previewPositions((start) => start); // revert the live preview
      if (err instanceof EditRejected) {
        this.tintElements(err.elementIds);
        this.setStatus(err.message, true);
      } else {
        this.setStatus(String(err), true);
      }
    }
  }

  // ------------------------------------------------------------------- ops

  private applyUpdate(update: VertexUpdate): void {
    const layer = this.ctx.getLayer();
    layer.updatePositions(update.verts, update.pos, update.pos2);
    if (update.dep) layer.updateDepths(update.verts, update.dep);
  }

  private applyResult(verb: string, res: EditOpResult): void {
    this.applyUpdate(res.update);
    this.journal = res.journal;
    this.refreshJournalButtons();
    const r = res.report;
    const angles =
      r.min_angle_before_deg !== null && r.min_angle_after_deg !== null
        ? ` · min angle ${(r.min_angle_before_deg as number).toFixed(1)}°→${(
            r.min_angle_after_deg as number
          ).toFixed(1)}°`
        : '';
    this.setStatus(`${verb} ${r.n_nodes.toLocaleString()} nodes${angles}`);
  }

  private async smooth(): Promise<void> {
    if (this.busy || !this.selection?.size) {
      if (!this.selection?.size) this.setStatus('select nodes to smooth first', true);
      return;
    }
    const iterations = Math.max(1, Math.round(Number(this.iterInput.value) || 1));
    const strength = Math.min(1, Math.max(0.01, Number(this.strInput.value) || 0.5));
    this.busy = true;
    try {
      const res = await editSmooth(
        this.ctx.currentMeshId(),
        this.selection.ids(),
        iterations,
        strength,
      );
      this.applyResult('smoothed', res);
    } catch (err) {
      if (err instanceof EditRejected) this.tintElements(err.elementIds);
      this.setStatus(err instanceof Error ? err.message : String(err), true);
    } finally {
      this.busy = false;
    }
  }

  private async undo(): Promise<void> {
    await this.step(editUndo, 'undid');
  }

  private async redo(): Promise<void> {
    await this.step(editRedo, 'redid');
  }

  private async step(
    call: (id: string) => Promise<{ update: VertexUpdate | null; journal: EditJournalState }>,
    verb: string,
  ): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const res = await call(this.ctx.currentMeshId());
      if (res.update) this.applyUpdate(res.update);
      this.journal = res.journal;
      this.refreshJournalButtons();
      this.setStatus(res.update ? `${verb} last op` : `nothing to ${verb.slice(0, 4)}`);
    } catch (err) {
      this.setStatus(String(err), true);
    } finally {
      this.busy = false;
    }
  }

  private async save(): Promise<void> {
    const out = this.outInput.value.trim();
    if (!out) {
      this.setStatus('set an output directory first', true);
      return;
    }
    if (this.busy) return;
    this.busy = true;
    this.setStatus('saving…');
    try {
      const res = await editSave(this.ctx.currentMeshId(), out);
      this.setStatus(`saved → ${res.out_dir} (mesh ${res.mesh_id}, journal included)`);
    } catch (err) {
      this.setStatus(String(err), true);
    } finally {
      this.busy = false;
    }
  }

  // ------------------------------------------------------------ tint layer

  /** Flash the elements an op would have inverted (quality warning). */
  private tintElements(elems: number[]): void {
    this.clearTint();
    if (!elems.length) return;
    const b = this.ctx.getBuffers();
    const pos = new Float32Array(elems.length * 9);
    const pos2 = new Float32Array(elems.length * 9);
    elems.forEach((e, k) => {
      for (let c = 0; c < 3; c++) {
        const v = b.tri[e * 3 + c];
        pos.set([b.pos[v * 3], b.pos[v * 3 + 1], b.pos[v * 3 + 2]], (k * 3 + c) * 3);
        pos2.set([b.pos2[v * 3], b.pos2[v * 3 + 1], 0.002], (k * 3 + c) * 3);
      }
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('position2', new THREE.BufferAttribute(pos2, 3));
    this.tint = new THREE.Mesh(
      geo,
      new THREE.ShaderMaterial({
        uniforms: { uMorph: this.ctx.morph },
        vertexShader: /* glsl */ `
          attribute vec3 position2; uniform float uMorph;
          void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
          }`,
        fragmentShader: `void main() { gl_FragColor = vec4(1.0, 0.15, 0.15, 0.55); }`,
        transparent: true,
        depthTest: false,
        side: THREE.DoubleSide,
      }),
    );
    this.tint.frustumCulled = false;
    this.tint.renderOrder = 4;
    this.ctx.scene.add(this.tint);
    this.tintTimer = window.setTimeout(() => this.clearTint(), 2500);
  }

  private clearTint(): void {
    clearTimeout(this.tintTimer);
    if (!this.tint) return;
    this.ctx.scene.remove(this.tint);
    this.tint.geometry.dispose();
    (this.tint.material as THREE.Material).dispose();
    this.tint = null;
  }

  // ------------------------------------------------------------------ keys

  private onKey = (e: KeyboardEvent): void => {
    if (!this.active) return;
    if ((e.target as HTMLElement)?.closest('input,textarea,select')) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      void (e.shiftKey ? this.redo() : this.undo());
    } else if (e.key === 'Escape' && !this.lasso.active) {
      this.exit();
    }
  };
}
