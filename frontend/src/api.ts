import type { CoastBuffers, MeshBuffers, MeshMeta } from './types';

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(`${url}: ${r.status} ${detail}`);
  }
  return r.json() as Promise<T>;
}

/** Streaming binary fetch; onBytes is called with each chunk's size. */
export async function fetchBuffer(
  url: string,
  onBytes: (n: number) => void = () => {},
): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok || !r.body) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  const reader = r.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const ch of chunks) {
    out.set(ch, off);
    off += ch.length;
  }
  return out.buffer;
}

export interface MeshListEntry {
  id: string;
  meta: MeshMeta;
}

export const listMeshes = () => fetchJson<MeshListEntry[]>('/api/meshes');

export const openMesh = (path: string) =>
  fetchJson<MeshListEntry>('/api/meshes/open', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path }),
  });

export async function loadMeshBuffers(
  id: string,
  onBytes: (n: number) => void = () => {},
): Promise<MeshBuffers> {
  const get = (name: string) => fetchBuffer(`/api/mesh/${id}/buffer/${name}`, onBytes);
  const [pos, pos2, tri, edge, res, dep, dupsrc] = await Promise.all(
    ['pos', 'pos2', 'tri', 'edge', 'res', 'dep', 'dupsrc'].map(get),
  );
  return {
    pos: new Float32Array(pos),
    pos2: new Float32Array(pos2),
    tri: new Uint32Array(tri),
    edge: new Uint32Array(edge),
    res: new Float32Array(res),
    dep: new Float32Array(dep),
    dupsrc: new Uint32Array(dupsrc),
  };
}

export async function loadCoastline(): Promise<CoastBuffers> {
  const [pos, pos2] = await Promise.all([
    fetchBuffer('/api/coastline/pos'),
    fetchBuffer('/api/coastline/pos2'),
  ]);
  return { pos: new Float32Array(pos), pos2: new Float32Array(pos2) };
}

export function totalBytes(meta: MeshMeta): number {
  return Object.values(meta.bytes).reduce((a, b) => a + b, 0);
}

// ---------------------------------------------------------- project (M2)

export interface LayerData {
  type: string;
  name?: string;
  enabled?: boolean;
  mode?: string;
  [key: string]: unknown;
}

export interface ProjectData {
  path?: string;
  name: string;
  topo?: string | null;
  grid?: { n_lat?: number | null; n_lon?: number | null };
  layers: LayerData[];
  [key: string]: unknown;
}

const jsonInit = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
});

export const openProject = (path: string) =>
  fetchJson<ProjectData>('/api/project/open', jsonInit('POST', { path }));
export const getProject = () => fetchJson<ProjectData>('/api/project');
export const putProject = (proj: ProjectData) =>
  fetchJson<ProjectData>('/api/project', jsonInit('PUT', proj));
export const saveRegion = (name: string, geojson: object) =>
  fetchJson<{ path: string }>(`/api/project/regions/${name}`, jsonInit('POST', geojson));

export interface Estimate {
  elements_including_land: number;
  ocean_nodes: number;
  warnings: string[];
}

export const fetchEstimate = () => fetchJson<Estimate>('/api/project/estimate');

// ------------------------------------------------------ refinement (M3)

export interface RefineParams {
  polygon: [number, number][]; // (lon, lat) ring
  target_km: number;
  transition_km?: number | null;
  /** Topography NetCDF path: re-runs the land cut inside the region, so the
   * coastline is re-resolved from data instead of keeping the old polyline. */
  topo?: string | null;
  out_dir?: string;
}

export interface RefineReport {
  watertight_seam: boolean;
  n_nodes_after: number;
  n_tri_after: number;
  n_new_interior_nodes: number;
  interior_resolution_km: number | null;
  target_km: number;
  /** Finest sizing the frozen-seam transition band allows inside the region. */
  reachable_min_km: number;
  target_reached: boolean;
  [key: string]: unknown;
}

export interface JobSnapshot {
  id: string;
  status: 'running' | 'done' | 'error' | 'cancelled';
  stage: string;
  fraction: number;
  log_tail: string[];
  result: { mesh_id?: string; out_dir?: string; report?: RefineReport } | null;
  error: string | null;
}

export const refineMesh = (meshId: string, params: RefineParams) =>
  fetchJson<{ job_id: string; out_dir: string }>(
    `/api/mesh/${meshId}/refine`,
    jsonInit('POST', params),
  );

export const getJob = (jobId: string) => fetchJson<JobSnapshot>(`/api/jobs/${jobId}`);

export const getMeshMeta = (meshId: string) =>
  fetchJson<MeshMeta>(`/api/mesh/${meshId}/meta`);

export interface HfunGridData {
  values: Float32Array;
  nLat: number;
  nLon: number;
  min: number;
  max: number;
}

export async function fetchHfun(): Promise<HfunGridData> {
  const r = await fetch('/api/project/hfun');
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(`hfun: ${r.status} ${detail}`);
  }
  return {
    values: new Float32Array(await r.arrayBuffer()),
    nLat: Number(r.headers.get('x-hfun-nlat')),
    nLon: Number(r.headers.get('x-hfun-nlon')),
    min: Number(r.headers.get('x-hfun-min')),
    max: Number(r.headers.get('x-hfun-max')),
  };
}

// ---------------------------------------------------------- editing (M4)

export interface EditJournalState {
  can_undo: boolean;
  can_redo: boolean;
  n_ops: number;
}

/** Positions to write into the GPU buffers (vertex indices incl. dateline
 * duplicates; server applies all buffer conventions). */
export interface VertexUpdate {
  verts: number[];
  pos: [number, number, number][];
  pos2: [number, number, number][];
  dep: number[] | null;
}

export interface EditOpReport {
  n_nodes: number;
  n_elements: number;
  max_displacement_km: number;
  min_angle_before_deg: number | null;
  min_angle_after_deg: number | null;
  [key: string]: unknown;
}

export interface EditOpResult {
  report: EditOpReport;
  update: VertexUpdate;
  journal: EditJournalState;
}

/** A 422 from move/smooth: the op would invert these elements. */
export class EditRejected extends Error {
  constructor(
    message: string,
    readonly elementIds: number[],
  ) {
    super(message);
  }
}

async function postEdit<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, jsonInit('POST', body));
  if (r.ok) return r.json() as Promise<T>;
  let detail: unknown = r.statusText;
  try {
    detail = (await r.json()).detail ?? detail;
  } catch {
    /* not json */
  }
  if (typeof detail === 'object' && detail !== null && 'message' in detail) {
    const d = detail as { message: string; element_ids?: number[] };
    throw new EditRejected(d.message, d.element_ids ?? []);
  }
  throw new Error(`${url}: ${r.status} ${String(detail)}`);
}

export const editOpen = (meshId: string, topo?: string) =>
  postEdit<{ n_nodes: number; has_depth: boolean; journal: EditJournalState }>(
    `/api/mesh/${meshId}/edit/open`,
    { topo: topo ?? null },
  );

export const editSelectLasso = (meshId: string, lasso: [number, number][]) =>
  postEdit<{ ids: number[] }>(`/api/mesh/${meshId}/edit/select`, { lasso });

export const editSelectBrush = (meshId: string, center: [number, number], radiusKm: number) =>
  postEdit<{ ids: number[] }>(`/api/mesh/${meshId}/edit/select`, {
    center,
    radius_km: radiusKm,
  });

export const editMove = (meshId: string, ids: number[], lonlat: [number, number][]) =>
  postEdit<EditOpResult>(`/api/mesh/${meshId}/edit/move`, { ids, lonlat });

export const editSmooth = (
  meshId: string,
  ids: number[],
  iterations: number,
  strength: number,
) =>
  postEdit<EditOpResult>(`/api/mesh/${meshId}/edit/smooth`, { ids, iterations, strength });

export const editUndo = (meshId: string) =>
  postEdit<{ update: VertexUpdate | null; journal: EditJournalState }>(
    `/api/mesh/${meshId}/edit/undo`,
    {},
  );

export const editRedo = (meshId: string) =>
  postEdit<{ update: VertexUpdate | null; journal: EditJournalState }>(
    `/api/mesh/${meshId}/edit/redo`,
    {},
  );

export const editSave = (meshId: string, outDir: string) =>
  postEdit<{ out_dir: string; mesh_id: string; hash: string }>(
    `/api/mesh/${meshId}/edit/save`,
    { out_dir: outDir },
  );
