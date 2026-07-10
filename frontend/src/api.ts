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
  const [pos, pos2, tri, edge, res, dep] = await Promise.all(
    ['pos', 'pos2', 'tri', 'edge', 'res', 'dep'].map(get),
  );
  return {
    pos: new Float32Array(pos),
    pos2: new Float32Array(pos2),
    tri: new Uint32Array(tri),
    edge: new Uint32Array(edge),
    res: new Float32Array(res),
    dep: new Float32Array(dep),
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
