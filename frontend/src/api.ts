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
