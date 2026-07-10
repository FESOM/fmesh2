export interface MeshMeta {
  name: string;
  nodes: number;
  verts: number;
  triangles: number;
  edges: number;
  res_range: [number, number];
  dep_range: [number, number];
  bytes: Record<string, number>;
}

export interface MeshBuffers {
  pos: Float32Array;
  pos2: Float32Array;
  tri: Uint32Array;
  edge: Uint32Array;
  res: Float32Array;
  dep: Float32Array;
}

export interface CoastBuffers {
  pos: Float32Array;
  pos2: Float32Array;
}
