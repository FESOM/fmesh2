/** Edit-mode node selection (pure model, no three.js).
 *
 * Tracks selected NODE ids (< nodes) and translates them to the GPU vertex
 * indices that must be flagged/moved: the node's own vertex plus its
 * dateline-seam duplicate when one exists (vertex `nodes + i` duplicates
 * node `dupsrc[i]`).
 */
export class Selection {
  private readonly set = new Set<number>();
  private readonly dupVertOf = new Map<number, number>();
  private readonly dupsrc: Uint32Array;

  constructor(
    readonly nodes: number,
    dupsrc: Uint32Array,
  ) {
    this.dupsrc = dupsrc;
    for (let i = 0; i < dupsrc.length; i++) this.dupVertOf.set(dupsrc[i], nodes + i);
  }

  /** Source node id of any picked vertex (duplicates map back). */
  nodeOfVertex(vid: number): number {
    return vid < this.nodes ? vid : this.dupsrc[vid - this.nodes];
  }

  /** Vertex indices representing a node on the GPU (itself + duplicate). */
  vertsOf(node: number): number[] {
    const dup = this.dupVertOf.get(node);
    return dup === undefined ? [node] : [node, dup];
  }

  get size(): number {
    return this.set.size;
  }

  has(node: number): boolean {
    return this.set.has(node);
  }

  /** Selected node ids, ascending. */
  ids(): number[] {
    return [...this.set].sort((a, b) => a - b);
  }

  /** Add nodes; returns the vertex indices that changed state. */
  add(nodes: Iterable<number>): number[] {
    const changed: number[] = [];
    for (const n of nodes) {
      if (this.set.has(n)) continue;
      this.set.add(n);
      changed.push(...this.vertsOf(n));
    }
    return changed;
  }

  /** Remove nodes; returns the vertex indices that changed state. */
  remove(nodes: Iterable<number>): number[] {
    const changed: number[] = [];
    for (const n of nodes) {
      if (!this.set.delete(n)) continue;
      changed.push(...this.vertsOf(n));
    }
    return changed;
  }

  /** Toggle one node; returns its verts and whether it is now selected. */
  toggle(node: number): { selected: boolean; verts: number[] } {
    if (this.set.has(node)) {
      this.set.delete(node);
      return { selected: false, verts: this.vertsOf(node) };
    }
    this.set.add(node);
    return { selected: true, verts: this.vertsOf(node) };
  }

  /** Clear everything; returns the vertex indices that were selected. */
  clear(): number[] {
    const verts: number[] = [];
    for (const n of this.set) verts.push(...this.vertsOf(n));
    this.set.clear();
    return verts;
  }
}
