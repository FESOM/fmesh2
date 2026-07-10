import { describe, expect, it } from 'vitest';
import {
  applyDragKm,
  dragDeltaKm,
  haversineKm,
  lonLatToSphere,
  type LonLat,
} from '../geo';
import { Selection } from '../selection';

describe('tangent-plane drag math', () => {
  it('east displacement magnitude is preserved at various latitudes', () => {
    for (const lat of [0, 45, 60, -70]) {
      const p: LonLat = { lon: 12, lat };
      const moved = applyDragKm(p, 50, 0);
      expect(haversineKm(p, moved)).toBeCloseTo(50, 1);
      expect(moved.lat).toBe(lat);
    }
  });

  it('north displacement magnitude is latitude-independent', () => {
    for (const lat of [0, 45, -60]) {
      const p: LonLat = { lon: -30, lat };
      const moved = applyDragKm(p, 0, 80);
      expect(haversineKm(p, moved)).toBeCloseTo(80, 3);
      expect(moved.lon).toBe(p.lon);
    }
  });

  it('a longitude step needs more degrees at high latitude', () => {
    const low = applyDragKm({ lon: 0, lat: 0 }, 100, 0);
    const high = applyDragKm({ lon: 0, lat: 60 }, 100, 0);
    expect(high.lon).toBeCloseTo(low.lon / Math.cos(Math.PI / 3), 6);
  });

  it('result stays on the unit sphere', () => {
    const moved = applyDragKm({ lon: 170, lat: 55 }, 300, -120);
    const [x, y, z] = lonLatToSphere(moved.lon, moved.lat);
    expect(Math.hypot(x, y, z)).toBeCloseTo(1, 9);
  });

  it('wraps across the dateline and clamps at the poles', () => {
    const wrapped = applyDragKm({ lon: 179.5, lat: 0 }, 120, 0);
    expect(wrapped.lon).toBeLessThan(-179);
    const polar = applyDragKm({ lon: 0, lat: 89.9 }, 0, 100);
    expect(polar.lat).toBe(90);
  });

  it('delta/apply round-trips through the anchor point', () => {
    const anchor: LonLat = { lon: -40, lat: 30 };
    const current: LonLat = { lon: -39.2, lat: 30.6 };
    const d = dragDeltaKm(anchor, current);
    const back = applyDragKm(anchor, d.eastKm, d.northKm);
    expect(back.lon).toBeCloseTo(current.lon, 9);
    expect(back.lat).toBeCloseTo(current.lat, 9);
  });

  it('delta measures east/north in km at the anchor latitude', () => {
    const d = dragDeltaKm({ lon: 0, lat: 60 }, { lon: 1, lat: 60 });
    expect(d.northKm).toBe(0);
    expect(d.eastKm).toBeCloseTo(((Math.PI / 180) * 6371) / 2, 3); // cos 60 = 0.5
  });
});

describe('selection model', () => {
  // 5 nodes; node 3 has a dateline duplicate at vertex 5
  const sel = () => new Selection(5, new Uint32Array([3]));

  it('maps picked duplicate vertices back to their source node', () => {
    const s = sel();
    expect(s.nodeOfVertex(2)).toBe(2);
    expect(s.nodeOfVertex(5)).toBe(3);
    expect(s.vertsOf(3)).toEqual([3, 5]);
    expect(s.vertsOf(1)).toEqual([1]);
  });

  it('add/remove report changed verts including duplicates', () => {
    const s = sel();
    expect(s.add([1, 3])).toEqual([1, 3, 5]);
    expect(s.add([1])).toEqual([]); // already selected
    expect(s.size).toBe(2);
    expect(s.ids()).toEqual([1, 3]);
    expect(s.remove([3, 4])).toEqual([3, 5]); // 4 was never selected
    expect(s.size).toBe(1);
  });

  it('toggle and clear', () => {
    const s = sel();
    expect(s.toggle(3)).toEqual({ selected: true, verts: [3, 5] });
    expect(s.toggle(3)).toEqual({ selected: false, verts: [3, 5] });
    s.add([0, 3]);
    expect(s.clear().sort()).toEqual([0, 3, 5]);
    expect(s.size).toBe(0);
  });
});
