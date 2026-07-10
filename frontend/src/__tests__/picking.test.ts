import { describe, expect, it } from 'vitest';
import { normalizeLon, nodeHtml } from '../inspect';
import { decodeId, encodeId, nearestIdInWindow, PICK_NONE } from '../picking';

describe('picking id codec', () => {
  it('roundtrips across the full range', () => {
    for (const id of [0, 1, 254, 255, 256, 65535, 65536, 7402885, 16777214]) {
      const [r, g, b] = encodeId(id);
      expect(decodeId(r, g, b)).toBe(id);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThan(256);
    }
  });

  it('cleared black pixel decodes to PICK_NONE', () => {
    expect(decodeId(0, 0, 0)).toBe(PICK_NONE);
  });
});

describe('nearestIdInWindow', () => {
  const w = 5;
  const put = (px: Uint8Array, x: number, y: number, id: number) => {
    const [r, g, b] = encodeId(id);
    px.set([r, g, b, 255], (y * w + x) * 4);
  };

  it('returns the id closest to the window center', () => {
    const px = new Uint8Array(w * w * 4);
    put(px, 0, 0, 7); // corner, far from center (2,2)
    put(px, 3, 2, 42); // one pixel from center
    expect(nearestIdInWindow(px, w)).toBe(42);
  });

  it('an id at the exact center beats everything', () => {
    const px = new Uint8Array(w * w * 4);
    put(px, 3, 2, 42);
    put(px, 2, 2, 9);
    expect(nearestIdInWindow(px, w)).toBe(9);
  });

  it('empty window returns PICK_NONE', () => {
    expect(nearestIdInWindow(new Uint8Array(w * w * 4), w)).toBe(PICK_NONE);
  });
});

describe('inspect formatting', () => {
  it('normalizes duplicated seam longitudes', () => {
    expect(normalizeLon(185)).toBe(-175);
    expect(normalizeLon(-185)).toBe(175);
    expect(normalizeLon(0)).toBe(0);
    expect(normalizeLon(180)).toBe(-180);
  });

  it('renders hemisphere-aware coordinates', () => {
    const html = nodeHtml(42, -70.5, 79.25, 4.52, 523);
    expect(html).toContain('node 42');
    expect(html).toContain('79.2500°N');
    expect(html).toContain('70.5000°W');
    expect(html).toContain('4.52 km');
    expect(html).toContain('523 m');
  });
});
