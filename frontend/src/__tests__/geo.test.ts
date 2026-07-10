import { describe, expect, it } from 'vitest';
import {
  closeThresholdKm,
  haversineKm,
  lonLatToPlane,
  lonLatToSphere,
  planeToLonLat,
  polygonGeoJSON,
  shouldClosePolygon,
  sphereToLonLat,
  unwrapRing,
  wrapLon,
} from '../geo';

describe('lon/lat <-> viewer space', () => {
  const cases = [
    { lon: 0, lat: 0 },
    { lon: 90, lat: 45 },
    { lon: -120, lat: -60 },
    { lon: 179.5, lat: 10 },
    { lon: -179.5, lat: -10 },
    { lon: 0, lat: 89.9 },
  ];

  it('sphere roundtrip', () => {
    for (const c of cases) {
      const [x, y, z] = lonLatToSphere(c.lon, c.lat);
      expect(Math.hypot(x, y, z)).toBeCloseTo(1, 12);
      const back = sphereToLonLat(x, y, z);
      expect(back.lon).toBeCloseTo(c.lon, 9);
      expect(back.lat).toBeCloseTo(c.lat, 9);
    }
  });

  it('matches the buffer-builder convention (Y north, Z west at lon 90)', () => {
    expect(lonLatToSphere(0, 90)[1]).toBeCloseTo(1);
    const [x, , z] = lonLatToSphere(90, 0);
    expect(x).toBeCloseTo(0);
    expect(z).toBeCloseTo(-1);
  });

  it('plane roundtrip incl. dateline', () => {
    for (const c of cases) {
      const [x, y] = lonLatToPlane(c.lon, c.lat);
      const back = planeToLonLat(x, y);
      expect(back.lon).toBeCloseTo(c.lon, 9);
      expect(back.lat).toBeCloseTo(c.lat, 9);
    }
    // +180 wraps onto -180
    expect(planeToLonLat(Math.PI, 0).lon).toBeCloseTo(-180);
  });

  it('wrapLon', () => {
    expect(wrapLon(190)).toBe(-170);
    expect(wrapLon(-190)).toBe(170);
    expect(wrapLon(540)).toBe(180 - 360);
    expect(wrapLon(45)).toBe(45);
  });
});

describe('polygon closure', () => {
  const square = [
    { lon: 0, lat: 0 },
    { lon: 10, lat: 0 },
    { lon: 10, lat: 10 },
  ];

  it('closes near the first vertex only with >= 3 vertices', () => {
    expect(shouldClosePolygon(square, { lon: 0.05, lat: 0.05 }, 50)).toBe(true);
    expect(shouldClosePolygon(square, { lon: 5, lat: 5 }, 50)).toBe(false);
    expect(shouldClosePolygon(square.slice(0, 2), { lon: 0, lat: 0 }, 50)).toBe(false);
  });

  it('threshold scales with altitude but has a floor', () => {
    expect(closeThresholdKm(1)).toBeCloseTo(6371 * 0.02);
    expect(closeThresholdKm(1e-5)).toBe(5);
  });

  it('haversine sanity: one degree of latitude', () => {
    expect(haversineKm({ lon: 0, lat: 0 }, { lon: 0, lat: 1 })).toBeCloseTo(111.19, 1);
  });
});

describe('ring export', () => {
  it('geojson ring is closed', () => {
    const gj = polygonGeoJSON([
      { lon: 0, lat: 0 },
      { lon: 10, lat: 0 },
      { lon: 10, lat: 10 },
    ]) as { features: { geometry: { coordinates: number[][][] } }[] };
    const ring = gj.features[0].geometry.coordinates[0];
    expect(ring).toHaveLength(4);
    expect(ring[0]).toEqual(ring[3]);
  });

  it('unwrapRing keeps dateline-crossing rings continuous', () => {
    const ring = unwrapRing([
      { lon: 170, lat: 0 },
      { lon: -170, lat: 0 }, // crosses the seam eastward
      { lon: -170, lat: 10 },
      { lon: 170, lat: 10 },
    ]);
    expect(ring.map((v) => v.lon)).toEqual([170, 190, 190, 170]);
    const west = unwrapRing([
      { lon: -170, lat: 0 },
      { lon: 170, lat: 0 }, // crosses westward
    ]);
    expect(west.map((v) => v.lon)).toEqual([-170, -190]);
  });
});
