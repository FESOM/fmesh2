import { describe, expect, it } from 'vitest';
import { viridis } from '../colormap';
import {
  controlsConfig,
  MIN_ALTITUDE,
  nearFar,
  radialClampFactor,
  rotateSpeed,
} from '../viewMode';

describe('viridis', () => {
  it('matches the spike polynomial at the endpoints', () => {
    const [r0, g0, b0] = viridis(0);
    expect(r0).toBeGreaterThan(55);
    expect(r0).toBeLessThan(85);
    expect(g0).toBeLessThan(15);
    expect(b0).toBeGreaterThan(70);
    const [r1, g1, b1] = viridis(1);
    expect(r1).toBeGreaterThan(240);
    expect(g1).toBeGreaterThan(215);
    expect(b1).toBeLessThan(70);
  });

  it('clamps out-of-range input', () => {
    expect(viridis(-1)).toEqual(viridis(0));
    expect(viridis(2)).toEqual(viridis(1));
  });

  it('green channel rises monotonically', () => {
    let prev = -1;
    for (let t = 0; t <= 1.001; t += 0.05) {
      const g = viridis(t)[1];
      expect(g).toBeGreaterThanOrEqual(prev);
      prev = g;
    }
  });
});

describe('view mode state machine', () => {
  it('globe orbits, map pans', () => {
    expect(controlsConfig(0).enableRotate).toBe(true);
    expect(controlsConfig(0).leftIsPan).toBe(false);
    expect(controlsConfig(1).enableRotate).toBe(false);
    expect(controlsConfig(1).leftIsPan).toBe(true);
  });

  it('target-relative min distances stay tiny; the surface guard is radial', () => {
    // consistency of the zoom floor comes from radialClampFactor, not from
    // OrbitControls minDistance (whose target drifts under zoomToCursor)
    expect(controlsConfig(0).minDistance).toBeLessThan(0.001);
    expect(controlsConfig(1).minDistance).toBeLessThan(0.01);
  });

  it('radial clamp keeps the camera above the surface everywhere', () => {
    const minR = 1 + MIN_ALTITUDE;
    // inside the globe -> pushed back out to exactly the floor
    expect(0.98 * radialClampFactor(0.98)).toBeCloseTo(minR, 10);
    expect(1.00001 * radialClampFactor(1.00001)).toBeCloseTo(minR, 10);
    // at or above the floor -> untouched
    expect(radialClampFactor(minR)).toBe(1);
    expect(radialClampFactor(2.4)).toBe(1);
    // degenerate input never divides by zero
    expect(radialClampFactor(0)).toBe(1);
  });

  it('near plane tracks altitude', () => {
    expect(nearFar(1).near).toBeCloseTo(0.03);
    expect(nearFar(0.0001).near).toBeGreaterThanOrEqual(1e-5);
    expect(nearFar(2).far).toBeGreaterThan(2);
  });

  it('rotate speed clamps at both ends', () => {
    expect(rotateSpeed(10)).toBe(1);
    expect(rotateSpeed(0.00001)).toBe(0.02);
    expect(rotateSpeed(0.5)).toBeCloseTo(0.45);
  });
});
