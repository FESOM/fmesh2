/** Pure view-mode state (globe = 0, Plate Carree map = 1) for testability. */

export type ViewMode = 0 | 1;

export interface ControlsConfig {
  enableRotate: boolean;
  leftIsPan: boolean;
  minDistance: number;
  maxDistance: number;
  cameraPos: [number, number, number];
  label: string;
  hint: string;
}

/** Minimum camera altitude above the globe surface (~2 km at Earth scale). */
export const MIN_ALTITUDE = 0.0003;

/** Minimum camera height above the Plate Carree map plane. */
export const MIN_MAP_HEIGHT = 0.0008;

export function controlsConfig(mode: ViewMode): ControlsConfig {
  // minDistance is deliberately tiny: OrbitControls measures it from the orbit
  // target, which zoomToCursor drifts away from the globe center — clamping on
  // it gives inconsistent zoom limits. The real surface guard is
  // radialClampFactor applied every frame against the globe center.
  if (mode === 0) {
    return {
      enableRotate: true,
      leftIsPan: false,
      minDistance: 0.0002,
      maxDistance: 5,
      cameraPos: [0, 0.7, 2.4],
      label: '[G] view: globe',
      hint: 'drag: orbit · wheel: zoom · hover: inspect',
    };
  }
  return {
    enableRotate: false,
    leftIsPan: true,
    minDistance: 0.0015,
    maxDistance: 7,
    cameraPos: [0, 0, 3.2],
    label: '[G] view: map',
    hint: 'drag: pan · wheel: zoom · hover: inspect',
  };
}

/**
 * Factor to rescale the camera position so it stays outside the globe;
 * returns 1 when no clamping is needed.
 */
export function radialClampFactor(r: number, minR: number = 1 + MIN_ALTITUDE): number {
  return r > 0 && r < minR ? minR / r : 1;
}

/** Camera near/far from the distance-to-surface d (globe) or target (map). */
export function nearFar(d: number): { near: number; far: number } {
  return { near: Math.max(1e-5, d * 0.03), far: d + 6 };
}

/** Rotate speed scaled with altitude so close-up orbiting stays controllable. */
export function rotateSpeed(d: number): number {
  return Math.min(1, Math.max(0.02, d * 0.9));
}
