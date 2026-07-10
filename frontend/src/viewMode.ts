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

export function controlsConfig(mode: ViewMode): ControlsConfig {
  if (mode === 0) {
    return {
      enableRotate: true,
      leftIsPan: false,
      minDistance: 1.0003,
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

/** Camera near/far from the distance-to-surface d (globe) or target (map). */
export function nearFar(d: number): { near: number; far: number } {
  return { near: Math.max(1e-5, d * 0.03), far: d + 6 };
}

/** Rotate speed scaled with altitude so close-up orbiting stays controllable. */
export function rotateSpeed(d: number): number {
  return Math.min(1, Math.max(0.02, d * 0.9));
}
