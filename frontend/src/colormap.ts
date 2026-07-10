/** Viridis via the polynomial fit used by the validated render spike. */

const C: [number, number, number][] = [
  [0.2777, 0.0054, 0.334],
  [0.105, 1.4046, 1.3845],
  [-0.3308, 0.2148, 0.095],
  [-4.6342, -5.7991, -19.3324],
  [6.2282, 14.1799, 56.6905],
  [4.7763, -13.7451, -65.353],
  [-5.4354, 4.6456, 26.3125],
];

export function viridis(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const out: [number, number, number] = [0, 0, 0];
  for (let k = C.length - 1; k >= 0; k--) {
    for (let i = 0; i < 3; i++) out[i] = out[i] * x + C[k][i];
  }
  return out.map((v) => Math.max(0, Math.min(255, Math.round(v * 255)))) as [
    number,
    number,
    number,
  ];
}

export const VIRIDIS_GLSL = /* glsl */ `
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.2777, 0.0054, 0.3340), c1 = vec3(0.1050, 1.4046, 1.3845),
    c2 = vec3(-0.3308, 0.2148, 0.0950), c3 = vec3(-4.6342, -5.7991, -19.3324),
    c4 = vec3(6.2282, 14.1799, 56.6905), c5 = vec3(4.7763, -13.7451, -65.3530),
    c6 = vec3(-5.4354, 4.6456, 26.3125);
  return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}`;
