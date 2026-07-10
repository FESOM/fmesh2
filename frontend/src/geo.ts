/** Pure geometry for picking and drawing: viewer-space <-> lon/lat.
 *
 * Conventions (must match the buffer builder): sphere position is
 * (cos(lat)cos(lon), sin(lat), -cos(lat)sin(lon)) on the unit sphere;
 * plane position is (lon_rad, lat_rad, 0).
 */

export const EARTH_RADIUS_KM = 6371;
const D2R = Math.PI / 180;

export interface LonLat {
  lon: number;
  lat: number;
}

export function wrapLon(lon: number): number {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

export function lonLatToSphere(lon: number, lat: number): [number, number, number] {
  const cl = Math.cos(lat * D2R);
  return [cl * Math.cos(lon * D2R), Math.sin(lat * D2R), -cl * Math.sin(lon * D2R)];
}

export function lonLatToPlane(lon: number, lat: number): [number, number, number] {
  return [lon * D2R, lat * D2R, 0];
}

export function sphereToLonLat(x: number, y: number, z: number): LonLat {
  const r = Math.hypot(x, y, z) || 1;
  return {
    lon: wrapLon(Math.atan2(-z, x) / D2R),
    lat: Math.asin(Math.min(1, Math.max(-1, y / r))) / D2R,
  };
}

export function planeToLonLat(x: number, y: number): LonLat {
  return { lon: wrapLon(x / D2R), lat: Math.min(90, Math.max(-90, y / D2R)) };
}

export function haversineKm(a: LonLat, b: LonLat): number {
  const s1 = Math.sin(((b.lat - a.lat) * D2R) / 2);
  const s2 = Math.sin(((b.lon - a.lon) * D2R) / 2);
  const h = s1 * s1 + Math.cos(a.lat * D2R) * Math.cos(b.lat * D2R) * s2 * s2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** Close the polygon when a new click lands near the first vertex.
 * The threshold scales with camera altitude so it tracks ~12 px on screen. */
export function closeThresholdKm(cameraDistanceToSurface: number): number {
  return Math.max(5, cameraDistanceToSurface * EARTH_RADIUS_KM * 0.02);
}

export function shouldClosePolygon(
  vertices: LonLat[],
  candidate: LonLat,
  thresholdKm: number,
): boolean {
  return vertices.length >= 3 && haversineKm(vertices[0], candidate) < thresholdKm;
}

/** GeoJSON Polygon feature from drawn vertices (closes the ring). */
export function polygonGeoJSON(vertices: LonLat[]): object {
  const ring = vertices.map((v) => [v.lon, v.lat]);
  ring.push([vertices[0].lon, vertices[0].lat]);
  return {
    type: 'FeatureCollection',
    features: [
      { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [ring] } },
    ],
  };
}

/** Keep a drawn ring on one side of the dateline seam: unwrap each vertex
 * next to its predecessor (matches how the region layer reads rings). */
export function unwrapRing(vertices: LonLat[]): LonLat[] {
  const out: LonLat[] = [];
  let prev = 0;
  for (const [i, v] of vertices.entries()) {
    let lon = v.lon;
    if (i > 0) {
      while (lon - prev > 180) lon -= 360;
      while (lon - prev < -180) lon += 360;
    }
    out.push({ lon, lat: v.lat });
    prev = lon;
  }
  return out;
}
