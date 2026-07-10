import * as THREE from 'three';
import { fetchJson } from './api';
import type { MeshBuffers } from './types';

export function normalizeLon(lonDeg: number): number {
  return ((((lonDeg + 180) % 360) + 360) % 360) - 180;
}

export function nodeHtml(
  idx: number,
  lon: number,
  lat: number,
  res: number,
  dep: number,
  extra = '',
): string {
  return (
    `<b>node ${idx}</b><br>` +
    `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? 'N' : 'S'} ` +
    `${Math.abs(lon).toFixed(4)}°${lon >= 0 ? 'E' : 'W'}<br>` +
    `resolution ${res.toFixed(2)} km<br>depth ${dep.toFixed(0)} m${extra}`
  );
}

interface NodeInfo {
  index: number;
  lon: number;
  lat: number;
  res_km: number;
  depth_m: number;
  neighbors: number[];
}

export class Inspector {
  private readonly panel = document.getElementById('inspect')!;
  readonly marker: THREE.Points;
  private readonly markerPos: THREE.BufferAttribute;
  private readonly markerPos2: THREE.BufferAttribute;
  private pinned = false;

  constructor(
    private readonly meshId: string,
    private readonly buffers: MeshBuffers,
    uMorph: { value: number },
  ) {
    const geo = new THREE.BufferGeometry();
    this.markerPos = new THREE.BufferAttribute(new Float32Array(3), 3);
    this.markerPos2 = new THREE.BufferAttribute(new Float32Array(3), 3);
    geo.setAttribute('position', this.markerPos);
    geo.setAttribute('position2', this.markerPos2);
    this.marker = new THREE.Points(
      geo,
      new THREE.ShaderMaterial({
        uniforms: { uMorph },
        vertexShader: /* glsl */ `
          attribute vec3 position2; uniform float uMorph;
          void main() {
            gl_PointSize = 11.0;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(position, position2, uMorph), 1.0);
          }`,
        fragmentShader: /* glsl */ `
          void main() {
            if (length(gl_PointCoord - 0.5) > 0.5) discard;
            gl_FragColor = vec4(1.0, 0.55, 0.15, 1.0);
          }`,
        transparent: true,
        depthTest: false,
      }),
    );
    this.marker.renderOrder = 5;
    this.marker.visible = false;
    this.marker.frustumCulled = false;
  }

  private moveMarker(idx: number): void {
    for (let c = 0; c < 3; c++) {
      this.markerPos.setComponent(0, c, this.buffers.pos[idx * 3 + c]);
      this.markerPos2.setComponent(0, c, this.buffers.pos2[idx * 3 + c]);
    }
    this.markerPos.needsUpdate = true;
    this.markerPos2.needsUpdate = true;
    this.marker.visible = true;
  }

  private info(idx: number): { lon: number; lat: number; res: number; dep: number } {
    return {
      lon: normalizeLon((this.buffers.pos2[idx * 3] * 180) / Math.PI),
      lat: (this.buffers.pos2[idx * 3 + 1] * 180) / Math.PI,
      res: this.buffers.res[idx],
      dep: this.buffers.dep[idx],
    };
  }

  hover(idx: number): void {
    if (this.pinned) return;
    if (idx < 0) {
      this.panel.style.display = 'none';
      this.marker.visible = false;
      return;
    }
    const { lon, lat, res, dep } = this.info(idx);
    this.panel.innerHTML = nodeHtml(idx, lon, lat, res, dep);
    this.panel.style.display = 'block';
    this.moveMarker(idx);
  }

  async pin(idx: number): Promise<void> {
    if (idx < 0) {
      this.pinned = false;
      this.hover(-1);
      return;
    }
    this.pinned = true;
    this.moveMarker(idx);
    const { lon, lat, res, dep } = this.info(idx);
    this.panel.innerHTML = nodeHtml(idx, lon, lat, res, dep, '<br><span class="dim">…</span>');
    this.panel.style.display = 'block';
    try {
      const info = await fetchJson<NodeInfo>(`/api/mesh/${this.meshId}/node/${idx}`);
      this.panel.innerHTML = nodeHtml(
        idx, info.lon, info.lat, info.res_km, info.depth_m,
        `<br>${info.neighbors.length} neighbors` +
          `<br><span class="dim">click elsewhere to unpin</span>`,
      );
    } catch {
      /* keep local info */
    }
  }
}
