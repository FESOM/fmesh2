import { listMeshes, loadCoastline, loadMeshBuffers, totalBytes } from './api';

const loadMsg = document.getElementById('loadmsg')!;
const bar = document.querySelector<HTMLDivElement>('#bar > div')!;

async function boot() {
  loadMsg.textContent = 'looking for meshes…';
  const meshes = await listMeshes();
  if (!meshes.length) {
    loadMsg.innerHTML =
      'no mesh loaded — start with <code>fmesh2 /path/to/mesh_dir</code>';
    return;
  }
  const { id, meta } = meshes[0];
  loadMsg.textContent = `loading ${meta.name} (${(meta.nodes / 1e6).toFixed(2)}M nodes)…`;

  const total = totalBytes(meta);
  let got = 0;
  const onBytes = (n: number) => {
    got += n;
    bar.style.width = `${((100 * got) / total).toFixed(1)}%`;
  };
  const t0 = performance.now();
  const [buffers, coast] = await Promise.all([
    loadMeshBuffers(id, onBytes),
    loadCoastline().catch(() => null),
  ]);
  const fetchSec = (performance.now() - t0) / 1e3;

  const { startViewer } = await import('./scene');
  startViewer({ id, meta, buffers, coast, fetchSec });
}

boot().catch((err) => {
  loadMsg.textContent = String(err);
});
