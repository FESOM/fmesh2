import {
  fetchEstimate,
  getProject,
  openProject,
  putProject,
  saveRegion,
  type LayerData,
  type ProjectData,
} from '../api';

/** Right-hand project panel: ordered layer stack with per-type parameter
 * forms, add/remove/reorder/toggle, and the live size-estimate readout.
 * Every mutation PUTs the whole project (the server persists project.yaml)
 * and signals the caller so the hfun overlay can refresh. */

const NUMERIC_FIELDS: Record<string, string[]> = {
  base: ['resolution'],
  mercator: [
    'resolution',
    'northern_boundary',
    'northern_lowest_resolution',
    'southern_boundary',
    'southern_lowest_resolution',
  ],
  coast: ['min_resolution', 'max_distance', 'min_length', 'averaging'],
  bathy: ['min_resolution', 'min_depth', 'max_depth'],
  region: ['resolution', 'transition_km'],
  field: [],
};

const NEW_LAYER: Record<string, LayerData> = {
  base: { type: 'base', resolution: 300 },
  mercator: { type: 'mercator', resolution: 300, mode: 'replace' },
  coast: { type: 'coast', min_resolution: 25, max_distance: 200 },
  bathy: { type: 'bathy', min_resolution: 25 },
};

export interface LayerPanel {
  reload(): Promise<void>;
  addRegionLayer(vertices: { lon: number; lat: number }[]): Promise<void>;
  hasProject(): boolean;
}

export function initLayerPanel(opts: { onHfunStale: () => void }): LayerPanel {
  const root = document.createElement('div');
  root.id = 'layers';
  root.className = 'panel';
  document.body.appendChild(root);

  let project: ProjectData | null = null;
  let estimateTimer: number | undefined;

  async function reload(): Promise<void> {
    try {
      project = await getProject();
    } catch {
      project = null;
    }
    render();
    scheduleEstimate();
  }

  async function commit(): Promise<void> {
    if (!project) return;
    try {
      project = await putProject(project);
    } catch (err) {
      status(String(err));
    }
    render();
    scheduleEstimate();
    opts.onHfunStale();
  }

  function scheduleEstimate(): void {
    clearTimeout(estimateTimer);
    if (!project) return;
    estimateTimer = window.setTimeout(async () => {
      const el = document.getElementById('estimate');
      if (!el) return;
      if (!project?.layers.length) {
        el.textContent = 'no layers yet';
        return;
      }
      try {
        const est = await fetchEstimate();
        el.innerHTML =
          `≈ <b>${est.elements_including_land.toLocaleString()}</b> elements incl. land · ` +
          `≈ <b>${est.ocean_nodes.toLocaleString()}</b> ocean nodes` +
          est.warnings.map((w) => `<div class="warn">⚠ ${w}</div>`).join('');
      } catch (err) {
        el.textContent = String(err);
      }
    }, 500);
  }

  function status(msg: string): void {
    const el = document.getElementById('projstatus');
    if (el) el.textContent = msg;
  }

  function layerRow(layer: LayerData, i: number): HTMLElement {
    const row = document.createElement('div');
    row.className = 'layer-row';
    const head = document.createElement('div');
    head.className = 'layer-head';

    const enable = document.createElement('input');
    enable.type = 'checkbox';
    enable.checked = layer.enabled !== false;
    enable.onchange = () => {
      layer.enabled = enable.checked;
      void commit();
    };

    const label = document.createElement('span');
    label.textContent = ` ${i + 1}. ${layer.type}${layer.name ? ` (${layer.name})` : ''} `;
    label.style.flex = '1';

    const btn = (txt: string, fn: () => void, title = ''): HTMLButtonElement => {
      const b = document.createElement('button');
      b.textContent = txt;
      b.title = title;
      b.onclick = fn;
      return b;
    };
    head.append(
      enable,
      label,
      btn('↑', () => {
        if (i === 0 || !project) return;
        [project.layers[i - 1], project.layers[i]] = [project.layers[i], project.layers[i - 1]];
        void commit();
      }),
      btn('↓', () => {
        if (!project || i === project.layers.length - 1) return;
        [project.layers[i + 1], project.layers[i]] = [project.layers[i], project.layers[i + 1]];
        void commit();
      }),
      btn('✕', () => {
        project?.layers.splice(i, 1);
        void commit();
      }),
    );
    row.appendChild(head);

    const form = document.createElement('div');
    form.className = 'layer-form';
    for (const key of NUMERIC_FIELDS[layer.type] ?? []) {
      const lab = document.createElement('label');
      lab.textContent = key.replaceAll('_', ' ');
      const input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      const val = layer[key];
      input.value = val == null ? '' : String(val);
      input.onchange = () => {
        layer[key] = input.value === '' ? null : Number(input.value);
        void commit();
      };
      lab.appendChild(input);
      form.appendChild(lab);
    }
    const modeSel = document.createElement('label');
    modeSel.textContent = 'mode';
    const sel = document.createElement('select');
    for (const m of ['min', 'replace']) {
      const o = document.createElement('option');
      o.value = m;
      o.textContent = m;
      sel.appendChild(o);
    }
    sel.value = (layer.mode as string) ?? 'min';
    sel.onchange = () => {
      layer.mode = sel.value;
      void commit();
    };
    modeSel.appendChild(sel);
    form.appendChild(modeSel);
    row.appendChild(form);
    return row;
  }

  function render(): void {
    root.innerHTML = '';
    const title = document.createElement('div');
    title.innerHTML = `<b>project</b> `;
    const open = document.createElement('button');
    open.id = 'bOpenProject';
    open.textContent = project ? project.name : 'open project…';
    open.onclick = async () => {
      const path = prompt('Project directory (created if empty):', project?.path ?? '');
      if (!path) return;
      try {
        project = await openProject(path);
        render();
        scheduleEstimate();
        opts.onHfunStale();
      } catch (err) {
        status(String(err));
      }
    };
    title.appendChild(open);
    root.appendChild(title);
    const st = document.createElement('div');
    st.id = 'projstatus';
    st.className = 'dim';
    root.appendChild(st);
    if (!project) {
      st.textContent = 'no project open — layers and generation live here';
      return;
    }

    for (const [i, layer] of project.layers.entries()) root.appendChild(layerRow(layer, i));

    const add = document.createElement('div');
    add.style.marginTop = '6px';
    const sel = document.createElement('select');
    for (const t of Object.keys(NEW_LAYER)) {
      const o = document.createElement('option');
      o.value = t;
      o.textContent = `+ ${t}`;
      sel.appendChild(o);
    }
    const addBtn = document.createElement('button');
    addBtn.textContent = 'add layer';
    addBtn.onclick = () => {
      project?.layers.push({ ...NEW_LAYER[sel.value] });
      void commit();
    };
    add.append(sel, addBtn);
    root.appendChild(add);

    const est = document.createElement('div');
    est.id = 'estimate';
    est.style.marginTop = '6px';
    root.appendChild(est);
  }

  async function addRegionLayer(vertices: { lon: number; lat: number }[]): Promise<void> {
    if (!project) {
      status('open a project before drawing regions');
      return;
    }
    const n = project.layers.filter((l) => l.type === 'region').length + 1;
    const name = `drawn-${n}`;
    const { polygonGeoJSON, unwrapRing } = await import('../geo');
    const saved = await saveRegion(name, polygonGeoJSON(unwrapRing(vertices)) as object);
    project.layers.push({
      type: 'region',
      name,
      path: saved.path,
      resolution: 50,
      transition_km: 200,
    });
    await commit();
  }

  void reload();
  return { reload, addRegionLayer, hasProject: () => project !== null };
}
