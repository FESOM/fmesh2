/**
 * M4 acceptance browser driver (invoked by experiments/m4_acceptance/runner.py).
 *
 * Drives the real edit-mode UI on the refined strait mesh: brush-select
 * coastal nodes, tangent-plane drag (commit), smooth, undo/redo, save-as.
 * Screen->lon/lat mapping assumes the untouched map-mode camera
 * (position (0,0,3.2), fov 45), so every interaction happens before any
 * zooming; documentation zoom-in comes after the ops are committed.
 *
 * Not a @playwright/test spec on purpose — it needs env-driven state and
 * ordered side effects; the smoke spec stays the only `npm run e2e` test.
 */
import { chromium } from 'playwright-core';

const BASE = process.env.FMESH2_M4_BASE ?? 'http://127.0.0.1:8323';
const MESH = process.env.FMESH2_M4_MESH ?? 'm2';
const OUT = process.env.FMESH2_M4_OUT;
const FIGDIR = process.env.FMESH2_M4_FIGDIR ?? '.';
if (!OUT) throw new Error('FMESH2_M4_OUT is required');

const W = 2000;
const H = 1200;
// default map camera: z = 3.2, vertical fov 45 deg, plane units = radians
const halfH = 3.2 * Math.tan((45 / 2) * (Math.PI / 180));
const halfW = halfH * (W / H);
const D2R = Math.PI / 180;
const px = (lon, lat) => ({
  x: (((lon * D2R) / halfW + 1) / 2) * W,
  y: ((1 - (lat * D2R) / halfH) / 2) * H,
});

async function waitText(page, sel, substr, timeout = 30_000) {
  const t0 = Date.now();
  for (;;) {
    const text = (await page.locator(sel).textContent()) ?? '';
    if (text.includes(substr)) return text.trim();
    if (Date.now() - t0 > timeout)
      throw new Error(`timeout waiting for "${substr}" in ${sel}; last: "${text}"`);
    await page.waitForTimeout(200);
  }
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: W, height: H } });
await page.goto(`${BASE}/?mesh=${MESH}`);
await page.locator('#hud').waitFor({ state: 'visible', timeout: 90_000 });

// map view (deterministic screen<->lon/lat), let the morph settle
await page.keyboard.press('g');
await page.waitForTimeout(2500);

// enter edit mode, wait for the live session
await page.keyboard.press('e');
const open = await waitText(page, '#editStatus', 'session open');
console.log(`EDIT_OPEN ${open}`);

// brush-select along the Denmark Strait coast (radius 120 km)
await page.click('#bToolBrush');
await page.fill('#editBrushR', '120');
const stroke = [px(-26.5, 67.0), px(-25.8, 66.7), px(-25.1, 66.4)];
await page.mouse.move(stroke[0].x, stroke[0].y);
await page.mouse.down();
for (const p of stroke.slice(1)) {
  await page.waitForTimeout(200); // brush applies are throttled at 120 ms
  await page.mouse.move(p.x, p.y, { steps: 3 });
}
await page.waitForTimeout(400);
await page.mouse.up();
const selText = await page.locator('#editSelCount').textContent();
const nSel = Number(selText.match(/\d+/)?.[0] ?? 0);
console.log(`SELECTED ${nSel}`);
if (nSel < 5) throw new Error(`brush selected too few nodes: "${selText}"`);
await page.screenshot({ path: `${FIGDIR}/m4-ui-selection.png` });

// drag the selection ~20 km east-north (tangent-plane move, then commit)
await page.click('#bToolSel');
const grab = stroke[1];
await page.mouse.move(grab.x, grab.y);
await page.mouse.down();
await page.mouse.move(grab.x + 3, grab.y - 1, { steps: 4 });
await page.waitForTimeout(150);
await page.mouse.up();
console.log(`MOVE ${await waitText(page, '#editStatus', 'moved')}`);

// smooth the same selection
await page.click('#bSmoothSel');
console.log(`SMOOTH ${await waitText(page, '#editStatus', 'smoothed')}`);

// undo / redo through the journal
await page.keyboard.press('Meta+z');
console.log(`UNDO ${await waitText(page, '#editStatus', 'undid')}`);
await page.keyboard.press('Meta+Shift+z');
console.log(`REDO ${await waitText(page, '#editStatus', 'redid')}`);

// zoom into the strait for a readable figure (ops are all committed)
const center = px(-25.8, 66.7);
await page.mouse.move(center.x, center.y);
for (let i = 0; i < 6; i++) {
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(150);
}
await page.waitForTimeout(500);
await page.screenshot({ path: `${FIGDIR}/m4-ui-after-edit.png` });

// save-as: FESOM triplet + edits.jsonl, auto-registered in the viewer
await page.fill('#editOut', OUT);
await page.click('#bSaveAs');
console.log(`SAVE ${await waitText(page, '#editStatus', 'saved →', 120_000)}`);
await page.screenshot({ path: `${FIGDIR}/m4-ui-saved.png` });

await browser.close();
console.log('DRIVER_OK');
