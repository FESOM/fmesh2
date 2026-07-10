import { expect, test } from '@playwright/test';

test('viewer boots: WebGL2 canvas, HUD counts, project panel', async ({ page }) => {
  await page.goto('/');

  // mesh buffers stream in, then the HUD appears
  await expect(page.locator('#hud')).toBeVisible({ timeout: 90_000 });

  // the canvas is driven by a live WebGL2 context
  const isWebGL2 = await page.evaluate(() => {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    return canvas.getContext('webgl2') instanceof WebGL2RenderingContext;
  });
  expect(isWebGL2).toBe(true);

  // HUD shows real mesh counts
  await expect(page.locator('#counts')).toContainText('nodes');
  await expect(page.locator('#counts')).toContainText('tri');
  await expect(page.locator('#fps')).not.toHaveText('-- fps', { timeout: 15_000 });

  // M2 controls exist: project panel and draw/hfun buttons
  await expect(page.locator('#layers')).toBeVisible();
  await expect(page.locator('#layers')).toContainText('project');
  await expect(page.locator('#bDraw')).toBeVisible();
  await expect(page.locator('#bHfun')).toBeVisible();
});
