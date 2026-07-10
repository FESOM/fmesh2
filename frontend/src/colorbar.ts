import { viridis } from './colormap';

export function initColorbar(): void {
  const canvas = document.getElementById('cbc') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(256, 1);
  for (let x = 0; x < 256; x++) {
    const [r, g, b] = viridis(x / 255);
    img.data.set([r, g, b, 255], x * 4);
  }
  ctx.putImageData(img, 0, 0);
  document.getElementById('cbar')!.style.display = 'block';
}

export function updateColorbar(title: string, lo: number, hi: number, digits: number): void {
  document.getElementById('cbtitle')!.innerHTML = `<b>${title}</b>`;
  document.getElementById('cbmin')!.textContent = lo.toFixed(digits);
  document.getElementById('cbmax')!.textContent = hi.toFixed(digits);
}
