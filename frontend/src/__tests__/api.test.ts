import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchBuffer, fetchJson, totalBytes } from '../api';
import type { MeshMeta } from '../types';

function streamOf(...parts: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const p of parts) controller.enqueue(p);
      controller.close();
    },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchBuffer', () => {
  it('concatenates chunks and reports progress', async () => {
    const parts = [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])];
    vi.stubGlobal('fetch', async () => new Response(streamOf(...parts)));
    const seen: number[] = [];
    const buf = await fetchBuffer('/x', (n) => seen.push(n));
    expect(new Uint8Array(buf)).toEqual(new Uint8Array([1, 2, 3, 4, 5]));
    expect(seen.reduce((a, b) => a + b, 0)).toBe(5);
  });

  it('throws on http error', async () => {
    vi.stubGlobal('fetch', async () => new Response('no', { status: 500 }));
    await expect(fetchBuffer('/x')).rejects.toThrow('/x: 500');
  });
});

describe('fetchJson', () => {
  it('surfaces API error detail', async () => {
    vi.stubGlobal(
      'fetch',
      async () =>
        new Response(JSON.stringify({ detail: 'not a mesh' }), {
          status: 400,
          headers: { 'content-type': 'application/json' },
        }),
    );
    await expect(fetchJson('/api/meshes/open')).rejects.toThrow('not a mesh');
  });
});

it('totalBytes sums the meta byte table', () => {
  const meta = { bytes: { a: 10, b: 32 } } as unknown as MeshMeta;
  expect(totalBytes(meta)).toBe(42);
});
