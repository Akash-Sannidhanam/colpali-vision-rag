import { deflateSync } from 'node:zlib'
import type { Page } from '@playwright/test'

// --- A real PNG, because the thing under test is geometry ---
//
// The fixture's only job is to carry an intrinsic size the browser will believe, so an
// SVG would have been fewer lines. It is a raster PNG anyway: production serves PNGs, and
// an SVG without a viewBox is "scalable" to the layout engine in ways a page image is not.
// Substituting a different intrinsic-sizing model into the one test whose subject *is*
// intrinsic sizing would quietly stop testing the real thing.

const CRC_TABLE = (() => {
  const table = new Int32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c
  }
  return table
})()

function crc32(buf: Buffer): number {
  let c = ~0
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8)
  return ~c >>> 0
}

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4)
  length.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([length, body, crc])
}

/** A valid 8-bit grayscale PNG of exactly `w` x `h`. Uniform white; it is never looked at. */
export function png(w: number, h: number): Buffer {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(w, 0)
  ihdr.writeUInt32BE(h, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 0 // colour type: grayscale
  // bytes 10-12 stay 0: compression, filter, interlace
  // Each scanline is a filter byte (0 = None) followed by one byte per pixel.
  const raw = Buffer.alloc((w + 1) * h, 0xff)
  for (let y = 0; y < h; y++) raw[y * (w + 1)] = 0
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

export const PDF = 'fixture.pdf'
export const PAGE_COUNT = 3

/**
 * Stub every endpoint the app touches on the way to the document viewer, and serve page
 * images at a chosen intrinsic size.
 *
 * Network-level stubs rather than a live backend on purpose: this test is about how the
 * browser lays out a page image, and standing up ColQwen2 + Qdrant + Gemini in CI to
 * answer that would be several minutes and an API key for a question none of them
 * influence.
 */
export async function mockBackend(page: Page, size: { w: number; h: number }): Promise<void> {
  await page.route('**/health', (route) =>
    route.fulfill({ json: { status: 'ok', model_loaded: true, qdrant: 'ok', corpus: 'ok' } }),
  )
  await page.route('**/corpus', (route) =>
    route.fulfill({
      json: {
        documents: [{ pdf: PDF, page_count: PAGE_COUNT }],
        total_pages: PAGE_COUNT,
        qdrant: 'ok',
      },
    }),
  )
  await page.route('**/corpus/*/pages', (route) =>
    route.fulfill({
      json: {
        pdf: PDF,
        page_count: PAGE_COUNT,
        has_original: true,
        pages: Array.from({ length: PAGE_COUNT }, (_, i) => ({
          page_number: i + 1,
          image: { url: `/images/fixture_page_${i + 1}.png`, data_uri: null },
        })),
      },
    }),
  )
  await page.route('**/images/**', (route) =>
    route.fulfill({ contentType: 'image/png', body: png(size.w, size.h) }),
  )
}
