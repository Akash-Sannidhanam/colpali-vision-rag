import { describe, it, expect } from 'vitest'
import { boxToOverlay, citedPage, heatmapRGBA, regionsOnPage } from './lib'
import type { PageHit, Region } from './types'

describe('boxToOverlay', () => {
  it('maps a 0-1000 box to CSS percentages', () => {
    expect(boxToOverlay([140, 300, 660, 700])).toEqual({
      top: '14%',
      left: '30%',
      width: '40%',
      height: '52%',
    })
  })

  it('normalizes a swapped min/max box', () => {
    expect(boxToOverlay([660, 700, 140, 300])).toEqual({
      top: '14%',
      left: '30%',
      width: '40%',
      height: '52%',
    })
  })

  it('returns null for a missing or malformed box', () => {
    expect(boxToOverlay([])).toBeNull()
    expect(boxToOverlay(null)).toBeNull()
    expect(boxToOverlay([1, 2, 3])).toBeNull()
  })
})

describe('citedPage', () => {
  const pages = [
    { index: 1, pdf: 'a.pdf', page_number: 3, score: 14.2, image: { url: 'u1', data_uri: null } },
    { index: 2, pdf: 'a.pdf', page_number: 5, score: 8.1, image: { url: 'u2', data_uri: null } },
  ] as PageHit[]

  it('resolves the 1-based source page', () => {
    expect(citedPage(pages, 1)?.page_number).toBe(3)
    expect(citedPage(pages, 2)?.page_number).toBe(5)
  })

  it('returns null when out of range (e.g. not-found -> 0)', () => {
    expect(citedPage(pages, 0)).toBeNull()
    expect(citedPage(pages, 3)).toBeNull()
  })
})

describe('regionsOnPage', () => {
  const regions = [
    { source_page: 1, box: [10, 10, 20, 20], pdf: 'a.pdf', page_number: 3, crop: null },
    { source_page: 2, box: [30, 30, 40, 40], pdf: 'a.pdf', page_number: 5, crop: null },
    { source_page: 1, box: [50, 50, 60, 60], pdf: 'a.pdf', page_number: 3, crop: null },
  ] as Region[]

  it('keeps only the regions that fall on the given page', () => {
    expect(regionsOnPage(regions, 1).map((r) => r.box)).toEqual([
      [10, 10, 20, 20],
      [50, 50, 60, 60],
    ])
    expect(regionsOnPage(regions, 2)).toHaveLength(1)
    expect(regionsOnPage(regions, 9)).toEqual([])
  })
})

describe('heatmapRGBA', () => {
  it('is fully transparent for a cold patch', () => {
    expect(heatmapRGBA(0)[3]).toBe(0)
  })

  it('is red-dominant and near-opaque at the hot end', () => {
    const [r, g, b, a] = heatmapRGBA(1)
    expect(a).toBeCloseTo(0.72)
    expect(r).toBeGreaterThan(g)
    expect(r).toBeGreaterThan(b)
  })

  it('has monotonically increasing alpha with value', () => {
    expect(heatmapRGBA(0.25)[3]).toBeLessThan(heatmapRGBA(0.75)[3])
  })

  it('clamps out-of-range input', () => {
    expect(heatmapRGBA(-1)[3]).toBe(0)
    expect(heatmapRGBA(2)[3]).toBeCloseTo(0.72)
  })
})
