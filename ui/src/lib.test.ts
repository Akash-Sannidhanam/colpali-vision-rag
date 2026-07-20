import { describe, it, expect } from 'vitest'
import { boxToOverlay, citedPage } from './lib'
import type { PageHit } from './types'

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
