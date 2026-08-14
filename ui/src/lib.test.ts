import { describe, it, expect } from 'vitest'
import {
  MAX_ZOOM,
  MIN_ZOOM,
  boxToOverlay,
  citedPage,
  clampZoom,
  decisivenessVsUniform,
  fitScale,
  heatmapRGBA,
  pageFrameStyle,
  pageIndex,
  regionsOnDocumentPage,
  regionsOnPage,
  zoomStep,
} from './lib'
import type { DocumentPage, PageHit, Region } from './types'

describe('decisivenessVsUniform', () => {
  it('reports an evenly-spread slate as 1x uniform', () => {
    // Every candidate holding an equal share is the undecided case, whatever k is.
    expect(decisivenessVsUniform(1 / 12, 12)).toBeCloseTo(1)
  })

  it('reports the measured hit average as roughly 1.5x uniform', () => {
    // 0.1253 is decisiveness_hit_avg from eval/reports/calib_baseline.json. As a raw
    // percentage it reads "13%" - the presentation defect this function exists to fix.
    expect(decisivenessVsUniform(0.1253, 12)).toBeCloseTo(1.5, 1)
  })

  it('rises with decisiveness', () => {
    const spread = decisivenessVsUniform(0.11, 12) ?? 0
    expect(decisivenessVsUniform(0.21, 12)).toBeGreaterThan(spread)
  })

  it('is null when there is no confidence or no slate to compare against', () => {
    expect(decisivenessVsUniform(null, 12)).toBeNull()
    expect(decisivenessVsUniform(0.5, 0)).toBeNull()
  })
})

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

describe('regionsOnDocumentPage', () => {
  const regions = [
    { source_page: 1, box: [10, 10, 20, 20], pdf: 'a.pdf', page_number: 7, crop: null },
    { source_page: 2, box: [30, 30, 40, 40], pdf: 'b.pdf', page_number: 7, crop: null },
    { source_page: 3, box: [50, 50, 60, 60], pdf: 'a.pdf', page_number: 7, crop: null },
    { source_page: 0, box: [70, 70, 80, 80], pdf: null, page_number: null, crop: null },
  ] as Region[]

  it('keeps only the regions on that page of that document', () => {
    expect(regionsOnDocumentPage(regions, 'a.pdf', 7).map((r) => r.box)).toEqual([
      [10, 10, 20, 20],
      [50, 50, 60, 60],
    ])
  })

  it('does not confuse two documents that both have that page number', () => {
    expect(regionsOnDocumentPage(regions, 'b.pdf', 7)).toHaveLength(1)
  })

  it('keys on the document page, not the slate position', () => {
    // The regression this exists to prevent. source_page 1 is an index into one answer's
    // retrieved slate; page 1 of a.pdf carries no cited region at all.
    expect(regionsOnDocumentPage(regions, 'a.pdf', 1)).toEqual([])
  })

  it('drops regions the backend could not attribute to a page', () => {
    expect(regionsOnDocumentPage(regions, 'a.pdf', 7).every((r) => r.pdf !== null)).toBe(true)
  })
})

describe('pageIndex', () => {
  const pages = [
    { page_number: 1, image: null },
    { page_number: 2, image: { url: 'u2', data_uri: null } },
    { page_number: 3, image: null },
  ] as DocumentPage[]

  it('resolves a page number to its position in the list', () => {
    expect(pageIndex(pages, 3)).toBe(2)
  })

  it('falls back to the first page for one this document does not have', () => {
    expect(pageIndex(pages, 99)).toBe(0)
    expect(pageIndex(pages, null)).toBe(0)
  })

  it('does not assume page_number equals position + 1', () => {
    // Navigation runs on indices precisely so a gap cannot desynchronize it: page 5 is
    // at index 1 here, not index 4.
    const gapped = [{ page_number: 2 }, { page_number: 5 }] as DocumentPage[]
    expect(pageIndex(gapped, 5)).toBe(1)
  })

  it('falls back to the first page of an empty list', () => {
    expect(pageIndex([], 1)).toBe(0)
  })
})

describe('pageFrameStyle', () => {
  it('gives the frame the page image’s own aspect ratio', () => {
    expect(pageFrameStyle({ w: 1275, h: 1650 })).toEqual({ aspectRatio: '1275 / 1650' })
  })

  it('does the same for a landscape page', () => {
    expect(pageFrameStyle({ w: 1650, h: 1275 })).toEqual({ aspectRatio: '1650 / 1275' })
  })

  it('hides the frame rather than laying it out unsized', () => {
    expect(pageFrameStyle(null)).toEqual({ visibility: 'hidden' })
    expect(pageFrameStyle(undefined)).toEqual({ visibility: 'hidden' })
  })

  it('treats a not-yet-decoded image as unknown, not as a zero ratio', () => {
    // naturalWidth/Height are 0 until the image decodes. `0 / 0` and `1275 / 0` are
    // invalid aspect-ratio values that the browser drops - which is the crop bug wearing
    // a hat, so these must take the hidden branch and never emit a ratio.
    for (const natural of [{ w: 0, h: 0 }, { w: 1275, h: 0 }, { w: 0, h: 1650 }]) {
      expect(pageFrameStyle(natural)).toEqual({ visibility: 'hidden' })
    }
  })

  it('never lays the frame out without a definite ratio, for any input', () => {
    // The regression guard. The frame carries the citation overlay's percentage
    // coordinates, and its children are sized in percentages of it, so an *indefinite*
    // frame silently crops the page and misplaces the box drawn on it. Exactly one of the
    // two branches must always be taken - never neither, which is what shipped broken.
    const inputs = [
      null, undefined,
      { w: 1275, h: 1650 }, { w: 1650, h: 1275 }, { w: 1, h: 1 },
      { w: 0, h: 0 }, { w: 1275, h: 0 }, { w: 0, h: 1650 },
      { w: -1, h: 100 }, { w: 100, h: -1 },
    ]
    for (const natural of inputs) {
      const style = pageFrameStyle(natural)
      const sized = typeof style.aspectRatio === 'string' && /^\d+ \/ \d+$/.test(style.aspectRatio)
      const hidden = style.visibility === 'hidden'
      expect(sized !== hidden).toBe(true) // exactly one, for every input
    }
  })
})

describe('fitScale', () => {
  it('fits a portrait page bound by the stage’s height', () => {
    // 792pt tall into 720px of stage: height binds, and 612 * 0.909 = 556px wide fits.
    expect(fitScale({ w: 612, h: 792 }, { w: 1000, h: 720 })).toBeCloseTo(720 / 792)
  })

  it('fits a panorama page bound by the stage’s width', () => {
    // The branch a height-only fit would pass while still overflowing sideways.
    expect(fitScale({ w: 3000, h: 500 }, { w: 1000, h: 720 })).toBeCloseTo(1000 / 3000)
  })

  it('takes the smaller of the two axes, never the larger', () => {
    for (const page of [{ w: 612, h: 792 }, { w: 792, h: 612 }, { w: 3000, h: 500 }]) {
      const s = fitScale(page, { w: 1000, h: 720 })!
      expect(page.w * s).toBeLessThanOrEqual(1000 + 0.001)
      expect(page.h * s).toBeLessThanOrEqual(720 + 0.001)
    }
  })

  it('is null rather than 0 or Infinity when either box is degenerate', () => {
    // A stage measured before layout, or a viewport that has not resolved. Both would
    // otherwise produce a canvas of no size or of ruinous size.
    expect(fitScale(null, { w: 1000, h: 720 })).toBeNull()
    expect(fitScale({ w: 612, h: 792 }, null)).toBeNull()
    expect(fitScale({ w: 0, h: 792 }, { w: 1000, h: 720 })).toBeNull()
    expect(fitScale({ w: 612, h: 792 }, { w: 1000, h: 0 })).toBeNull()
  })

  it('stays inside the supported zoom range even for an enormous stage', () => {
    expect(fitScale({ w: 10, h: 10 }, { w: 4000, h: 4000 })).toBe(MAX_ZOOM)
  })
})

describe('clampZoom / zoomStep', () => {
  it('holds the range at both ends', () => {
    expect(clampZoom(0.01)).toBe(MIN_ZOOM)
    expect(clampZoom(99)).toBe(MAX_ZOOM)
    expect(clampZoom(1)).toBe(1)
  })

  it('steps multiplicatively, so the control feels the same at both ends', () => {
    // A fixed +0.25 would be a 100% jump from 0.25 and a 4% nudge at 6.
    expect(zoomStep(1, 1)).toBeCloseTo(1.25)
    expect(zoomStep(4, 1) / 4).toBeCloseTo(zoomStep(1, 1) / 1)
  })

  it('round-trips in and back out', () => {
    expect(zoomStep(zoomStep(2, 1), -1)).toBeCloseTo(2)
  })

  it('cannot step outside the range', () => {
    expect(zoomStep(MAX_ZOOM, 1)).toBe(MAX_ZOOM)
    expect(zoomStep(MIN_ZOOM, -1)).toBe(MIN_ZOOM)
  })
})
