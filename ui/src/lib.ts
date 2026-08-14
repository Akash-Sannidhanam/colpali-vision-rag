import type { DocumentPage, PageHit, Region } from './types'

/**
 * Convert a Gemini box `[ymin, xmin, ymax, xmax]` on a 0-1000 scale into CSS
 * percentages for an absolutely-positioned overlay drawn over the page image.
 * Normalizes swapped min/max; returns null for a missing/malformed box.
 */
export function boxToOverlay(
  box: number[] | null | undefined,
): { top: string; left: string; width: string; height: string } | null {
  if (!box || box.length !== 4) return null
  const [ymin, xmin, ymax, xmax] = box
  const top = Math.min(ymin, ymax) / 10
  const left = Math.min(xmin, xmax) / 10
  const height = Math.abs(ymax - ymin) / 10
  const width = Math.abs(xmax - xmin) / 10
  return { top: `${top}%`, left: `${left}%`, width: `${width}%`, height: `${height}%` }
}

/** The cited page for a 1-based source_page index (null when out of range / not found). */
export function citedPage(pages: PageHit[], sourcePage: number): PageHit | null {
  return sourcePage >= 1 && sourcePage <= pages.length ? pages[sourcePage - 1] : null
}

/** The cited regions that fall on a given 1-based page - the ones to overlay on it. */
export function regionsOnPage(regions: Region[], sourcePage: number): Region[] {
  return regions.filter((r) => r.source_page === sourcePage)
}

/**
 * The cited regions that fall on a given page of a given document.
 *
 * Deliberately distinct from `regionsOnPage`, and the distinction is the whole point.
 * `source_page` is a 1-based index into one answer's retrieved slate; `page_number` is the
 * page inside the PDF. The document viewer walks pages 1..n and knows nothing about a
 * slate, so it has to key on the latter - and on the document too, because two different
 * PDFs routinely both have a page 3. Regions the backend could not attribute to a page
 * (`pdf`/`page_number` null) belong to no page and are dropped.
 */
export function regionsOnDocumentPage(
  regions: Region[],
  pdf: string,
  pageNumber: number,
): Region[] {
  return regions.filter((r) => r.pdf === pdf && r.page_number === pageNumber)
}

/**
 * The index into `pages` showing `pageNumber`, or 0 (the first page) when it isn't there.
 *
 * Viewer navigation runs on indices, never on page-number arithmetic. `page_number ===
 * index + 1` holds today, but `image: null` exists precisely because the index and the
 * rendered images can diverge, so the returned list is the only authority on what "next"
 * means - and clamping against `pages.length` makes a short list harmless instead of a
 * crash.
 */
export function pageIndex(
  pages: DocumentPage[],
  pageNumber: number | null | undefined,
): number {
  if (pageNumber == null) return 0
  const i = pages.findIndex((p) => p.page_number === pageNumber)
  return i === -1 ? 0 : i
}

/**
 * The inline style for the full-screen page frame, given the page image's natural size.
 *
 * The frame must never be laid out without a definite aspect ratio, and that is the whole
 * point of this function rather than a nicety. `.doc-page img` is sized in percentages of
 * this frame, and a frame whose own height is content-based is *indefinite* - so
 * `max-height: 100%` on the image resolves to nothing, the image renders at its full
 * natural height, and `overflow: hidden` crops the difference away in silence. Measured
 * when this shipped broken: a 1650px page inside a 488px frame, 1085px gone.
 *
 * The cropping is the visible half. The dangerous half is that the citation overlay is
 * positioned in percentages of this same frame, so a cropped frame draws the box against
 * a different rectangle than the one the model measured - a confidently wrong visual
 * citation, which is the one failure this product cannot have.
 *
 * So there are exactly two legal outcomes and no third: a definite ratio, or a frame that
 * is not laid out at all. A degenerate size (the image has not decoded yet, so
 * naturalWidth is 0) must take the second branch - `0 / 0` and `1275 / 0` are invalid
 * `aspect-ratio` values that the browser drops, which is the original bug wearing a hat.
 */
export function pageFrameStyle(
  natural: { w: number; h: number } | null | undefined,
): { aspectRatio?: string; visibility?: 'hidden' } {
  if (!natural || natural.w <= 0 || natural.h <= 0) return { visibility: 'hidden' }
  return { aspectRatio: `${natural.w} / ${natural.h}` }
}

// --- PDF zoom ---
//
// The scale a PDF page is rendered at, as a multiple of its own point size. These are
// pure so they can be tested here; what they cannot check is that the rendered frame
// really is the rectangle the citation overlay is drawn in - that is layout, and it lives
// in e2e/document-viewer.spec.ts.

/** Below this the text layer is unreadable; above it a page can exhaust canvas memory. */
export const MIN_ZOOM = 0.25
export const MAX_ZOOM = 6

/** `z` brought inside the supported range. */
export function clampZoom(z: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z))
}

/**
 * The scale at which a page of `page` points fits entirely inside `stage` pixels.
 *
 * Both axes, taking the smaller - a page must fit in the direction that binds, which for
 * a wide landscape scan is the width and for an ordinary portrait page is the height.
 * Returns null when either box is degenerate (a stage measured before layout, a page
 * whose viewport has not resolved), because the alternative is a 0 or Infinity scale that
 * renders a canvas of no size or of ruinous size. The caller shows nothing until there is
 * a real number, the same contract `pageFrameStyle` has above.
 */
export function fitScale(
  page: { w: number; h: number } | null | undefined,
  stage: { w: number; h: number } | null | undefined,
): number | null {
  if (!page || !stage) return null
  if (page.w <= 0 || page.h <= 0 || stage.w <= 0 || stage.h <= 0) return null
  return clampZoom(Math.min(stage.w / page.w, stage.h / page.h))
}

/**
 * One zoom step in `dir` from `z`.
 *
 * Multiplicative, not additive: a fixed +0.25 is a 100% jump from 0.25 and a 4% nudge at
 * 6, so the control would feel broken at both ends of its own range.
 */
export function zoomStep(z: number, dir: 1 | -1): number {
  return clampZoom(z * (dir === 1 ? 1.25 : 1 / 1.25))
}

/**
 * Retrieval decisiveness as a multiple of an evenly-spread slate, or null if N/A.
 *
 * `retrieval_confidence` is the cited page's share of the softmax mass over `k`
 * candidates, so its floor is `1/k` (every page equally liked) and not zero. Rendering
 * it as a bare percentage was actively misleading: with k=12 the whole observed range
 * across the 83-question eval is 6-21%, so a *maximally* decisive retrieval showed
 * "21%" and a typical correct answer showed "11%" - which reads as the system doubting
 * an answer it got right.
 *
 * Dividing by the floor removes that artifact: 1x is "no preference at all", and the
 * number stops moving when RETRIEVE_K changes. Measured on
 * eval/reports/calib_baseline.json, retrieval's top page averages 1.50x when it is the
 * gold page against 1.35x when it is not - a real difference (AUC 0.629, permutation
 * p=0.016 at n=49/24), but a weak one, which is why this is a trace-level diagnostic
 * and not a headline chip.
 */
export function decisivenessVsUniform(
  confidence: number | null | undefined,
  k: number,
): number | null {
  if (confidence == null || !k || k <= 0) return null
  return confidence * k
}

/**
 * Map a normalized patch score in [0,1] to an RGBA tuple (alpha 0-1) for the "why this
 * page?" heatmap: cold patches stay clear, ramping blue -> cyan -> yellow -> red as the
 * query match strengthens, with alpha rising so only the patches that matter tint the
 * page. Input is clamped to [0,1].
 */
export function heatmapRGBA(value: number): [number, number, number, number] {
  const v = Math.max(0, Math.min(1, value))
  const stops: [number, [number, number, number]][] = [
    [0.0, [30, 60, 220]], // blue
    [0.33, [0, 200, 210]], // cyan
    [0.66, [240, 220, 40]], // yellow
    [1.0, [230, 40, 40]], // red
  ]
  let rgb = stops[stops.length - 1][1]
  for (let i = 0; i < stops.length - 1; i++) {
    const [lo, c0] = stops[i]
    const [hi, c1] = stops[i + 1]
    if (v <= hi) {
      const t = hi === lo ? 0 : (v - lo) / (hi - lo)
      rgb = [
        Math.round(c0[0] + (c1[0] - c0[0]) * t),
        Math.round(c0[1] + (c1[1] - c0[1]) * t),
        Math.round(c0[2] + (c1[2] - c0[2]) * t),
      ]
      break
    }
  }
  return [rgb[0], rgb[1], rgb[2], v * 0.72] // cold -> clear; hottest patch ~0.72 alpha
}
