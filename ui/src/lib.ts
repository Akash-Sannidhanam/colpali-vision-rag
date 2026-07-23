import type { PageHit, Region } from './types'

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
