import type { PageHit } from './types'

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
