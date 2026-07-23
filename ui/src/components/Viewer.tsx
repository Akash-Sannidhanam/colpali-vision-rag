import { useEffect, useRef, useState } from 'react'
import { heatmap as fetchHeatmap, imageSrc } from '../api'
import { boxToOverlay, citedPage, heatmapRGBA, regionsOnPage } from '../lib'
import type { HeatmapResponse, QueryResponse } from '../types'
import { CandidateRail } from './CandidateRail'

/** (3) The document viewer: the cited page with a CSS bounding-box overlay drawn from
 *  citation.box, the pulled-out crop, and the reranked-candidate rail. A "why this page?"
 *  toggle overlays the ColQwen2 MaxSim patch heatmap (fetched on demand from /heatmap). */
export function Viewer({ res, loading }: { res: QueryResponse | null; loading: boolean }) {
  // Hooks must run before the early returns below (rules of hooks).
  const [heatOn, setHeatOn] = useState(false)
  const [heat, setHeat] = useState<HeatmapResponse | null>(null)
  const [heatLoading, setHeatLoading] = useState(false)
  const [heatError, setHeatError] = useState(false)
  const cacheRef = useRef<Map<string, HeatmapResponse>>(new Map())
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  // The cited page (null for not-found / out-of-range) - also gates the heatmap toggle.
  const cited = res ? citedPage(res.pages, res.citation.source_page) : null

  // A new answer invalidates the fetched heatmap; heatOn stays as the user's sticky choice.
  useEffect(() => {
    setHeat(null)
    setHeatError(false)
    setHeatLoading(false)
  }, [res])

  // Fetch (or restore from cache) when the toggle is on and there's a cited page.
  useEffect(() => {
    if (!heatOn || !res || !cited) return
    const key = `${res.question}|${cited.pdf}|${cited.page_number}`
    const cached = cacheRef.current.get(key)
    if (cached) {
      setHeat(cached)
      return
    }
    let cancelled = false
    setHeatLoading(true)
    setHeatError(false)
    fetchHeatmap(res.question, cited.pdf, cited.page_number)
      .then((h) => {
        if (cancelled) return
        cacheRef.current.set(key, h)
        setHeat(h)
      })
      .catch(() => {
        if (!cancelled) setHeatError(true)
      })
      .finally(() => {
        if (!cancelled) setHeatLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [heatOn, res, cited])

  // Paint the tiny n_x x n_y grid onto the canvas; CSS scales it up into a smooth gradient.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !heatOn || !heat) return
    canvas.width = heat.n_x
    canvas.height = heat.n_y
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(heat.n_x, heat.n_y)
    for (let y = 0; y < heat.n_y; y++) {
      for (let x = 0; x < heat.n_x; x++) {
        const [r, g, b, a] = heatmapRGBA(heat.grid[y][x])
        const o = (y * heat.n_x + x) * 4
        img.data[o] = r
        img.data[o + 1] = g
        img.data[o + 2] = b
        img.data[o + 3] = Math.round(a * 255)
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [heat, heatOn])

  if (loading) {
    return (
      <div className="viewer">
        <div className="viewer-head">
          <span>reading the pages…</span>
        </div>
        <div className="stage">
          <div className="skeleton" style={{ width: 300, height: 400 }} />
        </div>
      </div>
    )
  }

  if (!res) {
    return (
      <div className="viewer">
        <div className="viewer-head">
          <span>viewer</span>
        </div>
        <div className="empty">
          <div className="glyph">▧</div>
          <div className="sub">The cited page appears here once you ask a question.</div>
        </div>
      </div>
    )
  }

  const citedSrc = cited ? imageSrc(cited.image) : undefined
  const regions = res.citation.found ? res.citation.regions : []
  // Overlays for the regions that land on the page currently shown (the primary page).
  const overlays = regionsOnPage(regions, res.citation.source_page)
    .map((r) => boxToOverlay(r.box))
    .filter((o): o is NonNullable<typeof o> => o !== null)
  // A single box normally darkens the rest of the page with a spotlight scrim; that would
  // fight the heatmap, so with the heatmap on we keep just the outline+glow (the .multi look).
  const spotlight = overlays.length === 1 && !(heatOn && heat)

  return (
    <div className="viewer">
      <div className="viewer-head">
        {cited ? (
          <>
            <span className="file">{cited.pdf}</span>
            <span>· p.{cited.page_number}</span>
          </>
        ) : (
          <span>no cited page</span>
        )}
        {cited && (
          <div className="heat-controls">
            {heatError && <span className="heat-error">heatmap failed</span>}
            {heatOn && heat && !heatError && (
              <span className="heat-legend">
                <span>match</span>
                <span className="bar" />
              </span>
            )}
            <button
              type="button"
              className={`heat-toggle${heatOn ? ' on' : ''}`}
              onClick={() => setHeatOn((v) => !v)}
              disabled={heatLoading}
              title="Overlay which page patches the query matched (ColQwen2 MaxSim)"
            >
              {heatLoading ? 'reading patches…' : heatOn ? 'hide heatmap' : 'why this page?'}
            </button>
          </div>
        )}
      </div>

      <div className="stage">
        {cited && citedSrc ? (
          <div className="page-frame">
            <img src={citedSrc} alt={`${cited.pdf} page ${cited.page_number}`} />
            {heatOn && heat && <canvas ref={canvasRef} className="heat-canvas" aria-hidden />}
            {overlays.map((overlay, i) => (
              <div
                key={i}
                className={`box-overlay${spotlight ? '' : ' multi'}`}
                style={overlay}
              />
            ))}
            {overlays[0] && (
              <span className="coord-tag" style={{ top: overlays[0].top, left: overlays[0].left }}>
                cited · {regions.length} region{regions.length === 1 ? '' : 's'}
              </span>
            )}
          </div>
        ) : (
          <div className="empty">
            <div className="glyph">∅</div>
            <div className="sub">
              The answer wasn’t found on the indexed pages, so there’s no region to show.
            </div>
          </div>
        )}
      </div>

      {regions.length > 0 && (
        <div className="crop-block">
          <div className="section-label" style={{ padding: '0 0 8px' }}>
            {regions.length === 1
              ? 'crop · where the answer was read'
              : `crops · ${regions.length} regions read`}
          </div>
          <div className="crops-strip">
            {regions.map((r, i) =>
              imageSrc(r.crop) ? (
                <div className="crop-frame" key={i}>
                  <img src={imageSrc(r.crop)} alt={`cited crop ${i + 1}`} />
                  {regions.length > 1 && r.page_number != null && (
                    <span className="crop-page">p.{r.page_number}</span>
                  )}
                </div>
              ) : null,
            )}
          </div>
        </div>
      )}

      <CandidateRail
        pages={res.pages}
        citedIndex={res.citation.source_page}
        retrieveK={res.meta.retrieve_k}
      />
    </div>
  )
}
