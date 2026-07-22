import { imageSrc } from '../api'
import { boxToOverlay, citedPage, regionsOnPage } from '../lib'
import type { QueryResponse } from '../types'
import { CandidateRail } from './CandidateRail'

/** (3) The document viewer: the cited page with a CSS bounding-box overlay drawn from
 *  citation.box, the pulled-out crop, and the reranked-candidate rail. */
export function Viewer({ res, loading }: { res: QueryResponse | null; loading: boolean }) {
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

  const cited = citedPage(res.pages, res.citation.source_page)
  const citedSrc = cited ? imageSrc(cited.image) : undefined
  const regions = res.citation.found ? res.citation.regions : []
  // Overlays for the regions that land on the page currently shown (the primary page).
  const overlays = regionsOnPage(regions, res.citation.source_page)
    .map((r) => boxToOverlay(r.box))
    .filter((o): o is NonNullable<typeof o> => o !== null)

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
      </div>

      <div className="stage">
        {cited && citedSrc ? (
          <div className="page-frame">
            <img src={citedSrc} alt={`${cited.pdf} page ${cited.page_number}`} />
            {overlays.map((overlay, i) => (
              <div
                key={i}
                className={`box-overlay${overlays.length > 1 ? ' multi' : ''}`}
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
