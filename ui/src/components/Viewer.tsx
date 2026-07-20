import { imageSrc } from '../api'
import { boxToOverlay, citedPage } from '../lib'
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
  const overlay = res.citation.found ? boxToOverlay(res.citation.box) : null
  const citedSrc = cited ? imageSrc(cited.image) : undefined

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
            {overlay && (
              <>
                <div className="box-overlay" style={overlay} />
                <span className="coord-tag" style={{ top: overlay.top, left: overlay.left }}>
                  cited · box [{res.citation.box.join(',')}]
                </span>
              </>
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

      {res.crop && imageSrc(res.crop) && (
        <div className="crop-block">
          <div className="section-label" style={{ padding: '0 0 8px' }}>
            crop · where the answer was read
          </div>
          <div className="crop-frame">
            <img src={imageSrc(res.crop)} alt="cited crop" />
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
