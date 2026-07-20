import { imageSrc } from '../api'
import type { PageHit } from '../types'

/** The reranked-candidate thumbnail rail under the viewer. The cited page is highlighted;
 *  the pages Qdrant retrieved but rerank dropped collapse to a "N candidates trimmed" note. */
export function CandidateRail({
  pages,
  citedIndex,
  retrieveK,
}: {
  pages: PageHit[]
  citedIndex: number
  retrieveK: number
}) {
  const trimmed = Math.max(0, retrieveK - pages.length)
  return (
    <>
      <div className="section-label">reranked pages</div>
      <div className="candidates">
        {pages.map((p) => (
          <div key={p.index} className={`thumb${p.index === citedIndex ? ' kept' : ''}`}>
            {imageSrc(p.image) && <img src={imageSrc(p.image)} alt={`page ${p.page_number}`} />}
            <span className="thumb-label">
              p{p.page_number} · {p.score.toFixed(1)}
            </span>
          </div>
        ))}
        {trimmed > 0 && (
          <div
            style={{
              flex: 'none',
              alignSelf: 'center',
              color: 'var(--red)',
              font: '500 10px var(--mono)',
              lineHeight: 1.4,
              whiteSpace: 'nowrap',
            }}
          >
            {trimmed} candidates
            <br />
            trimmed →
          </div>
        )}
      </div>
    </>
  )
}
