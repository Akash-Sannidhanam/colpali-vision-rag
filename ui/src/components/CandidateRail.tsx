import { imageSrc } from '../api'
import type { PageHit } from '../types'

/** The reranked-candidate thumbnail rail under the viewer. The cited page is highlighted;
 *  the pages Qdrant retrieved but rerank dropped collapse to a "N candidates trimmed" note. */
export function CandidateRail({
  pages,
  citedIndex,
  retrieveK,
  onOpen,
}: {
  pages: PageHit[]
  citedIndex: number
  retrieveK: number
  onOpen: (page: PageHit) => void
}) {
  const trimmed = Math.max(0, retrieveK - pages.length)
  return (
    <>
      <div className="section-label">reranked pages</div>
      <div className="candidates">
        {pages.map((p) => (
          <button
            key={p.index}
            className={`thumb${p.index === citedIndex ? ' kept' : ''}`}
            onClick={() => onOpen(p)}
            aria-label={`Open page ${p.page_number} of ${p.pdf}`}
            title={`${p.pdf} · p.${p.page_number}`}
          >
            {/* alt="" because the button's aria-label already names it - a described
                image inside a labelled button is announced twice. */}
            {imageSrc(p.image) && <img src={imageSrc(p.image)} alt="" />}
            <span className="thumb-label">
              p{p.page_number} · {p.score.toFixed(1)}
            </span>
          </button>
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
