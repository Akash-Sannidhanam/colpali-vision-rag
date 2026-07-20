import type { CorpusResponse, HealthResponse } from '../types'

/** (1) The corpus rail: brand, ingest button, indexed-document list, Qdrant status. */
export function CorpusRail({
  corpus,
  health,
  onIngest,
}: {
  corpus: CorpusResponse | null
  health: HealthResponse | null
  onIngest: () => void
}) {
  const online = health?.qdrant === 'ok'
  const total = corpus?.total_pages ?? 0

  return (
    <div className="rail">
      <div className="brand">
        <div className="brand-mark" />
        <div className="brand-name">Vision RAG</div>
      </div>

      <button className="ingest-btn" onClick={onIngest}>
        ＋ ingest PDF
      </button>

      <div className="section-label">corpus · {total} pages</div>
      <div className="doc-list">
        {corpus?.documents.map((d) => (
          <div className="doc" key={d.pdf}>
            <div className="doc-status" />
            <div style={{ minWidth: 0 }}>
              <div className="doc-name">{d.pdf}</div>
              <div className="doc-sub">{d.page_count} pp · indexed</div>
            </div>
          </div>
        ))}
        {corpus && corpus.documents.length === 0 && (
          <div className="doc-sub" style={{ padding: 8 }}>
            No documents yet.
          </div>
        )}
      </div>

      <div className="rail-foot">
        <span className={`dot ${online ? 'online' : 'offline'}`} />
        Qdrant · {online ? 'online' : health?.qdrant ? 'offline' : '…'}
      </div>
    </div>
  )
}
