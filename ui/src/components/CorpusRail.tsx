import { useState } from 'react'
import type { CorpusResponse, HealthResponse } from '../types'

/** (1) The corpus rail: brand, ingest button, indexed-document list, Qdrant status.
 *
 *  Each document can be removed. The confirm is inline rather than a modal — deletion
 *  is a fast, local action, and a dialog would weigh more than the decision does. */
export function CorpusRail({
  corpus,
  health,
  onIngest,
  onDelete,
  onOpen,
}: {
  corpus: CorpusResponse | null
  health: HealthResponse | null
  onIngest: () => void
  onDelete: (pdf: string) => Promise<void>
  onOpen: (pdf: string) => void
}) {
  const online = health?.qdrant === 'ok'
  const total = corpus?.total_pages ?? 0
  // The half-persisted-corpus warning. The index and the page PNGs are one logical
  // corpus and the split is otherwise silent - /corpus still lists every document while
  // every query answers "not found" - so this is the only place a user learns about it.
  // "unknown" is the server's own deliberate placeholder (a 503 body, or a failed check),
  // not a problem to report.
  const corpusWarning =
    health?.corpus && health.corpus !== 'ok' && health.corpus !== 'unknown'
      ? health.corpus
      : null
  // The document awaiting confirmation, and the one currently being removed.
  const [confirming, setConfirming] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)

  const remove = async (pdf: string) => {
    setConfirming(null)
    setRemoving(pdf)
    try {
      await onDelete(pdf)
    } finally {
      setRemoving(null)
    }
  }

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
            <div className="doc-main">
              {/* The row opens the document; the ✕ and the confirm are *siblings* of this
                  button, never children - nesting interactive content is invalid HTML and
                  the inner control stops receiving clicks. */}
              <button
                className="doc-open"
                title={`Open ${d.pdf}`}
                disabled={confirming === d.pdf || removing === d.pdf}
                onClick={() => onOpen(d.pdf)}
              >
                <span className="doc-name">{d.pdf}</span>
                {confirming !== d.pdf && (
                  <span className="doc-sub">
                    {removing === d.pdf ? 'removing…' : `${d.page_count} pp · indexed`}
                  </span>
                )}
              </button>
              {confirming === d.pdf && (
                <div className="doc-confirm">
                  remove?
                  <button className="doc-confirm-btn danger" onClick={() => remove(d.pdf)}>
                    yes
                  </button>
                  <button className="doc-confirm-btn" onClick={() => setConfirming(null)}>
                    no
                  </button>
                </div>
              )}
            </div>
            {confirming !== d.pdf && removing !== d.pdf && (
              <button
                className="doc-remove"
                title={`Remove ${d.pdf}`}
                aria-label={`Remove ${d.pdf}`}
                onClick={() => setConfirming(d.pdf)}
              >
                ✕
              </button>
            )}
          </div>
        ))}
        {corpus && corpus.documents.length === 0 && (
          <div className="doc-sub" style={{ padding: 8 }}>
            No documents yet.
          </div>
        )}
      </div>

      {corpusWarning && (
        <div className="rail-warn" role="status">
          <span className="rail-warn-title">⚠ corpus incomplete</span>
          <span className="rail-warn-body">{corpusWarning}</span>
        </div>
      )}

      <div className="rail-foot">
        <span className={`dot ${online ? 'online' : 'offline'}`} />
        Qdrant · {online ? 'online' : health?.qdrant ? 'offline' : '…'}
      </div>
    </div>
  )
}
