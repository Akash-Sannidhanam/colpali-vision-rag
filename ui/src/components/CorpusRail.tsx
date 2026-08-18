import { useEffect, useRef, useState } from 'react'
import type { CorpusResponse, HealthResponse } from '../types'

/** (1) The corpus rail: brand, ingest button, indexed-document list, Qdrant status.
 *
 *  Each document can be removed. The confirm is inline rather than a modal — deletion
 *  is a fast, local action, and a dialog would weigh more than the decision does.
 *
 *  Below 1100px this same element is an off-canvas drawer (see the breakpoints at the
 *  foot of theme.css). The component does not know that: it takes an `open` flag and an
 *  `id` for the hamburger's aria-controls, and CSS decides whether either means anything
 *  at the current width. */
export function CorpusRail({
  id,
  open,
  corpus,
  corpusError,
  onRetry,
  health,
  onIngest,
  onDelete,
  onOpen,
}: {
  id: string
  /** Drawer state. Inert at widths where the rail is a static column. */
  open: boolean
  corpus: CorpusResponse | null
  /** A failed /corpus, already worded for a human. null while loading or loaded. */
  corpusError: string | null
  onRetry: () => void
  health: HealthResponse | null
  onIngest: () => void
  onDelete: (pdf: string) => Promise<void>
  onOpen: (pdf: string) => void
}) {
  const online = health?.qdrant === 'ok'
  const total = corpus?.total_pages ?? 0
  // "corpus · 0 pages" while the request is still in flight says the same wrong thing the
  // blank list did, one line higher up - so the count waits for a real number too.
  const countLabel = corpus ? `${total} pages` : corpusError ? 'unavailable' : '…'
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

  // Where focus goes when the inline confirm opens and closes. Confirming replaces the
  // focused ✕ with two new buttons, so without this focus falls to <body> and the next
  // Tab restarts at the top of the page - the keyboard user is thrown out of the row they
  // were acting on, mid-decision.
  const yesRef = useRef<HTMLButtonElement>(null)
  const removeRefs = useRef(new Map<string, HTMLButtonElement>())

  useEffect(() => {
    if (confirming) yesRef.current?.focus()
  }, [confirming])

  /** Dismiss the confirm and hand focus back to the ✕ that opened it. */
  const cancel = (pdf: string) => {
    setConfirming(null)
    // After the re-render that brings the ✕ back, or there is nothing to focus yet.
    requestAnimationFrame(() => removeRefs.current.get(pdf)?.focus())
  }

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
    /* A landmark, so screen-reader users can jump straight to the corpus instead of
       tabbing the whole shell. Not <nav>: this is a list of documents to open and delete,
       not site navigation. */
    <aside className={`rail${open ? ' open' : ''}`} id={id} aria-label="Corpus">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true" />
        {/* The app's only h1. It lives here because this is where the product is named;
            the rail being an off-canvas drawer at narrow widths does not change that -
            heading order is a document property, not a layout one. */}
        <h1 className="brand-name">Vision RAG</h1>
      </div>

      <button className="ingest-btn" onClick={onIngest}>
        ＋ ingest PDF
      </button>

      <div className="section-label">corpus · {countLabel}</div>
      <div className="doc-list">
        {corpus?.documents.map((d) => (
          <div className="doc" key={d.pdf}>
            <div className="doc-status" aria-hidden="true" />
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
                /* Announced as a group so the prompt and its two answers are read
                   together; not a dialog - it is three inline controls, and giving it a
                   dialog role would promise a focus trap that is not here. */
                <div className="doc-confirm" role="group" aria-label={`Remove ${d.pdf}?`}>
                  remove?
                  <button
                    ref={yesRef}
                    className="doc-confirm-btn danger"
                    onClick={() => remove(d.pdf)}
                  >
                    yes
                  </button>
                  <button className="doc-confirm-btn" onClick={() => cancel(d.pdf)}>
                    no
                  </button>
                </div>
              )}
            </div>
            {confirming !== d.pdf && removing !== d.pdf && (
              <button
                ref={(el) => {
                  if (el) removeRefs.current.set(d.pdf, el)
                  else removeRefs.current.delete(d.pdf)
                }}
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
        {/* Four states, where there used to be two. `corpus === null` was rendering as
            nothing at all, which is indistinguishable from an empty corpus - and an
            outright failure rendered as nothing *forever*. */}
        {corpusError ? (
          <div className="rail-error" role="alert">
            <span className="rail-error-body">{corpusError}</span>
            <button className="rail-retry" onClick={onRetry}>
              retry
            </button>
          </div>
        ) : corpus === null ? (
          <div className="doc-skeletons" aria-label="Loading the corpus" aria-busy="true">
            {[0, 1, 2].map((i) => (
              <div className="skeleton doc-skeleton" key={i} />
            ))}
          </div>
        ) : corpus.documents.length === 0 ? (
          <div className="doc-sub" style={{ padding: 8 }}>
            No documents yet.
          </div>
        ) : null}
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
    </aside>
  )
}
