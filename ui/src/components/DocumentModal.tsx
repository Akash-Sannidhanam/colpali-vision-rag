import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { UnauthorizedError, downloadDocument, getDocumentPages, imageSrc } from '../api'
import { boxToOverlay, pageFrameStyle, pageIndex, regionsOnDocumentPage } from '../lib'
import type { DocumentPagesResponse, Region } from '../types'

// Everything Tab can land on inside the dialog; used only to wrap at the two edges.
const FOCUSABLE = 'button:not(:disabled), [href], input, [tabindex]:not([tabindex="-1"])'

// Module-level so the default prop is referentially stable across renders.
const NO_REGIONS: Region[] = []

/**
 * The full-screen document viewer. Two ways in, one component: a corpus-rail row opens
 * page 1, and a candidate thumbnail or the cited page opens that page with the citation's
 * boxes still drawn over it.
 *
 * This is the app's first real dialog, so it carries the pattern: role="dialog" +
 * aria-modal, Esc to close, arrows/Home/End to page, a Tab trap, and focus returned to
 * whatever opened it.
 */
export function DocumentModal({
  pdf,
  initialPage,
  regions = NO_REGIONS,
  onClose,
  onAuthError,
}: {
  pdf: string
  initialPage: number
  regions?: Region[]
  onClose: () => void
  onAuthError: () => void
}) {
  const [doc, setDoc] = useState<DocumentPagesResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const [idx, setIdx] = useState(0)
  const [downloading, setDownloading] = useState(false)
  const [dlError, setDlError] = useState<string | null>(null)
  // The page's natural size, which becomes the frame's aspect-ratio. Without a definite
  // ratio the frame is content-sized, `max-height: 100%` on the image resolves to nothing,
  // and the page renders at full height inside a clipped box - cropping most of it *and*
  // misplacing the citation overlay, which is drawn in percentages of that box.
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)

  const shellRef = useRef<HTMLDivElement>(null)
  const curThumbRef = useRef<HTMLButtonElement>(null)
  const titleId = useId()

  const pages = doc?.pages ?? []
  const count = pages.length
  const page = pages[idx] ?? null
  const src = imageSrc(page?.image)

  // --- load the page list ---
  useEffect(() => {
    let cancelled = false
    setDoc(null)
    setError(null)
    getDocumentPages(pdf)
      .then((d) => {
        if (cancelled) return
        setDoc(d)
        setIdx(pageIndex(d.pages, initialPage))
      })
      .catch((e) => {
        if (cancelled) return
        // A 401 has to reach App or the key prompt never opens. App renders that prompt
        // after this modal and both sit at z-index 40, so DOM order stacks it above.
        if (e instanceof UnauthorizedError) onAuthError()
        setError(e instanceof Error ? e.message : 'Could not load this document.')
      })
    return () => {
      cancelled = true
    }
  }, [pdf, initialPage, attempt, onAuthError])

  // --- focus: into the dialog on open, back to the opener on close ---
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    shellRef.current?.focus()
    return () => {
      // Normally the rail row or thumbnail that opened this is still mounted. It is not if
      // the document was deleted underneath us, hence the isConnected guard.
      if (opener?.isConnected) opener.focus()
    }
  }, [])

  // --- keys: Esc closes, arrows page, Tab stays inside ---
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return
      // Never steal keys from a text field: the API-key prompt can be stacked on top.
      // `closest?.` and not just `?.closest` - a keydown's target is not always an Element
      // (it is `document` when the event is dispatched there), and calling a method that
      // does not exist would throw inside the listener and take Escape down with it.
      const target = e.target as HTMLElement | null
      if (target?.closest?.('input, textarea, [contenteditable="true"]')) return
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
        return
      }
      if (e.key === 'Tab') {
        const nodes = shellRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE)
        if (!nodes?.length) return
        const first = nodes[0]
        const last = nodes[nodes.length - 1]
        const active = document.activeElement
        if (e.shiftKey && (active === first || active === shellRef.current)) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && active === last) {
          e.preventDefault()
          first.focus()
        }
        return
      }
      if (!count) return
      // Functional updates keep `idx` out of this effect's deps, so the listener attaches
      // once per document rather than being torn down and rebound on every page turn.
      const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
      if (step) {
        e.preventDefault() // or the filmstrip scrolls under the same keypress
        setIdx((i) => Math.max(0, Math.min(count - 1, i + step)))
      } else if (e.key === 'Home') {
        e.preventDefault()
        setIdx(0)
      } else if (e.key === 'End') {
        e.preventDefault()
        setIdx(count - 1)
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [count, onClose])

  // Warm the neighbours so arrow-key browsing paints instantly. Pages are ~537 KB and
  // /images sets no Cache-Control, so even a revisit costs a conditional round-trip.
  // Only ±1: any more would compete with the page actually on screen.
  useEffect(() => {
    for (const d of [1, -1]) {
      const s = imageSrc(pages[idx + d]?.image)
      if (s) new Image().src = s
    }
  }, [pages, idx])

  // Keep the active filmstrip thumb in view. `block: 'nearest'` because the strip is a
  // horizontal scroller inside a fixed-height shell - vertical movement is never wanted.
  useEffect(() => {
    curThumbRef.current?.scrollIntoView({ block: 'nearest', inline: 'center' })
  }, [idx, count])

  // Deliberately never reset between pages: the pages of one PDF share a size, so keeping
  // the last ratio avoids a layout jump on every turn, and onLoad corrects it if one
  // differs. A ref callback as well as onLoad because onLoad does not fire for an image
  // that was already complete at mount - which is the common case here, since the
  // neighbours are preloaded.
  const measure = useCallback((el: HTMLImageElement | null) => {
    if (el?.complete && el.naturalWidth) setNatural({ w: el.naturalWidth, h: el.naturalHeight })
  }, [])

  const download = async () => {
    setDownloading(true)
    setDlError(null)
    try {
      await downloadDocument(pdf)
    } catch (e) {
      if (e instanceof UnauthorizedError) onAuthError()
      setDlError(e instanceof Error ? e.message : 'Download failed.')
    } finally {
      setDownloading(false)
    }
  }

  const overlays = page
    ? regionsOnDocumentPage(regions, pdf, page.page_number)
        .map((r) => boxToOverlay(r.box))
        .filter((o): o is NonNullable<typeof o> => o !== null)
    : []
  // Same rule as the viewer: a single box gets the spotlight scrim, several can't compose it.
  const spotlight = overlays.length === 1

  return (
    // mousedown, not click: a click fires when a text-drag starts inside and ends outside.
    <div className="doc-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className="doc-shell"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        ref={shellRef}
      >
        <div className="doc-head">
          <h2 id={titleId}>◧ {pdf}</h2>
          {page && (
            <span className="doc-count">
              p.{page.page_number} / {doc?.page_count ?? count}
            </span>
          )}
          <span className="doc-head-gap" />
          {dlError && <span className="error-line">{dlError}</span>}
          <button
            className="doc-btn"
            onClick={download}
            disabled={!doc?.has_original || downloading}
            title={
              doc && !doc.has_original
                ? 'The original PDF is no longer on disk — only its page images are.'
                : `Download ${pdf}`
            }
          >
            {downloading ? 'preparing…' : '↓ PDF'}
          </button>
          <button className="doc-btn" onClick={onClose} aria-label="Close (Esc)" title="Close (Esc)">
            ✕
          </button>
        </div>

        <div className="doc-stage">
          {error ? (
            <div className="empty">
              <div className="glyph">∅</div>
              <div className="sub">{error}</div>
              <button className="btn" onClick={() => setAttempt((n) => n + 1)}>
                try again
              </button>
            </div>
          ) : !doc ? (
            <div className="skeleton doc-page-skeleton" />
          ) : src && page ? (
            /* The frame's aspect ratio is load-bearing, not decorative: without a definite
               one the page is silently cropped and the citation box is drawn against the
               wrong rectangle. pageFrameStyle owns that rule so it can be tested - see its
               docstring in lib.ts. */
            <div className="doc-page" style={pageFrameStyle(natural)}>
              {/* No key on the <img>: a stable element identity lets the browser hold the
                  previous page painted until the new one decodes, which is a smoother swap
                  than any skeleton — a skeleton would blank the frame on every turn. */}
              <img
                ref={measure}
                src={src}
                alt={`${pdf} page ${page.page_number}`}
                onLoad={(e) =>
                  setNatural({
                    w: e.currentTarget.naturalWidth,
                    h: e.currentTarget.naturalHeight,
                  })
                }
              />
              {overlays.map((o, i) => (
                <div key={i} className={`box-overlay${spotlight ? '' : ' multi'}`} style={o} />
              ))}
            </div>
          ) : (
            <div className="empty">
              <div className="glyph">▧</div>
              <div className="sub">
                Page {page?.page_number ?? '—'} is indexed, but its page image is missing from
                disk. Re-ingest {pdf} to restore it.
              </div>
            </div>
          )}

          <button
            className="doc-nav prev"
            onClick={() => setIdx((i) => Math.max(0, i - 1))}
            disabled={idx === 0 || !count}
            aria-label="Previous page"
          >
            ‹
          </button>
          <button
            className="doc-nav next"
            onClick={() => setIdx((i) => Math.min(count - 1, i + 1))}
            disabled={idx >= count - 1}
            aria-label="Next page"
          >
            ›
          </button>
        </div>

        <div className="doc-strip">
          {pages.map((p, i) => {
            const thumb = imageSrc(p.image)
            const cited = regionsOnDocumentPage(regions, pdf, p.page_number).length > 0
            return (
              <button
                key={p.page_number}
                ref={i === idx ? curThumbRef : null}
                className={`thumb${i === idx ? ' on' : ''}${cited ? ' cited' : ''}`}
                onClick={() => setIdx(i)}
                aria-label={`Page ${p.page_number}${cited ? ' · cited' : ''}`}
                aria-current={i === idx ? 'true' : undefined}
              >
                {/* Full-resolution page PNGs stand in for thumbnails (~537 KB, 1241x1754)
                    and a long document has 59 of them, so lazy is what keeps opening one
                    from costing 21 MB and half a gigabyte of decoded bitmap. */}
                {thumb ? (
                  <img src={thumb} alt="" loading="lazy" decoding="async" />
                ) : (
                  <span className="thumb-missing" aria-hidden>
                    ▧
                  </span>
                )}
                <span className="thumb-label">p{p.page_number}</span>
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
