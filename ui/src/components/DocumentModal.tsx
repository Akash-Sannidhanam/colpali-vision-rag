import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { UnauthorizedError, getDocumentFile, getDocumentPages, imageSrc, saveBlob } from '../api'
import {
  clampZoom,
  fitScale,
  overlaysFor,
  pageFrameStyle,
  pageIndex,
  regionsOnDocumentPage,
  zoomStep,
} from '../lib'
import { closePdf, loadPdf } from '../pdf'
import type { DocumentPagesResponse, Region } from '../types'
import { useNaturalSize } from '../useNaturalSize'
import { PdfPageView } from './PdfPageView'

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
 *
 * **It shows the actual PDF, not the page PNG.** The PNGs are what `pdf_render` made for
 * ColQwen2 - 150 DPI rasters that do not scale, cannot be selected and cannot be searched.
 * The reader gets the document itself, rendered by pdf.js at whatever scale they zoom to,
 * with a real text layer over it. The PNG remains the fallback, and that is a first-class
 * path rather than an error one: `has_original: false` is a state the corpus genuinely
 * reaches (the source PDF was deleted, or a restore missed `pdfs/`), and a document in it
 * must still be readable. The selection is per page - PDF, else PNG, else the empty state -
 * so one unrenderable page cannot take the document down with it.
 *
 * The filmstrip stays on PNGs deliberately. Rendering 59 pdf.js thumbnails would cost far
 * more than the lazy full-resolution PNGs it replaced.
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
  // Gives .doc-page its aspect ratio on the PNG path; see the hook for why that matters.
  const { natural, imgProps } = useNaturalSize()

  // --- the PDF ---
  // The Blob is held, not just the parsed document: `getDocument({data})` detaches the
  // buffer it is given, so the download button needs its own copy and Blob.arrayBuffer()
  // is what provides one.
  const [blob, setBlob] = useState<Blob | null>(null)
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null)
  // Whole-document give-up (no source file, fetch failed, not a readable PDF) and the
  // per-page one. Both mean "render the PNG instead", quietly.
  const [pdfFailed, setPdfFailed] = useState(false)
  const [failedPages, setFailedPages] = useState<ReadonlySet<number>>(() => new Set())
  const [pdfSize, setPdfSize] = useState<{ w: number; h: number } | null>(null)
  const [stage, setStage] = useState<{ w: number; h: number } | null>(null)
  // null means "fit to the stage" - a mode, not a number, so a window resize keeps fitting.
  const [zoom, setZoom] = useState<number | null>(null)

  const shellRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const curThumbRef = useRef<HTMLButtonElement>(null)
  const titleId = useId()

  const pages = doc?.pages ?? []
  const count = pages.length
  const page = pages[idx] ?? null
  const src = imageSrc(page?.image)

  const fit = fitScale(pdfSize, stage)
  const scale = zoom ?? fit
  const usePdf =
    !!pdfDoc && !pdfFailed && !!page && !failedPages.has(page.page_number) && !!pdfSize && !!scale

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

  // --- load the PDF itself ---
  //
  // Waits for the page list, because `has_original` is what says whether there is a source
  // file at all - fetching first would 404 on every document in that state. A failure here
  // is never surfaced as an error: the PNGs are still there and still correct, so the
  // viewer quietly becomes the one it used to be.
  useEffect(() => {
    if (!doc) return
    if (!doc.has_original) {
      setPdfFailed(true)
      return
    }
    let cancelled = false
    let opened: PDFDocumentProxy | null = null
    setPdfFailed(false)
    getDocumentFile(pdf)
      .then(async (b) => {
        if (cancelled) return
        setBlob(b)
        const d = await loadPdf(b)
        if (cancelled) return closePdf(d)
        opened = d
        setPdfDoc(d)
      })
      .catch((e) => {
        if (cancelled) return
        if (e instanceof UnauthorizedError) onAuthError()
        setPdfFailed(true)
      })
    return () => {
      cancelled = true
      // It owns a worker; leaking one per document opened is a real leak.
      if (opened) closePdf(opened)
      setPdfDoc(null)
    }
  }, [doc, pdf, onAuthError])

  // The current page's size in PDF points, which is what `fitScale` needs. Never reset
  // between pages, for the same reason `natural` is not: the pages of one document
  // usually share a size, so keeping the last one avoids a layout jump on every turn.
  useEffect(() => {
    if (!pdfDoc || !page) return
    let cancelled = false
    pdfDoc
      .getPage(page.page_number)
      .then((p) => {
        if (cancelled) return
        const v = p.getViewport({ scale: 1 })
        setPdfSize({ w: v.width, h: v.height })
      })
      .catch(() => {
        if (!cancelled) setPdfFailed(true)
      })
    return () => {
      cancelled = true
    }
  }, [pdfDoc, page])

  // --- the stage's usable box, which is what "fit" fits into ---
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const read = () => {
      const cs = getComputedStyle(el)
      const pad = (s: string) => parseFloat(cs.getPropertyValue(`padding-${s}`)) || 0
      setStage({
        w: el.clientWidth - pad('left') - pad('right'),
        h: el.clientHeight - pad('top') - pad('bottom'),
      })
    }
    read()
    const ro = new ResizeObserver(read)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const onPageError = useCallback((n: number) => {
    setFailedPages((prev) => (prev.has(n) ? prev : new Set(prev).add(n)))
  }, [])

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

  // The keyboard listener below attaches once per document, so it cannot close over `fit`
  // without rebinding on every resize. A ref is how it reads the current one.
  const fitRef = useRef<number | null>(null)
  fitRef.current = fit

  // --- keys: Esc closes, arrows page, +/-/0 zoom, Tab stays inside ---
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return
      // Never steal keys from a text field: the API-key prompt can be stacked on top.
      // `closest?.` and not just `?.closest` - a keydown's target is not always an Element
      // (it is `document` when the event is dispatched there), and calling a method that
      // does not exist would throw inside the listener and take Escape down with it.
      const target = e.target as HTMLElement | null
      if (target?.closest?.('input, textarea, [contenteditable="true"]')) return
      // Only process our keys when this dialog is active: the API-key prompt can be
      // stacked above, and we must not steal its keys. Verify the active element is
      // inside our shell or the shell itself is focused.
      const active = document.activeElement
      const OURS = ['Escape', 'ArrowLeft', 'ArrowRight', 'Home', 'End', '+', '=', '-', '0']
      if (OURS.includes(e.key)) {
        if (!shellRef.current?.contains(active) && active !== shellRef.current) return
      }
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
      // Zoom. `=` as well as `+` because + is shifted on most layouts, and 0 resets to
      // fit rather than to 100% - fit is the mode, 100% is just a number inside it.
      if (e.key === '+' || e.key === '=') {
        e.preventDefault()
        setZoom((z) => zoomStep(z ?? fitRef.current ?? 1, 1))
        return
      }
      if (e.key === '-') {
        e.preventDefault()
        setZoom((z) => zoomStep(z ?? fitRef.current ?? 1, -1))
        return
      }
      if (e.key === '0') {
        e.preventDefault()
        setZoom(null)
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

  // Warm the neighbours so arrow-key browsing paints instantly. Pages are ~370 KB (as
  // measured from paligemma.pdf) and /images sets no Cache-Control, so even a revisit
  // costs a conditional round-trip. Only ±1: any more would compete with the page
  // actually on screen. Skipped while the PDF is showing - those bytes would never be
  // looked at.
  useEffect(() => {
    if (usePdf) return
    for (const d of [1, -1]) {
      const s = imageSrc(pages[idx + d]?.image)
      if (s) new Image().src = s
    }
  }, [pages, idx, usePdf])

  // Keep the active filmstrip thumb in view. `block: 'nearest'` because the strip is a
  // horizontal scroller inside a fixed-height shell - vertical movement is never wanted.
  useEffect(() => {
    curThumbRef.current?.scrollIntoView({ block: 'nearest', inline: 'center' })
  }, [idx, count])

  // --- zoom keeps its place ---
  //
  // Without this a zoom-in jumps to the top-left corner and the reader loses whatever they
  // were looking at, which is the one thing zoom exists to prevent. The centre of the
  // viewport is recorded as a fraction of the scrollable content before the scale changes
  // and restored after, so the page grows around what is on screen.
  const anchor = useRef<{ x: number; y: number } | null>(null)
  const prevScale = useRef<number | null>(null)
  useLayoutEffect(() => {
    const el = stageRef.current
    if (!el || !scale) return
    if (prevScale.current !== null && prevScale.current !== scale && anchor.current) {
      const { x, y } = anchor.current
      el.scrollLeft = x * el.scrollWidth - el.clientWidth / 2
      el.scrollTop = y * el.scrollHeight - el.clientHeight / 2
    }
    prevScale.current = scale
  }, [scale])

  const recordAnchor = () => {
    const el = stageRef.current
    if (!el || !el.scrollWidth || !el.scrollHeight) return
    anchor.current = {
      x: (el.scrollLeft + el.clientWidth / 2) / el.scrollWidth,
      y: (el.scrollTop + el.clientHeight / 2) / el.scrollHeight,
    }
  }

  const changeZoom = (next: number | null) => {
    recordAnchor()
    setZoom(next)
  }

  // ⌘/Ctrl + wheel zooms, which is the gesture a trackpad pinch sends. preventDefault or
  // the browser zooms its own chrome instead. A plain wheel is left alone - that is how
  // you scroll a page that is bigger than the stage.
  //
  // Native event listener with passive:false, because React's onWheel is passive by
  // default and preventDefault on a passive listener is silently ignored.
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      recordAnchor()
      setZoom((z) => clampZoom((z ?? fit ?? 1) * (e.deltaY < 0 ? 1.1 : 1 / 1.1)))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [fit])

  // --- drag to pan ---
  //
  // Scrolling, not a transform: the page's layout box has to stay the box the citation
  // overlay is drawn in. Drags that start inside the text layer are left alone, or
  // selecting a sentence would pan the page instead.
  const drag = useRef<{ x: number; y: number; left: number; top: number } | null>(null)
  const onPointerDown = (e: React.PointerEvent) => {
    const el = stageRef.current
    if (!el || e.button !== 0) return
    if ((e.target as HTMLElement).closest?.('.textLayer')) return
    if (el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight) return
    drag.current = { x: e.clientX, y: e.clientY, left: el.scrollLeft, top: el.scrollTop }
    el.setPointerCapture(e.pointerId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const el = stageRef.current
    if (!el || !drag.current) return
    el.scrollLeft = drag.current.left - (e.clientX - drag.current.x)
    el.scrollTop = drag.current.top - (e.clientY - drag.current.y)
  }
  const endDrag = (e: React.PointerEvent) => {
    if (drag.current) stageRef.current?.releasePointerCapture(e.pointerId)
    drag.current = null
  }

  const download = async () => {
    setDownloading(true)
    setDlError(null)
    try {
      // The bytes are usually already here - the viewer fetched them to render. Only a
      // document whose PDF never loaded pays for a second round-trip.
      saveBlob(blob ?? (await getDocumentFile(pdf)), pdf)
    } catch (e) {
      if (e instanceof UnauthorizedError) onAuthError()
      setDlError(e instanceof Error ? e.message : 'Download failed.')
    } finally {
      setDownloading(false)
    }
  }

  const overlays = page ? overlaysFor(regionsOnDocumentPage(regions, pdf, page.page_number)) : []
  // Same rule as the viewer: a single box gets the spotlight scrim, several can't compose it.
  const spotlight = overlays.length === 1
  const boxes = overlays.map((o, i) => (
    <div key={i} className={`box-overlay${spotlight ? '' : ' multi'}`} style={o} />
  ))

  const zoomPct = scale ? Math.round(scale * 100) : null

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
          {doc && !error && (pdfFailed || (page && failedPages.has(page.page_number))) && (
            <span className="doc-note" title="The source PDF is unavailable, so this is the stored page image.">
              page image
            </span>
          )}
          <span className="doc-head-gap" />
          {dlError && <span className="error-line">{dlError}</span>}
          {usePdf && (
            <span className="doc-zoom">
              <button
                className="doc-btn"
                onClick={() => changeZoom(zoomStep(scale ?? 1, -1))}
                aria-label="Zoom out"
                title="Zoom out (-)"
              >
                −
              </button>
              <button
                className="doc-btn zoom-reset"
                onClick={() => changeZoom(null)}
                title="Fit to window (0)"
              >
                {zoom === null ? 'fit' : `${zoomPct}%`}
              </button>
              <button
                className="doc-btn"
                onClick={() => changeZoom(zoomStep(scale ?? 1, 1))}
                aria-label="Zoom in"
                title="Zoom in (+)"
              >
                +
              </button>
            </span>
          )}
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

        {/* Two elements, not one: the arrows below are positioned against .doc-stage, and an
            absolutely positioned child of a scroll container scrolls away with its content.
            .doc-scroll is what pans; .doc-stage is what the arrows stay pinned to. */}
        <div className="doc-stage">
          <div
            className={`doc-scroll${zoom !== null ? ' zoomed' : ''}`}
            ref={stageRef}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
          >
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
            ) : usePdf && pdfDoc && page && pdfSize && scale ? (
              <PdfPageView
                doc={pdfDoc}
                pageNumber={page.page_number}
                scale={scale}
                size={{ w: pdfSize.w * scale, h: pdfSize.h * scale }}
                onError={onPageError}
              >
                {boxes}
              </PdfPageView>
            ) : src && page ? (
              /* The frame's aspect ratio is load-bearing, not decorative: without a definite
                 one the page is silently cropped and the citation box is drawn against the
                 wrong rectangle. pageFrameStyle owns that rule so it can be tested - see its
                 docstring in lib.ts. */
              <div className="doc-page" style={pageFrameStyle(natural)}>
                {/* No key on the <img>: a stable element identity lets the browser hold the
                    previous page painted until the new one decodes, which is a smoother swap
                    than any skeleton — a skeleton would blank the frame on every turn. */}
                <img {...imgProps} src={src} alt={`${pdf} page ${page.page_number}`} />
                {boxes}
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
          </div>

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
                {/* Full-resolution page PNGs stand in for thumbnails (1241x1754, ~370 KB mean,
                    as measured from paligemma.pdf) and a long document has 59 of them (e.g.,
                    paligemma.pdf), so lazy is what keeps opening one from costing 21 MB and
                    half a gigabyte of decoded bitmap. */}
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
